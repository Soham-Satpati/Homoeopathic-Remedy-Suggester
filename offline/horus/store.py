"""Encrypted local patient store (replaces the demo's GitHub-backed JSON file).

One SQLite file per clinic installation. Patient details, cases and intakes are sealed with
the clinic data key (see ``crypto.py``); only non-identifying metadata (record ids,
timestamps, status) is stored in the clear so lists can be sorted without decrypting.

Usage::

    store, recovery_key = Store.create(path, edition="clinic", clinic_name="…",
                                       admin_username="dr", admin_password="…")
    session = Store.open(path).login("dr", "…")
    pid = session.new_patient_id()
    session.save_patient_details(pid, {"name": "…"})
    session.add_case(pid, {...})

Roles
    admin   everything, including users, backup/restore, audit log and settings
    doctor  patients, cases, reports, intakes, backups
    staff   register patients, edit contact details, take intakes; cannot see cases/reports

Editions
    solo        one doctor account (admin role)
    clinic      several accounts on one PC
    clinic_lan  as clinic, plus the tablet intake server
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .crypto import (NONCE_LEN, Sealer, WrongSecret, new_data_key, new_recovery_key,
                     unwrap_with_password, unwrap_with_recovery, wrap_with_password,
                     wrap_with_recovery)

SCHEMA_VERSION = 1
EDITIONS = ("solo", "clinic", "clinic_lan")
ROLES = ("admin", "doctor", "staff")
MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 300
MIN_PASSWORD_LEN = 8
BACKUP_MAGIC = b"HORUSBK1"

PERMISSIONS = {
    "admin":  {"patients.read", "patients.write", "patients.delete", "cases.read", "cases.write",
               "intakes.read", "intakes.write", "intakes.review", "backup", "restore",
               "users.manage", "audit.read", "settings"},
    "doctor": {"patients.read", "patients.write", "patients.delete", "cases.read", "cases.write",
               "intakes.read", "intakes.write", "intakes.review", "backup"},
    "staff":  {"patients.read", "patients.write", "intakes.write"},
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin','doctor','staff')),
    wrapped_dek TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    must_change_password INTEGER NOT NULL DEFAULT 0,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    last_login TEXT
);
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    details BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    created_by INTEGER REFERENCES users(id),
    payload BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS cases_patient ON cases(patient_id, created_at);
CREATE TABLE IF NOT EXISTS intakes (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending','reviewed','dismissed')),
    reviewed_at TEXT,
    reviewed_by INTEGER REFERENCES users(id),
    payload BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS intakes_status ON intakes(status, created_at);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    user_id INTEGER,
    username TEXT,
    action TEXT NOT NULL,
    target TEXT,
    detail TEXT,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
"""


class StoreError(Exception):
    pass


class AuthError(StoreError):
    pass


class PermissionDenied(StoreError):
    pass


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _check_password(pw: str) -> None:
    if len(pw or "") < MIN_PASSWORD_LEN:
        raise StoreError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")


def _check_username(name: str) -> str:
    name = (name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{2,32}", name):
        raise StoreError("Username: 2–32 letters, digits, dot, dash or underscore.")
    return name


def _connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA synchronous = FULL")
    con.execute("PRAGMA secure_delete = ON")
    return con


class _Result:
    """Fully fetched query result, so no cursor outlives the lock."""

    def __init__(self, cur: sqlite3.Cursor):
        self.rows = cur.fetchall() if cur.description else []
        self.rowcount = cur.rowcount
        self.lastrowid = cur.lastrowid

    def __iter__(self):
        return iter(self.rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class _LockedConnection:
    """One SQLite connection shared by the window, background jobs and the tablet server.
    Every statement runs under the store's re-entrant lock; ``Session._tx`` holds the same lock
    for a whole transaction, so another thread's write can never land inside it."""

    def __init__(self, con: sqlite3.Connection, lock: threading.RLock):
        self._c = con
        self._lock = lock

    def execute(self, sql: str, params=()) -> _Result:
        with self._lock:
            return _Result(self._c.execute(sql, params))

    def executemany(self, sql: str, seq) -> _Result:
        with self._lock:
            return _Result(self._c.executemany(sql, seq))

    def backup(self, target: sqlite3.Connection) -> None:
        with self._lock:
            self._c.backup(target)

    def close(self) -> None:
        with self._lock:
            self._c.close()


@dataclass(frozen=True)
class User:
    id: int
    username: str
    display_name: str
    role: str
    must_change_password: bool = False

    def can(self, perm: str) -> bool:
        return perm in PERMISSIONS.get(self.role, set())


class Store:
    """An opened (still locked) clinic database. ``login`` returns a ``Session``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise StoreError(f"No HoRUS database at {self.path}")
        self._lock = threading.RLock()
        self._con = _LockedConnection(_connect(self.path), self._lock)
        v = int(self._meta("schema_version", "0"))
        if v > SCHEMA_VERSION:
            raise StoreError("This database was created by a newer version of HoRUS.")

    # ── creation ─────────────────────────────────────────────────────────
    @classmethod
    def create(cls, path: Path, *, edition: str, clinic_name: str, admin_username: str,
               admin_password: str, admin_display_name: str = "") -> tuple["Store", str]:
        """New database. Returns (store, recovery_key) — the key is shown to the user once."""
        if edition not in EDITIONS:
            raise StoreError(f"Unknown edition {edition!r}")
        username = _check_username(admin_username)
        _check_password(admin_password)
        path = Path(path)
        if path.exists():
            raise StoreError(f"A database already exists at {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        dek = new_data_key()
        recovery = new_recovery_key()
        con = _connect(path)
        try:
            con.executescript(_SCHEMA)
            meta = {"schema_version": str(SCHEMA_VERSION), "db_id": uuid.uuid4().hex,
                    "edition": edition, "clinic_name": clinic_name.strip() or "My clinic",
                    "created_at": _now(), "recovery_wrap": json.dumps(wrap_with_recovery(dek, recovery))}
            con.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", meta.items())
            con.execute(
                "INSERT INTO users(username, display_name, role, wrapped_dek, created_at) VALUES (?,?,?,?,?)",
                (username, admin_display_name.strip() or username, "admin",
                 json.dumps(wrap_with_password(dek, admin_password)), _now()))
        finally:
            con.close()
        store = cls(path)
        store._audit(None, username, "database.created", detail=f"edition={edition}")
        return store, recovery

    # ── metadata ─────────────────────────────────────────────────────────
    def _meta(self, key: str, default: str | None = None) -> str | None:
        try:
            row = self._con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        except sqlite3.OperationalError:
            return default
        return row["value"] if row else default

    @property
    def edition(self) -> str:
        return self._meta("edition", "solo")

    @property
    def clinic_name(self) -> str:
        return self._meta("clinic_name", "")

    @property
    def db_id(self) -> str:
        return self._meta("db_id", "")

    def usernames(self) -> list[str]:
        return [r["username"] for r in self._con.execute(
            "SELECT username FROM users WHERE active = 1 ORDER BY username")]

    # ── authentication ───────────────────────────────────────────────────
    def login(self, username: str, password: str) -> "Session":
        with self._lock:
            row = self._con.execute("SELECT * FROM users WHERE username = ?", (username.strip(),)).fetchone()
            if row is None or not row["active"]:
                self._audit(None, username, "login.failed", detail="unknown or inactive user")
                raise AuthError("Wrong username or password.")
            if row["locked_until"] > time.time():
                wait = int(row["locked_until"] - time.time()) + 1
                raise AuthError(f"Too many failed attempts. Try again in {wait} seconds.")
            try:
                dek = unwrap_with_password(json.loads(row["wrapped_dek"]), password)
            except WrongSecret:
                n = row["failed_attempts"] + 1
                locked = time.time() + LOCKOUT_SECONDS if n >= MAX_FAILED_LOGINS else 0
                self._con.execute("UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
                                  (0 if locked else n, locked, row["id"]))
                self._audit(row["id"], row["username"], "login.failed",
                            detail="locked" if locked else f"attempt {n}")
                raise AuthError("Wrong username or password.") from None
            self._con.execute("UPDATE users SET failed_attempts = 0, locked_until = 0, last_login = ? "
                              "WHERE id = ?", (_now(), row["id"]))
            user = User(row["id"], row["username"], row["display_name"], row["role"],
                        bool(row["must_change_password"]))
            session = Session(self, user, dek)
            session.audit("login")
            return session

    def recover(self, recovery_key: str, username: str, new_password: str) -> "Session":
        """Forgotten password: the recovery key resets an admin's password."""
        _check_password(new_password)
        with self._lock:
            try:
                dek = unwrap_with_recovery(json.loads(self._meta("recovery_wrap")), recovery_key)
            except WrongSecret:
                self._audit(None, username, "recovery.failed")
                raise AuthError("That recovery key is not correct.") from None
            row = self._con.execute("SELECT * FROM users WHERE username = ? AND role = 'admin'",
                                    (username.strip(),)).fetchone()
            if row is None:
                raise AuthError("Recovery resets an administrator account; that username is not one.")
            self._con.execute(
                "UPDATE users SET wrapped_dek = ?, active = 1, failed_attempts = 0, locked_until = 0, "
                "must_change_password = 0 WHERE id = ?",
                (json.dumps(wrap_with_password(dek, new_password)), row["id"]))
            self._audit(row["id"], row["username"], "recovery.password_reset")
        return self.login(username, new_password)

    # ── audit log (hash-chained) ─────────────────────────────────────────
    def _audit(self, user_id, username, action: str, target: str | None = None,
               detail: str | None = None) -> None:
        """Append an entry. Entries hold ids and actions only — never patient details."""
        with self._lock:
            last = self._con.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
            prev = last["hash"] if last else "0" * 64
            ts = datetime.now().isoformat(timespec="milliseconds")
            h = _audit_hash(prev, ts, user_id, username, action, target, detail)
            cur = self._con.execute(
                "INSERT INTO audit(ts, user_id, username, action, target, detail, prev_hash, hash) "
                "VALUES (?,?,?,?,?,?,?,?)", (ts, user_id, username, action, target, detail, prev, h))
            return cur.lastrowid, h

    def _checkpoint(self, sealer: Sealer, entry_id: int, entry_hash: str) -> None:
        """Keyed checkpoint of the newest entry. Without the clinic key nobody can move it, so
        deleting entries from the end (or rebuilding the chain) is detected."""
        mac = hmac.new(sealer.subkey("audit"), f"{entry_id}|{entry_hash}".encode(), "sha256").hexdigest()
        self._con.execute("INSERT INTO meta(key, value) VALUES ('audit_head', ?) "
                          "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                          (json.dumps({"id": entry_id, "hash": entry_hash, "mac": mac}),))

    def verify_audit_chain(self, sealer: Sealer | None = None) -> tuple[bool, int | None]:
        """(ok, first broken entry id). The hash chain detects edited, deleted or reordered
        entries; with ``sealer`` the keyed checkpoint also detects entries cut from the end."""
        prev = "0" * 64
        hashes = {}
        last_id = None
        for r in self._con.execute("SELECT * FROM audit ORDER BY id"):
            h = _audit_hash(prev, r["ts"], r["user_id"], r["username"], r["action"], r["target"], r["detail"])
            if r["prev_hash"] != prev or r["hash"] != h:
                return False, r["id"]
            prev = hashes[r["id"]] = r["hash"]
            last_id = r["id"]
        if sealer is not None:
            raw = self._meta("audit_head")
            if raw:
                head = json.loads(raw)
                mac = hmac.new(sealer.subkey("audit"), f"{head['id']}|{head['hash']}".encode(),
                               "sha256").hexdigest()
                if not hmac.compare_digest(mac, head["mac"]) or hashes.get(head["id"]) != head["hash"]:
                    return False, (last_id or 0) + 1
        return True, None

    def close(self) -> None:
        with self._lock:
            self._con.close()

    # ── backup / restore ─────────────────────────────────────────────────
    @staticmethod
    def restore(backup_path: Path, target_path: Path, username: str, secret: str,
                use_recovery_key: bool = False) -> "Store":
        """Restore a backup to ``target_path`` (which must not exist), proving access first."""
        target_path = Path(target_path)
        if target_path.exists():
            raise StoreError(f"{target_path} already exists — move it aside before restoring.")
        header, nonce, ct = _read_backup(Path(backup_path))
        dek = None
        if use_recovery_key:
            dek = unwrap_with_recovery(header["recovery_wrap"], secret)
        else:
            for w in header["users"]:
                if w["username"].lower() == username.strip().lower():
                    dek = unwrap_with_password(w["wrap"], secret)
                    break
            if dek is None:
                raise AuthError("That user is not in this backup.")
        key = Sealer(dek).subkey("backup")
        data = AESGCM(key).decrypt(nonce, ct, BACKUP_MAGIC + header["db_id"].encode())
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = target_path.with_suffix(".restoring")
        tmp.write_bytes(data)
        os.replace(tmp, target_path)
        store = Store(target_path)
        store._audit(None, username, "backup.restored", detail=f"from {Path(backup_path).name}")
        return store


def _audit_hash(prev, ts, user_id, username, action, target, detail) -> str:
    material = json.dumps([prev, ts, user_id, username, action, target, detail], ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _read_backup(path: Path) -> tuple[dict, bytes, bytes]:
    raw = path.read_bytes()
    if not raw.startswith(BACKUP_MAGIC):
        raise StoreError("Not a HoRUS backup file.")
    hlen = int.from_bytes(raw[8:12], "big")
    header = json.loads(raw[12:12 + hlen].decode("utf-8"))
    body = raw[12 + hlen:]
    return header, body[:NONCE_LEN], body[NONCE_LEN:]


class Session:
    """A logged-in user. Holds the data key in memory until ``logout``."""

    def __init__(self, store: Store, user: User, dek: bytes):
        self.store = store
        self.user = user
        self._sealer: Sealer | None = Sealer(dek)

    # ── plumbing ─────────────────────────────────────────────────────────
    @property
    def _con(self) -> sqlite3.Connection:
        return self.store._con

    @property
    def sealer(self) -> Sealer:
        if self._sealer is None:
            raise AuthError("Session has ended — please sign in again.")
        return self._sealer

    def require(self, perm: str) -> None:
        if self._sealer is None:
            raise AuthError("Session has ended — please sign in again.")
        if not self.user.can(perm):
            raise PermissionDenied(f"Your role ({self.user.role}) cannot do that.")

    def audit(self, action: str, target: str | None = None, detail: str | None = None) -> None:
        with self.store._lock:
            eid, h = self.store._audit(self.user.id, self.user.username, action, target, detail)
            if self._sealer is not None:
                self.store._checkpoint(self._sealer, eid, h)

    def verify_audit(self) -> tuple[bool, int | None]:
        self.require("audit.read")
        return self.store.verify_audit_chain(self.sealer)

    @contextmanager
    def _tx(self):
        with self.store._lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
            self._con.execute("COMMIT")

    def logout(self) -> None:
        if self._sealer is not None:
            self.audit("logout")
        self._sealer = None

    # ── own account ──────────────────────────────────────────────────────
    def verify_password(self, password: str, purpose: str = "unlock") -> bool:
        """Re-check the signed-in user's password (screen unlock, leaving kiosk mode)."""
        row = self._con.execute("SELECT wrapped_dek FROM users WHERE id = ?", (self.user.id,)).fetchone()
        try:
            unwrap_with_password(json.loads(row["wrapped_dek"]), password)
        except WrongSecret:
            self.audit(f"{purpose}.failed")
            return False
        self.audit(purpose)
        return True

    def change_password(self, old: str, new: str) -> None:
        _check_password(new)
        row = self._con.execute("SELECT wrapped_dek FROM users WHERE id = ?", (self.user.id,)).fetchone()
        try:
            unwrap_with_password(json.loads(row["wrapped_dek"]), old)
        except WrongSecret:
            raise AuthError("Current password is wrong.") from None
        self._con.execute("UPDATE users SET wrapped_dek = ?, must_change_password = 0 WHERE id = ?",
                          (json.dumps(wrap_with_password(self.sealer.dek, new)), self.user.id))
        self.user = User(self.user.id, self.user.username, self.user.display_name, self.user.role, False)
        self.audit("password.changed")

    # ── users (admin) ────────────────────────────────────────────────────
    def list_users(self) -> list[dict]:
        self.require("users.manage")
        return [dict(id=r["id"], username=r["username"], display_name=r["display_name"], role=r["role"],
                     active=bool(r["active"]), last_login=r["last_login"])
                for r in self._con.execute("SELECT * FROM users ORDER BY username")]

    def add_user(self, username: str, display_name: str, role: str, temp_password: str) -> int:
        self.require("users.manage")
        if self.store.edition == "solo":
            raise StoreError("The Solo edition has one account. Switch to the Clinic edition to add users.")
        if role not in ROLES:
            raise StoreError(f"Unknown role {role!r}")
        username = _check_username(username)
        _check_password(temp_password)
        try:
            cur = self._con.execute(
                "INSERT INTO users(username, display_name, role, wrapped_dek, must_change_password, created_at) "
                "VALUES (?,?,?,?,1,?)",
                (username, display_name.strip() or username, role,
                 json.dumps(wrap_with_password(self.sealer.dek, temp_password)), _now()))
        except sqlite3.IntegrityError:
            raise StoreError(f"A user called {username!r} already exists.") from None
        self.audit("user.added", username, f"role={role}")
        return cur.lastrowid

    def reset_user_password(self, user_id: int, temp_password: str) -> None:
        self.require("users.manage")
        _check_password(temp_password)
        self._con.execute(
            "UPDATE users SET wrapped_dek = ?, must_change_password = 1, failed_attempts = 0, "
            "locked_until = 0 WHERE id = ?",
            (json.dumps(wrap_with_password(self.sealer.dek, temp_password)), user_id))
        self.audit("user.password_reset", str(user_id))

    def set_user_role(self, user_id: int, role: str) -> None:
        self.require("users.manage")
        if role not in ROLES:
            raise StoreError(f"Unknown role {role!r}")
        if user_id == self.user.id and role != "admin":
            raise StoreError("You cannot remove your own administrator role.")
        self._con.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
        self.audit("user.role_changed", str(user_id), f"role={role}")

    def set_user_active(self, user_id: int, active: bool) -> None:
        self.require("users.manage")
        if user_id == self.user.id and not active:
            raise StoreError("You cannot deactivate your own account.")
        if not active:
            admins = self._con.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND active=1 "
                                       "AND id != ?", (user_id,)).fetchone()[0]
            if admins == 0:
                raise StoreError("At least one active administrator is required.")
        self._con.execute("UPDATE users SET active = ? WHERE id = ?", (1 if active else 0, user_id))
        self.audit("user.activated" if active else "user.deactivated", str(user_id))

    def set_edition(self, edition: str) -> None:
        self.require("settings")
        if edition not in EDITIONS:
            raise StoreError(f"Unknown edition {edition!r}")
        if edition == "solo":
            n = self._con.execute("SELECT COUNT(*) FROM users WHERE active = 1").fetchone()[0]
            if n > 1:
                raise StoreError("Deactivate the other accounts before switching to Solo.")
        self._con.execute("UPDATE meta SET value = ? WHERE key = 'edition'", (edition,))
        self.audit("settings.edition", detail=edition)

    def set_clinic_name(self, name: str) -> None:
        self.require("settings")
        self._con.execute("UPDATE meta SET value = ? WHERE key = 'clinic_name'", (name.strip(),))
        self.audit("settings.clinic_name")

    def rotate_recovery_key(self) -> str:
        """Issue a new recovery key (the old one stops working)."""
        self.require("settings")
        key = new_recovery_key()
        self._con.execute("UPDATE meta SET value = ? WHERE key = 'recovery_wrap'",
                          (json.dumps(wrap_with_recovery(self.sealer.dek, key)),))
        self.audit("recovery.rotated")
        return key

    # ── patients ─────────────────────────────────────────────────────────
    def new_patient_id(self, year: int | None = None) -> str:
        year = year or datetime.now().year
        prefix = f"PT-{year}-"
        nums = [int(r["id"].split("-")[-1]) for r in
                self._con.execute("SELECT id FROM patients WHERE id LIKE ?", (prefix + "%",))
                if r["id"].split("-")[-1].isdigit()]
        return f"{prefix}{(max(nums, default=0) + 1):03d}"

    def create_patient(self, details: dict, patient_id: str | None = None) -> str:
        self.require("patients.write")
        with self._tx():
            pid = patient_id or self.new_patient_id()
            now = _now()
            self._con.execute("INSERT INTO patients(id, created_at, updated_at, details) VALUES (?,?,?,?)",
                              (pid, now, now, self.sealer.seal(_clean_details(details), f"patients:{pid}")))
        self.audit("patient.created", pid)
        return pid

    def save_patient_details(self, pid: str, details: dict) -> None:
        self.require("patients.write")
        with self._tx():
            if not self._con.execute("SELECT 1 FROM patients WHERE id = ?", (pid,)).fetchone():
                raise StoreError(f"No patient {pid}")
            self._con.execute("UPDATE patients SET details = ?, updated_at = ? WHERE id = ?",
                              (self.sealer.seal(_clean_details(details), f"patients:{pid}"), _now(), pid))
        self.audit("patient.updated", pid)

    def get_patient(self, pid: str) -> dict | None:
        self.require("patients.read")
        r = self._con.execute("SELECT * FROM patients WHERE id = ?", (pid,)).fetchone()
        if r is None:
            return None
        self.audit("patient.viewed", pid)
        return {"id": r["id"], "created_at": r["created_at"], "updated_at": r["updated_at"],
                "details": self.sealer.open(r["details"], f"patients:{pid}")}

    def list_patients(self, query: str = "") -> list[dict]:
        """All patients with details and last-visit summary; ``query`` matches id/name/phone."""
        self.require("patients.read")
        q = (query or "").strip().lower()
        can_cases = self.user.can("cases.read")
        last = {}
        if can_cases:
            for r in self._con.execute("SELECT patient_id, MAX(created_at) AS last, COUNT(*) AS n "
                                       "FROM cases GROUP BY patient_id"):
                last[r["patient_id"]] = (r["last"], r["n"])
        pending = {r["patient_id"]: r["n"] for r in self._con.execute(
            "SELECT patient_id, COUNT(*) AS n FROM intakes WHERE status = 'pending' GROUP BY patient_id")}
        out = []
        for r in self._con.execute("SELECT * FROM patients ORDER BY updated_at DESC"):
            d = self.sealer.open(r["details"], f"patients:{r['id']}")
            if q and q not in r["id"].lower() and q not in str(d.get("name", "")).lower() \
                    and q not in str(d.get("contact", "")).lower():
                continue
            lv, n = last.get(r["id"], (None, 0))
            out.append({"id": r["id"], "details": d, "updated_at": r["updated_at"],
                        "last_case": lv, "case_count": n, "pending_intakes": pending.get(r["id"], 0)})
        return out

    def delete_patient(self, pid: str) -> None:
        self.require("patients.delete")
        with self._tx():
            n = self._con.execute("DELETE FROM patients WHERE id = ?", (pid,)).rowcount
        if n:
            self.audit("patient.deleted", pid)

    # ── cases ────────────────────────────────────────────────────────────
    def add_case(self, pid: str, case: dict, timestamp: str | None = None) -> int:
        """``case`` follows the demo's shape: raw_symptoms, categorised, top_remedies, clinical_notes,
        patient_details, analysis_mode — plus optionally report / enhanced data."""
        self.require("cases.write")
        with self._tx():
            if not self._con.execute("SELECT 1 FROM patients WHERE id = ?", (pid,)).fetchone():
                raise StoreError(f"No patient {pid}")
            ts = timestamp or _now()
            cur = self._con.execute("INSERT INTO cases(patient_id, created_at, created_by, payload) "
                                    "VALUES (?,?,?,x'')", (pid, ts, self.user.id))
            cid = cur.lastrowid
            self._con.execute("UPDATE cases SET payload = ? WHERE id = ?",
                              (self.sealer.seal(case, f"cases:{cid}"), cid))
            self._con.execute("UPDATE patients SET updated_at = ? WHERE id = ?", (_now(), pid))
        self.audit("case.created", pid, f"case={cid}")
        return cid

    def update_case(self, case_id: int, changes: dict) -> None:
        """Merge ``changes`` into a saved case (e.g. AI explanations that arrived after saving)."""
        self.require("cases.write")
        with self._tx():
            r = self._con.execute("SELECT patient_id, payload FROM cases WHERE id = ?", (case_id,)).fetchone()
            if r is None:
                raise StoreError(f"No case {case_id}")
            payload = {**self.sealer.open(r["payload"], f"cases:{case_id}"), **changes}
            self._con.execute("UPDATE cases SET payload = ? WHERE id = ?",
                              (self.sealer.seal(payload, f"cases:{case_id}"), case_id))
        self.audit("case.updated", r["patient_id"], f"case={case_id}")

    def list_cases(self, pid: str) -> list[dict]:
        self.require("cases.read")
        rows = self._con.execute("SELECT c.*, u.display_name AS author FROM cases c LEFT JOIN users u "
                                 "ON u.id = c.created_by WHERE patient_id = ? ORDER BY created_at", (pid,))
        out = [{"id": r["id"], "timestamp": r["created_at"], "author": r["author"],
                **self.sealer.open(r["payload"], f"cases:{r['id']}")} for r in rows]
        self.audit("cases.viewed", pid)
        return out

    def learning_records(self) -> dict:
        """Every saved case in the demo's ``{pid: {"cases": [...]}}`` shape, for engine.learn_from."""
        self.require("cases.read")
        out: dict = {}
        rows = self._con.execute("SELECT c.id, c.patient_id, c.payload FROM cases c JOIN patients p "
                                 "ON p.id = c.patient_id ORDER BY p.rowid, c.created_at, c.id")
        for r in rows:
            c = self.sealer.open(r["payload"], f"cases:{r['id']}")
            out.setdefault(r["patient_id"], {"cases": []})["cases"].append(
                {"categorised": c.get("categorised") or {}, "top_remedies": c.get("top_remedies") or []})
        return out

    # ── intakes ──────────────────────────────────────────────────────────
    def add_intake(self, pid: str, intake: dict, source: str = "kiosk") -> str:
        """``intake``: {qa: [...], raw_text: str, details: {...}}. Merges non-empty details."""
        self.require("intakes.write")
        iid = uuid.uuid4().hex[:8]
        with self._tx():
            r = self._con.execute("SELECT details FROM patients WHERE id = ?", (pid,)).fetchone()
            if r is None:
                raise StoreError(f"No patient {pid}")
            details = self.sealer.open(r["details"], f"patients:{pid}")
            details.update({k: v for k, v in (intake.get("details") or {}).items() if v})
            self._con.execute("UPDATE patients SET details = ?, updated_at = ? WHERE id = ?",
                              (self.sealer.seal(_clean_details(details), f"patients:{pid}"), _now(), pid))
            self._con.execute("INSERT INTO intakes(id, patient_id, created_at, source, status, payload) "
                              "VALUES (?,?,?,?, 'pending', ?)",
                              (iid, pid, _now(), source, self.sealer.seal(intake, f"intakes:{iid}")))
        self.audit("intake.created", pid, f"intake={iid} source={source}")
        return iid

    def pending_intakes(self) -> list[dict]:
        self.require("intakes.read")
        out = []
        for r in self._con.execute("SELECT i.*, p.details FROM intakes i JOIN patients p ON p.id = i.patient_id "
                                   "WHERE status = 'pending' ORDER BY i.created_at DESC"):
            out.append({"id": r["id"], "patient_id": r["patient_id"], "timestamp": r["created_at"],
                        "source": r["source"],
                        "details": self.sealer.open(r["details"], f"patients:{r['patient_id']}"),
                        **self.sealer.open(r["payload"], f"intakes:{r['id']}")})
        return out

    def patient_intakes(self, pid: str) -> list[dict]:
        self.require("intakes.read")
        return [{"id": r["id"], "timestamp": r["created_at"], "status": r["status"], "source": r["source"],
                 **self.sealer.open(r["payload"], f"intakes:{r['id']}")}
                for r in self._con.execute("SELECT * FROM intakes WHERE patient_id = ? ORDER BY created_at DESC",
                                           (pid,))]

    def mark_intake(self, intake_id: str, status: str) -> None:
        self.require("intakes.review")
        if status not in ("reviewed", "dismissed", "pending"):
            raise StoreError(f"Bad status {status!r}")
        self._con.execute("UPDATE intakes SET status = ?, reviewed_at = ?, reviewed_by = ? WHERE id = ?",
                          (status, _now(), self.user.id, intake_id))
        self.audit("intake." + status, intake_id)

    # ── audit (admin) ────────────────────────────────────────────────────
    def audit_entries(self, limit: int = 500) -> list[dict]:
        self.require("audit.read")
        return [dict(r) for r in self._con.execute("SELECT id, ts, username, action, target, detail FROM audit "
                                                   "ORDER BY id DESC LIMIT ?", (limit,))]

    # ── backup ───────────────────────────────────────────────────────────
    def backup(self, dest: Path) -> Path:
        """Write a sealed backup. Restoring needs a user's password or the recovery key."""
        self.require("backup")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            snap = Path(td) / "snapshot.db"
            with self.store._lock:
                target = sqlite3.connect(str(snap))
                self._con.backup(target)
                target.close()
            data = snap.read_bytes()
        users = [{"username": r["username"], "wrap": json.loads(r["wrapped_dek"])}
                 for r in self._con.execute("SELECT username, wrapped_dek FROM users WHERE active = 1")]
        header = {"format": 1, "db_id": self.store.db_id, "created_at": _now(),
                  "clinic_name": self.store.clinic_name, "users": users,
                  "recovery_wrap": json.loads(self.store._meta("recovery_wrap"))}
        hbytes = json.dumps(header).encode("utf-8")
        nonce = os.urandom(NONCE_LEN)
        ct = AESGCM(self.sealer.subkey("backup")).encrypt(nonce, data, BACKUP_MAGIC + header["db_id"].encode())
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(BACKUP_MAGIC + len(hbytes).to_bytes(4, "big") + hbytes + nonce + ct)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
        self.audit("backup.created", detail=dest.name)
        return dest

    # ── import from the demo ─────────────────────────────────────────────
    def import_demo_patients(self, data: dict) -> dict:
        """Import a demo ``patients.json`` (GitHub file). Existing patient ids are merged:
        new cases/intakes are added, identical ones (same timestamp) are skipped."""
        self.require("cases.write")
        stats = {"patients_new": 0, "patients_merged": 0, "cases": 0, "intakes": 0, "skipped": 0}
        for pid, rec in data.items():
            if not isinstance(pid, str) or not pid.strip():
                continue
            if isinstance(rec, list):
                rec = {"details": {}, "cases": rec}
            details = rec.get("details") or {}
            exists = self._con.execute("SELECT 1 FROM patients WHERE id = ?", (pid,)).fetchone()
            if exists:
                stats["patients_merged"] += 1
                seen = {c["timestamp"] for c in self.list_cases(pid)}
            else:
                first_case_details = next((c.get("patient_details") for c in rec.get("cases", [])
                                           if c.get("patient_details")), {}) or {}
                self.create_patient({**first_case_details, **details}, patient_id=pid)
                stats["patients_new"] += 1
                seen = set()
            for c in rec.get("cases", []):
                ts = _demo_ts(c.get("timestamp"))
                if ts in seen:
                    stats["skipped"] += 1
                    continue
                self.add_case(pid, {k: v for k, v in c.items() if k != "timestamp"}, timestamp=ts)
                stats["cases"] += 1
            for it in rec.get("intakes", []):
                payload = {k: v for k, v in it.items() if k not in ("id", "status", "timestamp", "reviewed_at")}
                iid = self.add_intake(pid, payload, source="import")
                if it.get("status") in ("reviewed", "dismissed"):
                    self._con.execute("UPDATE intakes SET status = ?, created_at = ? WHERE id = ?",
                                      (it["status"], _demo_ts(it.get("timestamp")), iid))
                stats["intakes"] += 1
        self.audit("import.demo", detail=json.dumps(stats))
        return stats


def _clean_details(d: dict) -> dict:
    keep = ("name", "age", "gender", "contact", "address", "notes", "prev_id")
    return {k: str(d.get(k, "")).strip() for k in keep if str(d.get(k, "")).strip()}


def _demo_ts(ts) -> str:
    """Demo timestamps are 'YYYY-MM-DD HH:MM'; store ISO."""
    if not ts:
        return _now()
    try:
        return datetime.strptime(str(ts), "%Y-%m-%d %H:%M").isoformat(timespec="seconds")
    except ValueError:
        return str(ts)


def default_db_path(user_dir: Path) -> Path:
    return Path(user_dir) / "data" / "horus.db"


def iter_backups(folder: Path) -> Iterable[Path]:
    return sorted(Path(folder).glob("*.horusbak"), reverse=True)
