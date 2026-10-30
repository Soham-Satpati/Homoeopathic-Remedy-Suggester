"""Clinic LAN tablet server: TLS, pairing, lockout, write-only intake, input limits."""

from __future__ import annotations

import http.cookiejar
import json
import socket
import ssl
import urllib.error
import urllib.request

import pytest

pytest.importorskip("PySide6")
from horus_app.lan import IntakeServer, ensure_certificate  # noqa: E402


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path):
    got = []

    def submit(details, qa, raw):
        got.append((details, qa, raw))
        return "PT-2026-042"
    srv = IntakeServer(submit, tmp_path / "lan", port=_port())
    srv.start()
    yield srv, got
    srv.stop()


def _client():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx),
                                       urllib.request.HTTPCookieProcessor(jar),
                                       urllib.request.ProxyHandler({})), jar


def _post(op, srv, path, body, raw: bytes | None = None):
    data = raw if raw is not None else json.dumps(body).encode()
    req = urllib.request.Request(f"https://127.0.0.1:{srv.port}{path}", data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with op.open(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"{}"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), dict(e.headers)


def test_requires_pairing_and_accepts_intake(server):
    srv, got = server
    op, jar = _client()
    assert _post(op, srv, "/api/intake", {"details": {"name": "A"}, "answers": {}})[0] == 401
    code, _, headers = _post(op, srv, "/api/pair", {"pin": srv.pin})
    assert code == 200
    cookie = headers.get("Set-Cookie", "")
    assert "HttpOnly" in cookie and "Secure" in cookie and "SameSite=Strict" in cookie
    code, body, _ = _post(op, srv, "/api/intake",
                          {"details": {"name": "Rina", "age": "30"}, "answers": {"complaint": "headache"}})
    assert code == 200 and body["patient_id"] == "PT-2026-042"
    details, qa, raw = got[0]
    assert details["name"] == "Rina" and "Main complaint: headache" in raw
    assert qa[0]["answer"] == "headache"


def test_wrong_pins_lock_the_device(server):
    srv, _ = server
    op, _ = _client()
    wrong = "000000" if srv.pin != "000000" else "111111"
    for _ in range(5):
        assert _post(op, srv, "/api/pair", {"pin": wrong})[0] == 403
    assert _post(op, srv, "/api/pair", {"pin": srv.pin})[0] == 403        # locked even with the right PIN


def test_new_pin_revokes_sessions(server):
    srv, _ = server
    op, _ = _client()
    _post(op, srv, "/api/pair", {"pin": srv.pin})
    srv.new_pin()
    assert _post(op, srv, "/api/intake", {"details": {"name": "A"}, "answers": {}})[0] == 401


def test_limits_and_no_read_access(server):
    srv, got = server
    op, _ = _client()
    _post(op, srv, "/api/pair", {"pin": srv.pin})
    assert _post(op, srv, "/api/intake", None, raw=b"x" * (70 * 1024))[0] == 400      # oversized body
    assert _post(op, srv, "/api/intake", None, raw=b"{not json")[0] == 400
    assert _post(op, srv, "/api/intake", {"details": {"name": " "}, "answers": {}})[0] == 400
    _post(op, srv, "/api/intake", {"details": {"name": "B" * 5000}, "answers": {"complaint": "y" * 50000}})
    details, qa, _ = got[-1]
    assert len(details["name"]) == 200 and len(qa[0]["answer"]) == 4000
    req = urllib.request.Request(f"https://127.0.0.1:{srv.port}/api/patients")
    with pytest.raises(urllib.error.HTTPError) as e:
        op.open(req, timeout=10)
    assert e.value.code == 404


def test_page_has_strict_headers(server):
    srv, _ = server
    op, _ = _client()
    with op.open(f"https://127.0.0.1:{srv.port}/", timeout=10) as r:
        html = r.read().decode()
        h = dict(r.headers)
    assert "Patient questionnaire" in html and "default-src 'none'" in h["Content-Security-Policy"]
    assert h["X-Frame-Options"] == "DENY" and "no-store" in h["Cache-Control"]
    assert "http://" not in html.replace("http://www.w3.org", "")                    # no external resources


def test_certificate_reused_until_ips_change(tmp_path):
    c1, k1, f1 = ensure_certificate(tmp_path, ["192.168.1.10"])
    _, _, f2 = ensure_certificate(tmp_path, ["192.168.1.10"])
    _, _, f3 = ensure_certificate(tmp_path, ["192.168.1.11"])
    assert f1 == f2 != f3


def test_tablet_shows_three_questions_per_screen(server):
    """Drive the tablet page in a real browser: pair, details, 4 screens of ≤3 questions, submit."""
    pw = pytest.importorskip("playwright.sync_api")
    from horus.intake import PATIENT_QUESTIONS
    srv, got = server
    try:
        p = pw.sync_playwright().start()
        browser = p.chromium.launch()
    except Exception as e:                       # no browser on this machine
        pytest.skip(f"Chromium not available: {e}")
    try:
        page = browser.new_page(ignore_https_errors=True)
        page.goto(f"https://127.0.0.1:{srv.port}/")
        page.fill("#pin", srv.pin)
        page.click("text=Continue")
        page.fill("#name", "Tablet Patient")
        page.click("text=Start questions")
        screens = 0
        while page.is_visible("#qs"):
            boxes = page.locator("#qlist textarea")
            n = boxes.count()
            assert 1 <= n <= 3
            for k in range(n):
                boxes.nth(k).fill(f"answer {screens}-{k}")
            screens += 1
            page.click("#nextb")
            page.wait_for_timeout(150)
        assert screens == 4 and page.is_visible("#done")
        # going back keeps answers
    finally:
        browser.close()
        p.stop()
    details, qa, narrative = got[-1] if isinstance(got[-1], tuple) else (None, got[-1], None)
    answered = [x for x in (qa or []) if x.get("answer")] if isinstance(qa, list) else []
    assert len(answered) == len(PATIENT_QUESTIONS)
