"""Clinic LAN edition: tablets on the clinic Wi-Fi submit intake questionnaires to this PC.

No internet is involved: the tablet talks only to this computer over the local network.

Security
* HTTPS with a certificate generated on this PC for its LAN addresses (the tablet is asked
  to trust it once).
* Pairing: a 6-digit PIN shown in HoRUS. A paired tablet gets a random session cookie
  (HttpOnly, Secure, SameSite=Strict) valid for 12 hours. Wrong PINs lock that device out
  for 5 minutes after 5 attempts.
* Write-only: the only action is "submit a questionnaire". Nothing can be read back.
* Small request bodies only; all text is length-capped; strict security headers, no external
  resources.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import json
import logging
import secrets
import socket
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from horus.intake import PATIENT_QUESTIONS, compose_patient_narrative

log = logging.getLogger(__name__)
SESSION_HOURS = 12
MAX_BODY = 64 * 1024
MAX_FIELD = 4000


def lan_addresses() -> list[str]:
    ips = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))          # no packet is sent; picks the LAN interface
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127.") and ipaddress.ip_address(ip).is_private)


def ensure_certificate(folder: Path, ips: list[str]) -> tuple[Path, Path, str]:
    """Self-signed cert for this PC's LAN IPs. Returns (cert, key, SHA-256 fingerprint)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    folder.mkdir(parents=True, exist_ok=True)
    cert_p, key_p, ips_p = folder / "lan-cert.pem", folder / "lan-key.pem", folder / "lan-ips.json"
    if cert_p.exists() and key_p.exists() and ips_p.exists() and json.loads(ips_p.read_text()) == ips:
        cert = x509.load_pem_x509_certificate(cert_p.read_bytes())
        if cert.not_valid_after_utc > dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=30):
            return cert_p, key_p, cert.fingerprint(hashes.SHA256()).hex(":").upper()
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "HoRUS clinic PC")])
    san = [x509.DNSName("localhost")] + [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=825))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    key_p.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                        serialization.NoEncryption()))
    cert_p.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    ips_p.write_text(json.dumps(ips))
    return cert_p, key_p, cert.fingerprint(hashes.SHA256()).hex(":").upper()


class IntakeServer:
    """Runs in a background thread while the app is signed in. ``submit(details, qa, raw_text)``
    is the app's callback that stores the intake (it returns the patient id)."""

    def __init__(self, submit: Callable[[dict, list, str], str], cert_dir: Path, port: int = 8765,
                 on_event: Callable[[str], None] | None = None):
        self.submit = submit
        self.cert_dir = cert_dir
        self.port = port
        self.on_event = on_event or (lambda _m: None)
        self.pin = f"{secrets.randbelow(10**6):06d}"
        self.sessions: dict[str, float] = {}
        self.failures: dict[str, list[float]] = {}
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.fingerprint = ""
        self.urls: list[str] = []

    @property
    def running(self) -> bool:
        return self._httpd is not None

    def new_pin(self) -> str:
        self.pin = f"{secrets.randbelow(10**6):06d}"
        self.sessions.clear()
        return self.pin

    def start(self) -> None:
        if self.running:
            return
        ips = lan_addresses()
        cert, key, self.fingerprint = ensure_certificate(self.cert_dir, ips)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(str(cert), str(key))
        server = self

        class Handler(_Handler):
            srv = server

        httpd = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        httpd.daemon_threads = True
        self._httpd = httpd
        self.urls = [f"https://{ip}:{self.port}/" for ip in ips] or [f"https://localhost:{self.port}/"]
        self._thread = threading.Thread(target=httpd.serve_forever, name="lan-intake", daemon=True)
        self._thread.start()
        log.info("LAN intake server on port %s", self.port)

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        self.sessions.clear()

    # used by the handler
    def check_session(self, token: str | None) -> bool:
        exp = self.sessions.get(token or "")
        return bool(exp and exp > time.time())

    def try_pair(self, ip: str, pin: str) -> str | None:
        now = time.time()
        recent = [t for t in self.failures.get(ip, []) if now - t < 300]
        if len(recent) >= 5:
            return None
        if secrets.compare_digest(pin.strip(), self.pin):
            token = secrets.token_urlsafe(32)
            self.sessions[token] = now + SESSION_HOURS * 3600
            self.failures.pop(ip, None)
            self.on_event(f"Tablet paired ({ip})")
            return token
        self.failures[ip] = recent + [now]
        return None


class _Handler(BaseHTTPRequestHandler):
    srv: IntakeServer
    server_version = "HoRUS"
    sys_version = ""

    def log_message(self, fmt, *args):                     # no patient data in logs
        log.debug("lan %s %s", self.client_address[0], fmt % args)

    def _headers(self, code: int, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                         "connect-src 'self'; form-action 'self'; base-uri 'none'")
        self.send_header("Strict-Transport-Security", "max-age=86400")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _json(self, code: int, obj, extra=None):
        body = json.dumps(obj).encode()
        self._headers(code, "application/json", extra)
        self.wfile.write(body)

    def _token(self) -> str | None:
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "horus_session":
                return v
        return None

    def _body(self) -> dict | None:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            return None
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = TABLET_PAGE.replace("__QUESTIONS__", json.dumps(PATIENT_QUESTIONS))
            self._headers(200, "text/html; charset=utf-8")
            self.wfile.write(page.encode("utf-8"))
        elif self.path == "/api/status":
            self._json(200, {"paired": self.srv.check_session(self._token())})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.headers.get("Origin") and not self.headers["Origin"].startswith("https://"):
            return self._json(403, {"error": "bad origin"})
        data = self._body()
        if data is None:
            return self._json(400, {"error": "bad request"})
        if self.path == "/api/pair":
            token = self.srv.try_pair(self.client_address[0], str(data.get("pin", "")))
            if not token:
                return self._json(403, {"error": "That PIN is not correct, or too many attempts. Ask the reception."})
            cookie = (f"horus_session={token}; Max-Age={SESSION_HOURS * 3600}; Path=/; HttpOnly; Secure; "
                      "SameSite=Strict")
            return self._json(200, {"ok": True}, {"Set-Cookie": cookie})
        if self.path == "/api/intake":
            if not self.srv.check_session(self._token()):
                return self._json(401, {"error": "This tablet needs to be paired again."})
            details = {k: str(data.get("details", {}).get(k, ""))[:200] for k in
                       ("name", "age", "gender", "contact", "prev_id")}
            if not details["name"].strip():
                return self._json(400, {"error": "Please enter your name."})
            answers = {str(k)[:40]: str(v)[:MAX_FIELD] for k, v in (data.get("answers") or {}).items()}
            qa = [{"question": q["q"], "label": q["label"], "answer": answers.get(q["id"], "").strip()}
                  for q in PATIENT_QUESTIONS]
            try:
                pid = self.srv.submit(details, qa, compose_patient_narrative(PATIENT_QUESTIONS, answers))
            except Exception:
                log.exception("LAN intake could not be stored")
                return self._json(500, {"error": "Could not save. Please tell the reception."})
            self.srv.on_event("New intake received")
            return self._json(200, {"ok": True, "patient_id": pid})
        self._json(404, {"error": "not found"})


TABLET_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>HoRUS — patient questionnaire</title>
<style>
:root{--bg:#f6f8fb;--ink:#1e293b;--muted:#64748b;--accent:#2563eb;--border:#dbe3ee}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:18px/1.5 system-ui,sans-serif}
main{max-width:720px;margin:0 auto;padding:24px 16px}h1{font-size:26px;margin:0 0 8px}
.card{background:#fff;border:1px solid var(--border);border-radius:12px;padding:18px;margin:14px 0}
label{display:block;font-weight:600;margin:12px 0 4px}input,select,textarea{width:100%;font:inherit;padding:10px;
border:1px solid #cbd5e1;border-radius:8px;background:#fff}textarea{min-height:180px}
button{font:inherit;font-weight:600;padding:12px 22px;border-radius:9px;border:1px solid var(--accent);
background:var(--accent);color:#fff}button.sec{background:#fff;color:var(--accent)}.row{display:flex;gap:10px;
justify-content:space-between;margin-top:16px}.muted{color:var(--muted)}.q{font-size:24px;font-weight:600}
.err{color:#b91c1c;margin-top:8px}.hide{display:none}
</style></head><body><main>
<h1>Patient questionnaire</h1>
<section id="pair" class="card"><p>Please ask the reception for the 6-digit code shown on the clinic computer.</p>
<label for="pin">Code</label><input id="pin" inputmode="numeric" maxlength="6" autocomplete="off">
<div class="row"><span></span><button onclick="pair()">Continue</button></div><div id="pairerr" class="err"></div></section>
<section id="details" class="card hide"><p class="muted">Answer in English, Bengali or Hindi — whatever is easiest.</p>
<label>Your full name</label><input id="name"><label>Age</label><input id="age" inputmode="numeric">
<label>Gender</label><select id="gender"><option></option><option>Female</option><option>Male</option><option>Other</option></select>
<label>Phone</label><input id="contact" inputmode="tel"><label>Patient ID (if you have visited before)</label>
<input id="prev_id" placeholder="e.g. PT-2026-004">
<div class="row"><span></span><button onclick="begin()">Start questions</button></div><div id="deterr" class="err"></div></section>
<section id="qs" class="card hide"><p class="muted" id="prog"></p><p class="q" id="qtext"></p><p class="muted" id="qhint"></p>
<textarea id="ans"></textarea><div class="row"><button class="sec" onclick="back()">Back</button>
<button id="nextb" onclick="next()">Next</button></div><div id="qerr" class="err"></div></section>
<section id="done" class="card hide"><p class="q">Thank you. Your answers have been given to the doctor.</p>
<p id="pidmsg"></p><div class="row"><span></span><button onclick="again()">Next patient</button></div></section>
</main><script>
const Q=__QUESTIONS__;let i=0,ans={};const $=id=>document.getElementById(id);
function show(id){for(const s of ['pair','details','qs','done'])$(s).classList.toggle('hide',s!==id)}
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify(body),credentials:'same-origin'});let j={};try{j=await r.json()}catch(e){}return [r.ok,j]}
async function pair(){const [ok,j]=await post('/api/pair',{pin:$('pin').value});if(ok){show('details')}else{$('pairerr').textContent=j.error||'Error'}}
function begin(){if(!$('name').value.trim()){$('deterr').textContent='Please enter your name.';return}i=0;ans={};show('qs');render()}
function render(){const q=Q[i];$('prog').textContent='Question '+(i+1)+' of '+Q.length;$('qtext').textContent=q.q;
 $('qhint').textContent=q.hint||'';$('ans').value=ans[q.id]||'';$('nextb').textContent=i===Q.length-1?'Finish':'Next';$('ans').focus()}
function keep(){ans[Q[i].id]=$('ans').value}
function back(){keep();if(i>0){i--;render()}}
async function next(){keep();if(i<Q.length-1){i++;render();return}
 const d={};for(const k of ['name','age','gender','contact','prev_id'])d[k]=$(k).value;
 const [ok,j]=await post('/api/intake',{details:d,answers:ans});
 if(ok){$('pidmsg').textContent='Your patient ID is '+j.patient_id+'. Please keep it for your next visit.';show('done')}
 else if(j.error&&j.error.includes('paired')){show('pair');$('pairerr').textContent=j.error}
 else{$('qerr').textContent=j.error||'Could not save.'}}
function again(){for(const k of ['name','age','gender','contact','prev_id'])$(k).value='';show('details')}
fetch('/api/status',{credentials:'same-origin'}).then(r=>r.json()).then(j=>{if(j.paired)show('details')});
</script></body></html>"""
