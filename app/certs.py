"""Self-signed certificate for the LAN address.

Why: browsers only hand out the camera on a "secure context" - https:// or localhost.  http://192.168.x.x:8000 on a
phone gets `navigator.mediaDevices === undefined`, i.e. the page would say "This browser does not support camera access".
So the bridge serves https.  The phone shows a one-time "not private" warning for a self-signed cert: accept it once.
"""
from __future__ import annotations

import datetime as dt
import ipaddress
import socket
from pathlib import Path
from typing import List, Tuple

from app.settings import CERT_DIR


def lan_ips() -> List[str]:
    """This machine's LAN addresses, the one the OS would route to the internet / wifi first."""
    ips: List[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))             # UDP "connect" sends nothing, it only asks the OS which interface it would use
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.append(info[4][0])
    except OSError:
        pass
    seen, out = set(), []
    for ip in ips:
        if ip not in seen and not ip.startswith(("127.", "169.254.", "0.")):
            seen.add(ip)
            out.append(ip)
    return out


def ensure_cert(ips: List[str], cert_dir: Path = CERT_DIR) -> Tuple[Path, Path]:
    """-> (cert.pem, key.pem).  Re-created when the machine's addresses changed (a cert only matches the names it lists)."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    except ImportError:
        raise SystemExit("https needs the 'cryptography' package:  pip install cryptography\n"
                         "  (or run with --no-https, which only works for localhost on this PC, not for a phone)")

    cert_dir.mkdir(parents=True, exist_ok=True)
    cert, key, stamp = cert_dir / "cert.pem", cert_dir / "key.pem", cert_dir / "names.txt"
    host = socket.gethostname()
    names = sorted({"localhost", host, "127.0.0.1", *ips})
    if cert.is_file() and key.is_file() and stamp.is_file() and stamp.read_text().split() == names:
        return cert, key

    san = [x509.DNSName("localhost"), x509.DNSName(host)]
    for n in names:
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(n)))
        except ValueError:
            pass
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "MaToS local bridge")])
    now = dt.datetime.now(dt.timezone.utc)
    c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
         .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(days=1))
         .not_valid_after(now + dt.timedelta(days=365))
         .add_extension(x509.SubjectAlternativeName(san), critical=False)
         .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
         .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
         .sign(k, hashes.SHA256()))
    key.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                    serialization.NoEncryption()))
    cert.write_bytes(c.public_bytes(serialization.Encoding.PEM))
    stamp.write_text("\n".join(names))
    return cert, key
