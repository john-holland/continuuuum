#!/usr/bin/env python3
"""
Entropythief petal compute node. Probes target, registers with continuum,
submits guesses, participates in ring pass-along.
Run: python -m entropy.entropythief_node --continuum-url http://localhost:5050 --probe-target 8.8.8.8:53
"""
from __future__ import annotations

import argparse
import json
import signal
import socket
import statistics
import sys
import threading
import time
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Ensure continuum repo root on path (parent of entropy/)
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _tcp_rtt_ms(host: str, port: int, timeout: float) -> float:
    start = time.perf_counter()
    with socket.create_connection((host, port), timeout=timeout):
        pass
    return (time.perf_counter() - start) * 1000.0


def _loopback_rtt_ms(timeout: float) -> float:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(timeout)
    host, port = listener.getsockname()
    start = time.perf_counter()
    client = socket.create_connection((host, port), timeout=timeout)
    conn, _ = listener.accept()
    client.close()
    conn.close()
    listener.close()
    return (time.perf_counter() - start) * 1000.0


def _probe_once(probe_target: str, probe_class: str, timeout: float) -> float:
    if probe_class == "localhost_loopback":
        return _loopback_rtt_ms(timeout)
    host, port_s = probe_target.rsplit(":", 1)
    return _tcp_rtt_ms(host, int(port_s), timeout)


def _predict_fallback(series: list[float]) -> float:
    """Fallback predictor: exponential moving average."""
    if not series:
        return 0.0
    alpha = 0.3
    pred = series[0]
    for x in series[1:]:
        pred = alpha * x + (1 - alpha) * pred
    return pred


def _http_json(url: str, method: str = "GET", data: dict | None = None) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8") if data else None,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Entropythief petal node")
    parser.add_argument("--continuum-url", default="http://localhost:5050", help="Continuum server URL")
    parser.add_argument("--probe-target", default="8.8.8.8:53", help="Probe target host:port")
    parser.add_argument("--probe-class", default="google_dns", help="Probe class (google_dns, akamai_edge, localhost_loopback)")
    parser.add_argument("--tenant-id", default="default", help="Tenant ID")
    parser.add_argument("--port", type=int, default=0, help="Port for health server (0=auto)")
    parser.add_argument("--base-url", default="", help="This node's base URL for health checks (default: http://127.0.0.1:<port>)")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between rounds")
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), _HealthHandler)
    port = server.server_address[1]
    base_url = args.base_url or f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    continuum_url = args.continuum_url.rstrip("/")
    probe_target = args.probe_target
    probe_class = args.probe_class
    tenant_id = args.tenant_id
    interval = args.interval

    node_id: str | None = None
    rtt_history: list[float] = []
    running = True

    def _sig(*_args: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    while running:
        try:
            if node_id is None:
                body = {
                    "probe_target": probe_target,
                    "tenant_id": tenant_id,
                    "base_url": base_url or None,
                }
                resp = _http_json(f"{continuum_url}/api/entropy/nodes/register", "POST", body)
                node_id = resp.get("node_id")
                if not node_id:
                    print("Failed to register", resp, file=sys.stderr)
                    time.sleep(interval)
                    continue

            rtt = _probe_once(probe_target, probe_class, 2.0)
            rtt_history.append(rtt)
            if len(rtt_history) > 100:
                rtt_history = rtt_history[-50:]
            pred = _predict_fallback(rtt_history)
            residual = rtt - pred

            center_resp = _http_json(f"{continuum_url}/api/entropy/center")
            center_hex = center_resp.get("center_hex", "")
            center_bytes = bytes.fromhex(center_hex) if center_hex else b""

            guess_bytes = (int(residual * 1000) % 256).to_bytes(1, "big") + center_bytes[:31]
            guess_payload = {
                "guess_bytes": guess_bytes.hex(),
                "rtt_mean_ms": statistics.fmean(rtt_history[-10:]) if len(rtt_history) >= 2 else rtt,
                "residual_ms": residual,
            }
            _http_json(f"{continuum_url}/api/entropy/guess", "POST", {
                "node_id": node_id,
                "guess_payload": guess_payload,
            })

        except Exception as e:
            print(f"Round error: {e}", file=sys.stderr)

        time.sleep(interval)

    return 0


if __name__ == "__main__":
    sys.exit(main())
