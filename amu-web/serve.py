#!/usr/bin/env python3
"""Static server for amu-web, so the page can be opened from a phone.

    python serve.py                # 0.0.0.0:8080
    python serve.py --port 9000
    python serve.py --host 127.0.0.1

Standard library only: no install step, works the same on desktop and Termux.
The page needs no special headers, so a plain file:// open mostly works too,
except that browsers block fetch() of the .onnx file from file:// - use this
server instead.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socket
import socketserver
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXTRA_TYPES = {
    ".onnx": "application/octet-stream",
    ".mjs": "text/javascript",
    ".js": "text/javascript",
    ".json": "application/json",
    ".wasm": "application/wasm",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # No cross-origin isolation on purpose: numThreads = 1 in app.js needs
        # neither COOP nor COEP, so this stays usable from any static host.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):  # quieter console
        print("  %s" % (fmt % args))


def lan_address() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            return probe.getsockname()[0]
    except OSError:
        return "your-lan-ip"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    handler = functools.partial(Handler, directory=str(ROOT))
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.ThreadingTCPServer((args.host, args.port), handler) as httpd:
        print("amu web serving %s" % ROOT)
        print("  desktop : http://127.0.0.1:%d/" % args.port)
        if args.host == "0.0.0.0":
            print("  phone   : http://%s:%d/   (same Wi-Fi)" % (lan_address(), args.port))
        print("Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
