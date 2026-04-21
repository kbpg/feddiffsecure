from __future__ import annotations

import os
import socket
import sys
from contextlib import closing

from demo_portal.app import create_app


HOST = os.environ.get("PAPER_DEMO_HOST", "127.0.0.1")
PORT = int(os.environ.get("PAPER_DEMO_PORT", "5055"))
app = create_app()


def _port_is_occupied(host: str, port: int) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


if __name__ == "__main__":
    if _port_is_occupied(HOST, PORT):
        print(
            f"Demo portal did not start because {HOST}:{PORT} is already in use.\n"
            "Stop the existing portal process first, or launch with a different PAPER_DEMO_PORT.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    app.run(host=HOST, port=PORT, debug=False)
