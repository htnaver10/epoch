"""Local launcher: one server, no training environment or GPU required."""
import argparse
import json
from pathlib import Path
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

import uvicorn


def health(base):
    try:
        with urllib.request.urlopen(base + "/viewer-health", timeout=1) as response:
            result = json.load(response)
        return result if result.get("service") == "epoch-local-viewer" else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    root = Path(__file__).resolve().parents[1]
    if not (root / "ready-to-run/index.html").is_file():
        sys.exit("Compiled frontend missing. Run npm.cmd run build in frontend and copy dist to ready-to-run.")
    base = f"http://127.0.0.1:{args.port}"
    url = base + "/?mode=ml"
    existing = health(base)
    if existing:
        print("EPOCH is already running: " + url)
        if not args.no_browser:
            webbrowser.open(url)
        return
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError:
            sys.exit(f"Port {args.port} is occupied by another service. Use --port 8001 or stop that service yourself.")

    if not args.no_browser:
        def open_when_ready():
            for _ in range(40):
                if health(base):
                    webbrowser.open(url)
                    return
                time.sleep(0.25)
        threading.Thread(target=open_when_ready, daemon=True).start()
    print(f"Open {url}\nKeep this window open. Press Ctrl+C to stop.", flush=True)
    uvicorn.run("viewer:app", host="127.0.0.1", port=args.port, app_dir=str(root / "backend"))


if __name__ == "__main__":
    main()
