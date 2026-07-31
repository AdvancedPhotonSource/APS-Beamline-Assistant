#!/usr/bin/env python3
"""
APEXA Desktop UI — the web UI in a native window.

Wraps the *exact same* React frontend + FastAPI backend used by the web server
(``web_server.py``) in a lightweight, OS-native webview window — no bundled
Chromium, so it runs comfortably on modest beamline workstations. The FastAPI
app is served on a loopback port in a background thread and a pywebview window
points at it.

Because the React frontend talks to the backend over *relative* URLs
(``ws://<host>/ws`` and ``/api/*``), the desktop app reuses the web UI verbatim:
there is no second frontend to build or keep in sync. ``web_server.py`` is not
modified — this is a pure additive launcher.

Run::

    uv run python apexa_desktop.py      # or ./start_desktop_ui.sh

Environment::

    APEXA_DESKTOP_PORT     fixed loopback port (default: an auto-picked free one)
    APEXA_DESKTOP_WIDTH    initial window width  (default 1440)
    APEXA_DESKTOP_HEIGHT   initial window height (default 900)
"""
import os
import socket
import sys
import threading
import time
from pathlib import Path

# web_server resolves frontend/dist, servers.config and .env relative to the CWD,
# so anchor to the repo root before importing it (lets the app be launched from
# anywhere, e.g. a double-clicked shortcut).
os.chdir(Path(__file__).resolve().parent)

import uvicorn  # noqa: E402  (after chdir)
import webview  # noqa: E402

HOST = "127.0.0.1"  # desktop app binds loopback only — never 0.0.0.0


def _pick_port() -> int:
    """Honour APEXA_DESKTOP_PORT, else grab a free ephemeral loopback port."""
    env = os.environ.get("APEXA_DESKTOP_PORT", "").strip()
    if env:
        return int(env)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _wait_until_up(host: str, port: int, timeout: float = 40.0) -> bool:
    """Block until the server accepts a TCP connection (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.25)
    return False


def main() -> int:
    if not Path("frontend/dist/index.html").exists():
        print("⚠  frontend/dist not found — build the React UI first:",
              file=sys.stderr)
        print("     (cd frontend && npm install && npm run build)", file=sys.stderr)
        print("   or run ./start_desktop_ui.sh, which builds it for you.",
              file=sys.stderr)
        return 1

    port = _pick_port()

    # Serve the same FastAPI `app` the web server runs, in a daemon thread.
    # uvicorn.Server installs signal handlers only on the main thread (it detects
    # and skips this off-thread), which leaves the main thread free for pywebview
    # — required for its native GUI loop on macOS.
    from web_server import app  # imported after chdir so relative paths resolve
    # Quiet by default; set APEXA_DESKTOP_LOG=info to see every request the
    # webview makes (assets + /ws upgrade) when troubleshooting.
    log_level = os.environ.get("APEXA_DESKTOP_LOG", "warning").strip()
    config = uvicorn.Config(app, host=HOST, port=port, log_level=log_level,
                            access_log=log_level in ("debug", "info", "trace"))
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="apexa-uvicorn", daemon=True)
    thread.start()

    if not _wait_until_up(HOST, port):
        print("❌ backend did not come up in time", file=sys.stderr)
        server.should_exit = True
        return 1

    # Print the URL so it can be opened in a real browser too (useful for telling
    # a WebKit-specific problem apart from an app/build problem).
    print(f"🖥  APEXA desktop backend on http://{HOST}:{port}  "
          f"(open in a browser to compare)", flush=True)

    width = int(os.environ.get("APEXA_DESKTOP_WIDTH", "1440"))
    height = int(os.environ.get("APEXA_DESKTOP_HEIGHT", "900"))
    webview.create_window(
        "APEXA — Advanced Photon Experiment Assistant",
        f"http://{HOST}:{port}",
        width=width,
        height=height,
        min_size=(1024, 700),
    )

    # Set APEXA_DESKTOP_DEBUG=1 to enable the WebKit inspector (right-click →
    # Inspect Element) and print a one-line DOM/connection diagnosis to the
    # terminal — useful if the window ever comes up blank or unresponsive.
    debug = os.environ.get("APEXA_DESKTOP_DEBUG", "0").strip() not in ("0", "false", "no")
    try:
        webview.start(_diagnose if debug else None, debug=debug)
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    return 0


def _diagnose() -> None:
    """After the page loads, report DOM mount + resource/WS state to the terminal."""
    try:
        win = webview.windows[0]
        time.sleep(3.0)  # let the SPA boot and attempt its WebSocket
        # Install a forward error hook, then wait to catch async failures too.
        win.evaluate_js(
            "window.__ax=window.__ax||[];"
            "if(!window.__axHook){window.__axHook=1;"
            "addEventListener('error',e=>window.__ax.push(String(e.message||e)));"
            "addEventListener('unhandledrejection',e=>window.__ax.push('promise:'+String(e.reason)));}"
        )
        time.sleep(2.0)
        report = win.evaluate_js(
            "JSON.stringify({"
            "rootKids:(document.getElementById('root')||{}).childElementCount??-1,"
            "title:document.title,"
            "res:performance.getEntriesByType('resource').map(r=>r.name.split('/').pop()+':'+(r.responseStatus??'?')),"
            "errs:(window.__ax||[]).slice(0,8)})"
        )
        print("\n[apexa-desktop DIAG] " + str(report) + "\n", flush=True)
        print("[apexa-desktop] rootKids=0 → React did not mount (JS error above); "
              ">0 → UI mounted. Right-click the window → Inspect Element for details.",
              flush=True)
    except Exception as exc:  # diagnostics must never crash the app
        print(f"[apexa-desktop DIAG] introspection failed: {exc!r}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
