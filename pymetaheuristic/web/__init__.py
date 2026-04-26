"""
pymetaheuristic.web
===================
Provides the ``launch()`` entry point for the browser-based UI.

Install extras:
    pip install pymetaheuristic[web]

Usage:
    import pymetaheuristic
    pymetaheuristic.web_app()          # opens http://127.0.0.1:8765
    pymetaheuristic.web_app(port=9090) # custom port

Compatible with plain Python scripts, Spyder IDE, Jupyter notebooks,
and JupyterLab — all of which may already own an asyncio event loop.
"""
from __future__ import annotations

__all__ = ["launch"]

import asyncio
import threading
import time
import webbrowser


def launch(
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    log_level: str = "warning",
) -> None:
    """Start the pyMetaheuristic web UI.

    Automatically detects whether an asyncio event loop is already running
    (Spyder, Jupyter, JupyterLab) and starts Uvicorn in a background thread
    with its own isolated event loop so both can coexist.

    Parameters
    ----------
    host : str
        Bind address. Use ``"0.0.0.0"`` to expose on the network.
    port : int
        TCP port.
    open_browser : bool
        Open the default system browser automatically after startup.
    log_level : str
        Uvicorn log level (``"info"``, ``"warning"``, ``"error"`` …).
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "\n\nThe pyMetaheuristic web UI requires extra dependencies.\n"
            "Install them with:\n\n"
            "    pip install pymetaheuristic[web]\n\n"
            "(needs: fastapi, uvicorn[standard])"
        ) from exc

    from .server import app  # relative import — works inside the package

    url = f"http://{host}:{port}"

    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"      pyMetaheuristic Optimization Lab       ")
    print(f"      {url:<40}")
    print(f"  ╚══════════════════════════════════════════╝\n")

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=False,
    )
    server = uvicorn.Server(config)

    # ------------------------------------------------------------------ #
    # Detect whether we are inside a running event loop.                  #
    # This is always True in Spyder, Jupyter, and JupyterLab.             #
    # ------------------------------------------------------------------ #
    _running_loop: bool = False
    try:
        loop = asyncio.get_running_loop()
        if loop is not None and loop.is_running():
            _running_loop = True
    except RuntimeError:
        _running_loop = False

    if _running_loop:
        # ── Spyder / Jupyter path ──────────────────────────────────────
        # Run Uvicorn in a daemon thread that owns its own event loop.
        # The call returns immediately so the IDE stays responsive.
        _ready = threading.Event()

        def _thread_target() -> None:
            # Each thread needs its own event loop on Windows + Python 3.10+
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_serve_and_signal(server, _ready))
            finally:
                loop.close()

        t = threading.Thread(target=_thread_target, daemon=True)
        t.start()

        # Wait up to 5 s for the server to actually bind before opening browser
        if _ready.wait(timeout=5.0) and open_browser:
            webbrowser.open(url)
        elif open_browser:
            # Server took too long — open anyway
            webbrowser.open(url)

        print(f"Server running at {url}")
        print("Terminate the web service using: pymetaheuristic.web.web_stop() \n")

        # Store reference so the user can stop it
        import pymetaheuristic.web as _web_module
        _web_module._server_instance = server

    else:
        # ── Plain Python script path ───────────────────────────────────
        # Block the main thread (Ctrl+C stops everything cleanly).
        print("Press Ctrl+C to stop.\n")

        if open_browser:
            def _delayed_open() -> None:
                time.sleep(0.9)
                webbrowser.open(url)
            threading.Thread(target=_delayed_open, daemon=True).start()

        server.run()


async def _serve_and_signal(server, ready_event: threading.Event) -> None:
    """Coroutine that starts Uvicorn and signals when the server is ready."""
    import asyncio

    # Uvicorn's serve() sets server.started when it has bound the port.
    # We poll for it so we can fire the ready event at the right moment.
    serve_task = asyncio.create_task(server.serve())

    for _ in range(50):          # wait up to 5 s (50 × 100 ms)
        if server.started:
            ready_event.set()
            break
        await asyncio.sleep(0.1)
    else:
        ready_event.set()        # signal anyway so browser open doesn't hang

    await serve_task             # keep running until shutdown


def web_stop() -> None:
    """Gracefully stop the background web server (Spyder / Jupyter only).

    Usage::

        import pymetaheuristic.web as w
        w.web_stop()
    """
    import pymetaheuristic.web as _web_module
    srv = getattr(_web_module, "_server_instance", None)
    if srv is None:
        print("No running web server found.")
        return
    srv.should_exit = True
    print("Web server stop signal sent.")


_server_instance = None   # populated by launch() in Spyder/Jupyter mode