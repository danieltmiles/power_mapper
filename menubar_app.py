"""
Power Mapper LLM Worker — macOS menu bar app.

Manages the llm.py worker process: configure credentials, start, stop.
Config is stored in ~/Library/Application Support/PowerMapper/config.json.
The RabbitMQ password is stored in the macOS Keychain via keyring.
The GGUF model is downloaded on first run if not present.
"""

import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.request

import keyring
import rumps

APP_NAME = "PowerMapper"
KEYCHAIN_SERVICE = "PowerMapper-RabbitMQ"
KEYCHAIN_ACCOUNT = "rabbitmq"

RABBITMQ_HOST = "rabbitmq.doodledome.org"
RABBITMQ_PORT = 5671

REDIS_HOST = "doodledome.org"
REDIS_PORT = 6380

MODEL_URL = "https://static.doodledome.org/Qwen3-32B-Q4_K_M.gguf"
MODEL_FILENAME = "Qwen3-32B-Q4_K_M.gguf"
MODEL_SHA256 = "511678e89daafd24763ad2eb72173f8095feca910fc1599cb361064c51e920bf"

SUPPORT_DIR = os.path.expanduser(f"~/Library/Application Support/{APP_NAME}")
CONFIG_PATH = os.path.join(SUPPORT_DIR, "config.json")
MODEL_PATH = os.path.join(SUPPORT_DIR, "models", MODEL_FILENAME)
LOG_PATH = os.path.join(SUPPORT_DIR, "llm_worker.log")
APP_LOG_PATH = os.path.join(SUPPORT_DIR, "app.log")

LOG_TAIL_LINES = 30

ICON_STOPPED = "⬛"
ICON_RUNNING = "🟩"
ICON_BUSY    = "🟡"


def get_resource_path(relative_path: str) -> str:
    """Return the absolute path to a bundled resource.

    When running as a PyInstaller bundle, files land in sys._MEIPASS.
    When running from source, they're next to this script.
    """
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path)


def ensure_support_dir():
    os.makedirs(os.path.join(SUPPORT_DIR, "models"), exist_ok=True)


def setup_logging():
    ensure_support_dir()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(APP_LOG_PATH),
            logging.StreamHandler(sys.stderr),
        ],
    )


log = logging.getLogger(__name__)


def load_config() -> dict:
    ensure_support_dir()
    if not os.path.exists(CONFIG_PATH):
        return {"username": "guest", "work_queue": "llm/generic"}
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(config: dict):
    ensure_support_dir()
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def load_password() -> str:
    return keyring.get_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT) or ""


def save_password(password: str):
    keyring.set_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT, password)


def build_worker_config(config: dict) -> dict:
    """Assemble the full config dict that llm.py expects."""
    return {
        "work_queue":    config.get("work_queue", "llm/generic"),
        "host":          RABBITMQ_HOST,
        "port":          RABBITMQ_PORT,
        "username":      config.get("username", "guest"),
        "password":      load_password(),
        "ssl_cert_file": get_resource_path("server_certificate.pem"),
        "model_path":    MODEL_PATH,
        "hf_model_name": "Qwen/Qwen3-32B",
        "redis": {
            "host":         REDIS_HOST,
            "port":         REDIS_PORT,
            "ssl":          True,
            "ssl_ca_certs": get_resource_path("tls/ca.crt"),
            "ssl_certfile": get_resource_path("tls/client.crt"),
            "ssl_keyfile":  get_resource_path("tls/client.key"),
        },
    }


def write_worker_config_file(config: dict) -> str:
    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False,
        dir=SUPPORT_DIR, prefix="worker_config_"
    )
    json.dump(config, tmp)
    tmp.close()
    return tmp.name


def tail_log(n: int) -> str:
    """Return the last n lines of the worker log, or a placeholder if empty."""
    if not os.path.exists(LOG_PATH):
        return "(no log yet)"
    with open(LOG_PATH) as f:
        lines = f.readlines()
    if not lines:
        return "(log is empty)"
    return "".join(lines[-n:])


def compute_sha256(path: str, progress_callback=None) -> str:
    """Return the hex SHA-256 digest of the file at path.

    progress_callback, if provided, is called with (bytes_done, total_bytes)
    after each chunk so the caller can update status.
    """
    digest = hashlib.sha256()
    total = os.path.getsize(path)
    done = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)  # 4 MB
            if not chunk:
                break
            digest.update(chunk)
            done += len(chunk)
            if progress_callback:
                progress_callback(done, total)
    return digest.hexdigest()


def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


class PowerMapperApp(rumps.App):
    def __init__(self):
        super().__init__(APP_NAME, title=ICON_STOPPED)
        self.worker_process = None
        self.worker_config_file = None
        self.caffeinate_process = None

        # Shared state written by background threads, read by the UI timer.
        # Never touch rumps/AppKit objects from background threads.
        self._pending_icon = None      # str or None
        self._pending_status = None    # str or None
        self._pending_notify = None    # (title, message) or None
        self._pending_configure = False
        self._pending_worker_exited = False
        self._state_lock = threading.Lock()
        self._status_text = "Stopped"
        self._status_window = None
        self._status_text_view = None

        self._status_item = rumps.MenuItem("⬛  Stopped")
        self._status_item.set_callback(None)   # display only, not clickable

        self.menu = [
            rumps.MenuItem("Start", callback=self.start_worker),
            rumps.MenuItem("Stop",  callback=self.stop_worker),
            None,
            self._status_item,
            rumps.MenuItem("Show Power Mapper...", callback=self.show_status),
            None,
            rumps.MenuItem("Configure...", callback=self.configure),
            None,
        ]
        self._update_menu_state()

        # Timer runs on the main thread; flushes any pending UI state.
        self._ui_timer = rumps.Timer(self._flush_pending_ui, 0.5)
        self._ui_timer.start()

        threading.Thread(target=self._first_launch_check, daemon=True).start()

    # ------------------------------------------------------------------
    # Thread-safe status helpers
    # ------------------------------------------------------------------

    def _request_status(self, icon: str, text: str):
        """Called from any thread. Queues a UI update for the main thread."""
        log.info("status: %s %s", icon, text)
        with self._state_lock:
            self._pending_icon = icon
            self._pending_status = text

    def _request_notify(self, title: str, message: str):
        """Called from any thread. Queues a notification for the main thread."""
        log.info("notify: %s — %s", title, message)
        with self._state_lock:
            self._pending_notify = (title, message)

    def _flush_pending_ui(self, _timer):
        """Runs on the main thread every 0.5 s. Applies any queued state."""
        with self._state_lock:
            icon = self._pending_icon
            status = self._pending_status
            notify = self._pending_notify
            do_configure = self._pending_configure
            worker_exited = self._pending_worker_exited
            self._pending_icon = None
            self._pending_status = None
            self._pending_notify = None
            self._pending_configure = False
            self._pending_worker_exited = False

        if icon is not None and status is not None:
            self.title = icon
            self._status_item.title = f"{icon}  {status}"
            self._status_text = status
        if notify is not None:
            rumps.notification(APP_NAME, notify[0], notify[1])
        if do_configure:
            self.configure(None)
        if worker_exited:
            self._update_menu_state()
        if self._status_window is not None and self._status_window.isVisible():
            self._refresh_status_window()

    # ------------------------------------------------------------------
    # First-launch / model download
    # ------------------------------------------------------------------

    def _first_launch_check(self):
        log.info("first launch check")
        if not load_password():
            with self._state_lock:
                self._pending_configure = True

        if not os.path.exists(MODEL_PATH):
            self._download_model()
        else:
            self._verify_model(delete_on_mismatch=True)

    def _download_model(self):
        log.info("starting model download to %s", MODEL_PATH)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        tmp_path = MODEL_PATH + ".part"
        retry_delay = 5  # seconds between retries

        for attempt in range(1, 999):
            # Re-read partial size on every attempt so we resume correctly.
            downloaded = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
            headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
            log.info("download attempt %d, resuming from %d bytes", attempt, downloaded)
            self._request_status(ICON_BUSY, "Starting download…" if attempt == 1 else f"Retrying download (attempt {attempt})…")

            try:
                request = urllib.request.Request(MODEL_URL, headers=headers)
                with urllib.request.urlopen(request) as response:
                    total_remaining = int(response.headers.get("Content-Length", 0))
                    total = downloaded + total_remaining
                    speed_start_bytes = downloaded
                    speed_start_time = time.monotonic()
                    last_log_time = time.monotonic()

                    with open(tmp_path, "ab") as f:
                        while True:
                            chunk = response.read(1024 * 1024)  # 1 MB
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)

                            if total:
                                elapsed = time.monotonic() - speed_start_time or 0.001
                                speed = (downloaded - speed_start_bytes) / elapsed  # bytes/s
                                remaining_bytes = total - downloaded
                                eta = format_eta(remaining_bytes / speed) if speed > 0 else "?"
                                pct = downloaded * 100 // total
                                gb_done = downloaded / 1e9
                                gb_total = total / 1e9
                                mb_s = speed / 1e6

                                status = (
                                    f"Downloading model: {pct}%  "
                                    f"({gb_done:.1f}/{gb_total:.1f} GB  "
                                    f"{mb_s:.1f} MB/s  ETA {eta})"
                                )
                                icon = f"{pct}%"
                                with self._state_lock:
                                    self._pending_icon = icon
                                    self._pending_status = status

                                now = time.monotonic()
                                if now - last_log_time >= 30:
                                    log.info("download progress: %s", status)
                                    last_log_time = now

                # Confirm we received every byte before trusting the file.
                if downloaded != total:
                    raise IOError(
                        f"Download ended early: got {downloaded} of {total} bytes"
                    )

                # Completed without error — verify then rename.
                log.info("download complete, verifying checksum")
                if self._verify_model(path=tmp_path, delete_on_mismatch=False):
                    os.rename(tmp_path, MODEL_PATH)
                    self._request_status(ICON_STOPPED, "Stopped (model ready)")
                    self._request_notify("Download complete", "Model is ready. Click Start to begin.")
                else:
                    # Checksum failed — delete the corrupt .part and retry.
                    log.warning("checksum mismatch on completed download, deleting and retrying")
                    os.remove(tmp_path)
                    # Fall through to the next loop iteration.
                    continue
                return

            except Exception as exc:
                log.warning("download attempt %d failed: %s — retrying in %ds", attempt, exc, retry_delay)
                self._request_status(ICON_BUSY, f"Download interrupted, retrying in {retry_delay}s… ({exc})")
                time.sleep(retry_delay)

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    @rumps.clicked("Start")
    def start_worker(self, _):
        log.info("start_worker clicked")
        if self.worker_process and self.worker_process.poll() is None:
            return

        if not load_password():
            self._notify("Not configured", "Open Configure… to set your credentials.")
            return
        if not os.path.exists(MODEL_PATH):
            self._notify("Model not ready", "Model is still downloading — check Show Power Mapper.")
            return

        config = load_config()
        worker_config = build_worker_config(config)
        self.worker_config_file = write_worker_config_file(worker_config)

        # When bundled, llm_worker is a standalone executable in Contents/MacOS/
        # (next to sys.executable). When running from source, sys.executable is
        # a real Python interpreter and llm.py is a sibling script.
        if hasattr(sys, "_MEIPASS"):
            worker_exe = os.path.join(os.path.dirname(sys.executable), "llm_worker")
            cmd = [worker_exe, self.worker_config_file]
        else:
            cmd = [sys.executable, get_resource_path("llm.py"), self.worker_config_file]
        log.info(f"{cmd=}")

        log.info("launching worker: %s", " ".join(cmd))
        log_file = open(LOG_PATH, "a")
        self.worker_process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
        )

        self.caffeinate_process = subprocess.Popen(
            ["caffeinate", "-i", "-s", "-w", str(self.worker_process.pid)]
        )
        log.info("caffeinate started (pid %d)", self.caffeinate_process.pid)

        self.title = ICON_RUNNING
        self._status_item.title = f"{ICON_RUNNING}  Worker running"
        self._status_text = "Worker running"
        self._update_menu_state()
        threading.Thread(target=self._watch_worker, daemon=True).start()

    @rumps.clicked("Stop")
    def stop_worker(self, _):
        if self.worker_process:
            self.worker_process.terminate()
            self.worker_process = None
        if self.caffeinate_process:
            self.caffeinate_process.terminate()
            self.caffeinate_process = None
        self._cleanup_config_file()
        self.title = ICON_STOPPED
        self._status_item.title = f"{ICON_STOPPED}  Stopped"
        self._status_text = "Stopped"
        self._update_menu_state()

    def _watch_worker(self):
        if self.worker_process:
            self.worker_process.wait()
        log.info("worker process exited")
        # caffeinate exits on its own via -w, but clear the reference.
        self.caffeinate_process = None
        self._cleanup_config_file()
        self._request_status(ICON_STOPPED, "Worker exited")
        # _update_menu_state touches AppKit objects; schedule it on main thread.
        with self._state_lock:
            self._pending_worker_exited = True

    def _cleanup_config_file(self):
        if self.worker_config_file and os.path.exists(self.worker_config_file):
            os.remove(self.worker_config_file)
            self.worker_config_file = None

    # ------------------------------------------------------------------
    # Status window
    # ------------------------------------------------------------------

    def _refresh_status_window(self):
        content = f"Status: {self._status_text}\n\nRecent log:\n\n{tail_log(LOG_TAIL_LINES)}"
        self._status_text_view.setString_(content)

    @rumps.clicked("Show Power Mapper...")
    def show_status(self, _):
        # If the window is already open, just bring it forward.
        if self._status_window is not None and self._status_window.isVisible():
            self._status_window.makeKeyAndOrderFront_(None)
            return

        from AppKit import (
            NSBackingStoreBuffered, NSMakeRect, NSScrollView, NSTextView, NSWindow,
            NSWindowStyleMaskClosable, NSWindowStyleMaskMiniaturizable,
            NSWindowStyleMaskResizable, NSWindowStyleMaskTitled,
        )

        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable
            | NSWindowStyleMaskResizable
        )
        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 620, 400), style, NSBackingStoreBuffered, False
        )
        window.setTitle_("Power Mapper — Status")
        window.center()

        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, 620, 400))
        scroll.setHasVerticalScroller_(True)
        scroll.setAutoresizingMask_(2 | 16)  # width + height sizable

        text_view = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, 620, 400))
        text_view.setEditable_(False)
        text_view.setAutoresizingMask_(2 | 16)

        scroll.setDocumentView_(text_view)
        window.setContentView_(scroll)
        window.makeKeyAndOrderFront_(None)
        # Note: do NOT call NSApp.activateIgnoringOtherApps_ here — it
        # interferes with the main run loop and pauses the UI timer.

        # Hold references so neither the window nor the text view gets GC'd.
        self._status_window = window
        self._status_text_view = text_view

        # Populate immediately.
        self._refresh_status_window()

    # ------------------------------------------------------------------
    # Configuration dialog
    # ------------------------------------------------------------------

    @rumps.clicked("Configure...")
    def configure(self, _):
        config = load_config()

        fields = [
            ("RabbitMQ username", config.get("username", "guest")),
            ("RabbitMQ password", load_password()),
            ("Work queue",        config.get("work_queue", "llm/generic")),
        ]

        values = {}
        for label, default in fields:
            response = rumps.Window(
                message=label,
                title="Power Mapper — Configure",
                default_text=default,
                ok="Next",
                cancel="Cancel",
                dimensions=(320, 24),
            ).run()
            if response.clicked == 0:
                return
            values[label] = response.text.strip()

        config["username"]   = values["RabbitMQ username"]
        config["work_queue"] = values["Work queue"]
        save_config(config)
        save_password(values["RabbitMQ password"])
        self._notify("Saved", "Configuration updated.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _verify_model(self, path: str = MODEL_PATH, delete_on_mismatch: bool = False) -> bool:
        """Hash the model file and compare against MODEL_SHA256.

        Returns True on match, False on mismatch.
        If delete_on_mismatch is True, removes the file and triggers a fresh
        download when a mismatch is found.
        """
        log.info("verifying checksum of %s", path)
        self._request_status(ICON_BUSY, "Verifying model checksum…")

        def on_progress(done, total):
            pct = done * 100 // total
            with self._state_lock:
                self._pending_icon = ICON_BUSY
                self._pending_status = f"Verifying model: {pct}%"

        actual = compute_sha256(path, progress_callback=on_progress)
        if actual == MODEL_SHA256:
            log.info("checksum OK: %s", actual)
            self._request_status(ICON_STOPPED, "Stopped (model verified)")
            return True

        log.error("checksum MISMATCH: expected %s, got %s", MODEL_SHA256, actual)
        if delete_on_mismatch:
            log.warning("deleting corrupt model file and re-downloading")
            os.remove(path)
            self._request_notify("Model corrupt", "Checksum mismatch — re-downloading.")
            self._download_model()
        else:
            self._request_status(ICON_BUSY, "Checksum mismatch — will retry download")
        return False

    def _update_menu_state(self):
        running = bool(self.worker_process and self.worker_process.poll() is None)
        self.menu["Start"].set_callback(None if running else self.start_worker)
        self.menu["Stop"].set_callback(self.stop_worker if running else None)

    def _notify(self, title: str, message: str):
        rumps.notification(APP_NAME, title, message)


if __name__ == "__main__":
    ensure_support_dir()
    setup_logging()
    log.info("PowerMapper starting")
    PowerMapperApp().run()
