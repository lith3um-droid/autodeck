"""The standalone autodeck.app: first-launch setup and the runtime environment.

The app bundle is small (~40 MB DMG; a ~60 MB zip on Windows): a Python runtime, uv, the
rubberband CLI and autodeck's own source + uv.lock. Everything heavy is installed on first launch
into config.app_home() (~/Library/Application Support/autodeck/ on macOS, %LOCALAPPDATA%\autodeck
on Windows):

    runtime/python/   copy of the bundled Python (the venv must not point into the .app,
                      which can be moved or replaced by an update)
    runtime/venv/     `uv sync --frozen --no-dev` against the bundled uv.lock: exact pinned
                      versions, hashes verified by uv (venv/bin; venv/Scripts on Windows)
    runtime/bin/      ffmpeg -> the imageio-ffmpeg wheel's binary (a copy on Windows; deno is in the venv)
    runtime/setup.json  marker: app version, lock hash, Python version; written last
    models/           beat_this, Discogs-EffNet, Whisper tiny, htdemucs (pinned URLs + sha256)

Launch sequence (Contents/MacOS/autodeck is a shell script; on Windows autodeck.cmd next to
Resources/ does the same with pythonw.exe, see scripts/build_standalone_windows.py):
  1. bootstrap(), on the bundled Python, stdlib only: copy Python, then `uv sync --only-group ui`
     (pywebview + pyobjc, ~15 MB) so a window can be shown, then exec the venv's Python.
  2. desktop.main() on the venv's Python: environment(), then the "Setting up autodeck" window
     (web/setup.html) when setup_reasons() says so, then the normal UI.
Every step is idempotent (uv sync converges, downloads resume from .part files), and the marker
is written only after all of them succeeded, so an interrupted setup simply continues next time.
This module must import nothing outside the stdlib: step 1 runs before any package exists.
"""
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import urllib.request
from pathlib import Path

from . import config

# Set by the app's launcher script; absent in dev (`uv run`), where none of this runs.
RESOURCES = os.environ.get("AUTODECK_RESOURCES")
ACTIVE = bool(RESOURCES)
PY = f"python{sys.version_info.major}.{sys.version_info.minor}"
WIN = sys.platform == "win32"
EXE = ".exe" if WIN else ""
# Windows: the app runs under pythonw (no console), so a console child (uv here; rubberband and
# ffmpeg later, see desktop.py) would pop up a black window each time. Harmless from a terminal.
QUIET = {"creationflags": subprocess.CREATE_NO_WINDOW} if WIN else {}

# (path under models/, URL, sha256, bytes). Hashes pinned from the files themselves (the
# Whisper one is also Hugging Face's LFS id); a mismatch is treated as a failed download.
WHISPER = "https://huggingface.co/Systran/faster-whisper-tiny/resolve/d90ca5fe260221311c53c58e660288d3deb8d356/"
DISCOGS = "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/"
# Vocal/instrumental stems (stems.py), which also reads this entry for its dev download + check.
HTDEMUCS = ("htdemucs-955717e8-8726e21a.th", "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
            "8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4", 84141911)
MODELS = [
    ("beat_this-final0.ckpt", "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt",
     "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331", 81058141),
    ("discogs-effnet-bsdynamic-1.onnx", DISCOGS + "discogs-effnet-bsdynamic-1.onnx",
     "a280825b334797cf677939db8cd5762c0392aedd0ca6415dbc1cd083f045e43c", 18027718),
    ("discogs-effnet-bsdynamic-1.json", DISCOGS + "discogs-effnet-bsdynamic-1.json",
     "a2e85b2e7372d5f8e0f35bdd6aeae1139f101087d183d0b2fb60b0ea0f01a0ff", 14986),
    ("faster-whisper-tiny/config.json", WHISPER + "config.json",
     "a73a28cdfe1c43ccc7202fa333d1f89c202477271407ae9a7f19afa52039cac8", 2249),
    ("faster-whisper-tiny/model.bin", WHISPER + "model.bin",
     "dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1", 75538270),
    ("faster-whisper-tiny/tokenizer.json", WHISPER + "tokenizer.json",
     "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab", 2203239),
    ("faster-whisper-tiny/vocabulary.txt", WHISPER + "vocabulary.txt",
     "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913", 459861),
    HTDEMUCS,
]


# --- paths ---------------------------------------------------------------------------------
def resources() -> Path:
    return Path(RESOURCES or ".")


def app_dir() -> Path:
    return resources() / "app"  # autodeck/, web/, design/, pyproject.toml, uv.lock


def runtime() -> Path:
    return config.app_home() / "runtime"


def venv_python() -> Path:
    return runtime() / "venv" / ("Scripts/python.exe" if WIN else f"bin/{PY}")


def venv_pythonw() -> Path:
    """The interpreter the app itself runs on: pythonw on Windows (no console window)."""
    return runtime() / "venv" / ("Scripts/pythonw.exe" if WIN else f"bin/{PY}")


def venv_bin() -> Path:
    """The venv's console scripts and tool binaries (deno)."""
    return runtime() / "venv" / ("Scripts" if WIN else "bin")


def site_packages() -> Path:
    return runtime() / "venv" / ("Lib/site-packages" if WIN else f"lib/{PY}/site-packages")


def runtime_python() -> Path:
    """The copy of the bundled interpreter the venv is built on."""
    return runtime() / "python" / ("python.exe" if WIN else f"bin/{PY}")


def marker_file() -> Path:
    return runtime() / "setup.json"


def models_dir() -> Path:
    return config.app_home() / "models"


# --- pure logic (unit-tested) ------------------------------------------------------------
def expected(app_version: str, lock_hash: str, python_version: str) -> dict:
    return {"app_version": app_version, "lock": lock_hash, "python": python_version}


def setup_reasons(marker: dict | None, want: dict, venv_ok: bool, models_ok: bool) -> list[str]:
    """Why setup has to run; [] = ready to start. A new app version with the same uv.lock
    needs nothing (just a marker refresh); a different lock re-syncs (uv only changes the diff)."""
    if not marker:
        return ["first run"]
    why = []
    if marker.get("repair"):
        why.append("repair requested")
    if marker.get("lock") != want["lock"]:
        why.append("packages changed")
    if marker.get("python") != want["python"]:
        why.append("Python changed")
    if not venv_ok:
        why.append("packages missing")
    if not models_ok:
        why.append("models missing")
    return why


_DOWNLOADING = re.compile(r"^\s*Downloading (\S+) \(([\d.]+)\s*([KMG]i?B)\)")
_DONE = re.compile(r"^\s*(?:Downloaded|Built) (\S+?)(?:==\S+)?\s*$")
_UNIT = {"KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30, "KB": 1e3, "MB": 1e6, "GB": 1e9}


def parse_uv_line(line: str) -> tuple[str, str, float] | None:
    """One line of `uv sync` stderr -> ("start", pkg, bytes) / ("done", pkg, 0) / ("prepared",
    "", 0) / ("installed", "", 0), or None. uv only reports big downloads by name, which is
    where nearly all of the bytes are (torch alone is a third)."""
    m = _DOWNLOADING.match(line)
    if m:
        return "start", m.group(1), float(m.group(2)) * _UNIT[m.group(3)]
    m = _DONE.match(line)
    if m:
        return "done", m.group(1), 0.0
    if line.startswith("Prepared "):
        return "prepared", "", 0.0
    if line.startswith(("Installed ", "Uninstalled ", "Audited ")):
        return "installed", "", 0.0
    return None


def would_install(dry_run_output: str) -> set[str]:
    """Package names from `uv sync --dry-run` (" + name==version" lines)."""
    return {m.group(1) for m in re.finditer(r"^ \+ ([A-Za-z0-9._-]+)==", dry_run_output, re.M)}


class Progress:
    """What the setup window shows: steps, overall percentage (bytes-weighted), the current
    action, a log tail and the outcome. Mutated by the setup thread, read by the window."""

    def __init__(self, steps: list[tuple[str, float]]):
        # steps: (name, weight in bytes) so the overall bar moves with the data, not step count
        self.steps = [{"name": n, "weight": w, "done": 0.0, "status": "waiting", "detail": ""} for n, w in steps]
        self.status = "running"  # running | done | failed | cancelled
        self.error = ""
        self.log: list[str] = []
        self.lock = threading.Lock()

    def step(self, i: int, **kw) -> None:
        with self.lock:
            s = self.steps[i]
            s.update(kw)
            s["done"] = min(s["done"], s["weight"])

    def add_log(self, line: str) -> None:
        with self.lock:
            self.log.append(line.rstrip())
            del self.log[:-400]  # the window shows a tail; the full log is in autodeck.log

    def snapshot(self) -> dict:
        with self.lock:
            total = sum(s["weight"] for s in self.steps) or 1
            done = sum(s["weight"] if s["status"] == "done" else s["done"] for s in self.steps)
            return {"status": self.status, "error": self.error, "percent": round(100 * done / total, 1),
                    "done_bytes": done, "total_bytes": total,
                    "steps": [{k: s[k] for k in ("name", "status", "detail")} | {"percent": round(100 * s["done"] / s["weight"]) if s["weight"] else 0}
                              for s in self.steps],
                    "log": "\n".join(self.log[-200:])}


# --- state on disk -------------------------------------------------------------------------
def read_marker() -> dict | None:
    try:
        return json.loads(marker_file().read_text())
    except (OSError, ValueError):
        return None


def want() -> dict:
    proj = tomllib.loads((app_dir() / "pyproject.toml").read_text())
    lock = hashlib.sha256((app_dir() / "uv.lock").read_bytes()).hexdigest()
    return expected(proj["project"]["version"], lock, platform.python_version())


def models_ok() -> bool:
    return all((models_dir() / rel).is_file() and (models_dir() / rel).stat().st_size == size
               for rel, _, _, size in MODELS)


def reasons() -> list[str]:
    return setup_reasons(read_marker(), want(), venv_python().exists(), models_ok())


def request_repair() -> None:
    """Settings -> Repair: the next launch re-runs setup with `uv sync --reinstall` and
    re-checks every model's sha256."""
    config.write_json_atomic(marker_file(), {**(read_marker() or {}), "repair": True})


def status() -> dict:
    """For Settings: is this the standalone app, and what's installed."""
    if not ACTIVE:
        return {"standalone": False, "python": platform.python_version()}
    m = read_marker() or {}
    size = 0
    for root in (runtime(), models_dir()):
        for dirpath, _, files in os.walk(root):
            for f in files:
                try:
                    size += os.lstat(os.path.join(dirpath, f)).st_size
                except OSError:
                    pass
    return {"standalone": True, "python": platform.python_version(), "app_version": m.get("app_version"),
            "installed_at": m.get("installed_at"), "size": size, "location": str(config.app_home())}


# --- step 1: bootstrap on the bundled Python (stdlib only) ----------------------------------
def uv_env() -> dict:
    caches = config.caches_dir()
    return {**os.environ,
            "UV_PROJECT_ENVIRONMENT": str(runtime() / "venv"),
            "UV_CACHE_DIR": str(caches / "uv"),
            "UV_PYTHON": str(runtime_python()),
            "UV_PYTHON_DOWNLOADS": "never",  # use our Python, never fetch another
            "UV_NO_CONFIG": "1",             # ignore any uv.toml on this machine
            "NO_COLOR": "1"}


def uv_cmd(*args: str) -> list[str]:
    return [str(resources() / "bin" / ("uv" + EXE)), "sync", "--project", str(app_dir()), "--frozen", *args]


def _python_copied() -> bool:
    stamp = runtime() / "python" / ".from"
    return runtime_python().exists() and stamp.exists() and stamp.read_text() == platform.python_version()


def bootstrap() -> None:
    """Contents/MacOS/autodeck runs this on the bundled Python. Never returns."""
    rt = runtime()
    rt.mkdir(parents=True, exist_ok=True)
    config.log_file().parent.mkdir(parents=True, exist_ok=True)
    log = open(config.log_file(), "a", buffering=1, encoding="utf-8")  # noqa: SIM115
    print(f"--- {time.ctime()} autodeck {resources()}", file=log)
    if not _python_copied():
        # Copy, don't point at the bundle: the venv must survive the app being moved/updated.
        # Signatures are embedded in the binaries, so the copy stays Developer ID signed.
        tmp = rt / "python.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(Path(sys.base_prefix), tmp, symlinks=True)
        (tmp / ".from").write_text(platform.python_version())
        shutil.rmtree(rt / "python", ignore_errors=True)
        shutil.rmtree(rt / "venv", ignore_errors=True)  # its interpreter is gone
        os.replace(tmp, rt / "python")
    if not venv_python().exists() or not list(site_packages().glob("webview")):
        # Just enough for the setup window. No window yet, so say something in Notification Center.
        _notify("Preparing first launch…")
        r = subprocess.run(uv_cmd("--only-group", "ui", "--inexact"), env=uv_env(), stdout=log, stderr=log, **QUIET)
        if r.returncode != 0:
            _alert("autodeck couldn't start setup", f"Is this computer online? Details: {config.log_file()}")
            sys.exit(1)
    log.close()
    argv = [str(venv_pythonw()), *python_flags(), "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); from autodeck.desktop import main; main()", str(app_dir())]
    if WIN:  # no real exec on Windows: run the app as a child and pass its exit code on
        sys.exit(subprocess.call(argv))
    os.execv(argv[0], argv)


def _notify(text: str) -> None:
    """A passing notification before any window exists (macOS Notification Center; Windows has
    no equivalent without extra packages, and the setup window follows within seconds)."""
    if sys.platform == "darwin":
        subprocess.run(["/usr/bin/osascript", "-e", f'display notification "{text}" with title "autodeck"'], check=False)


def _alert(title: str, text: str) -> None:
    """A modal error box before any window exists."""
    if WIN:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)  # MB_ICONERROR
    elif sys.platform == "darwin":
        subprocess.run(["/usr/bin/osascript", "-e", f'display alert "{title}" message "{text}"'], check=False)
    else:
        print(f"{title}: {text}", file=sys.stderr)


def python_flags() -> list[str]:
    # -I: ignore PYTHON* env vars and user site-packages. pycache_prefix: compiled bytecode goes
    # to the caches folder, never next to the sources inside the signed app.
    return ["-I", "-X", f"pycache_prefix={config.caches_dir() / 'pycache'}"]


def relaunch_command() -> tuple[list[str], dict]:
    """A detached shell that reopens the app 2 s after this process has quit (Settings -> Repair):
    the command and the Popen keyword arguments that detach it from us."""
    root = resources().parent  # Contents/ in autodeck.app; the unzipped app folder on Windows
    if WIN:  # a hidden console (timeout needs one), its own process group: our exit never takes it down
        return (["cmd", "/c", f'timeout /t 2 /nobreak >nul & start "" "{root / "autodeck.cmd"}"'],
                {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP})
    return ["/bin/sh", "-c", f'sleep 2; /usr/bin/open "{root.parent}"'], {"start_new_session": True}


# --- step 2: the environment of the running app (venv Python) -------------------------------
def environment() -> None:
    """PATH for the tools, writable caches, logging, CA certificates, yt-dlp updates."""
    import logging

    config.log_file().parent.mkdir(parents=True, exist_ok=True)
    out = open(config.log_file(), "a", buffering=1, encoding="utf-8")  # noqa: SIM115 (lives as long as the app)
    sys.stdout = sys.stderr = out
    import faulthandler
    faulthandler.enable(out)  # a native crash (torch, essentia, numba) leaves a traceback in the log
    logging.basicConfig(stream=out, level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")

    rt = runtime()
    # Ours first (rubberband in the app, ffmpeg, deno in the venv); then /usr/bin:/bin for the rest on
    # macOS, or the system PATH on Windows (System32 and the WebView2 runtime must stay reachable).
    rest = os.environ.get("PATH", "") if WIN else "/usr/bin:/bin:/usr/sbin:/sbin"
    os.environ["PATH"] = os.pathsep.join([str(resources() / "bin"), str(rt / "bin"), str(venv_bin()), rest])
    caches = config.caches_dir()
    for var, sub in (("NUMBA_CACHE_DIR", "numba"), ("MPLCONFIGDIR", "matplotlib"),
                     ("TORCH_HOME", "torch"), ("HF_HOME", "huggingface")):
        os.environ.setdefault(var, str(caches / sub))
    after_packages()


def after_packages() -> None:
    """The parts of the environment that need the installed packages: at every launch, and on
    first launch again right after setup installed them."""
    # First launch: setup installed the packages into this very process's venv. Dirs that new
    # .pth files add (pywin32's win32/ and win32/lib, which numba imports on Windows) and new
    # packages in already-cached directories are picked up only if asked; without this the
    # engine import failed with "No module named pywintypes" and Retry could not fix it.
    import importlib
    import site
    site.addsitedir(str(site_packages()))
    importlib.invalidate_caches()
    rt = runtime()
    ff = sorted((site_packages() / "imageio_ffmpeg" / "binaries").glob("ffmpeg-*"))
    if ff:  # yt-dlp and audioread look for a program called "ffmpeg" on PATH
        (rt / "bin").mkdir(parents=True, exist_ok=True)
        link = rt / "bin" / ("ffmpeg" + EXE)
        if WIN:  # a symlink needs developer mode or admin rights on Windows: copy it once (~90 MB)
            if not link.exists() or link.stat().st_size != ff[-1].stat().st_size:
                shutil.copy2(ff[-1], link)
        elif not link.is_symlink() or os.readlink(link) != str(ff[-1]):
            link.unlink(missing_ok=True)
            link.symlink_to(ff[-1])
    config.ensure_ca_bundle()  # Windows: verify through the OS trust store (truststore), see config
    try:  # urllib (Last.fm, Deezer, PyPI, models) verifies HTTPS with certifi's CA bundle
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass  # first launch, before the packages
    for tool in ("ffmpeg", "rubberband", "deno"):
        print(f"tool {tool}: {shutil.which(tool)}", flush=True)


# --- the setup itself (runs in a thread behind the window) ------------------------------------
class Cancelled(Exception):
    pass


def lock_sizes(lock_text: str, os_name: str = sys.platform, machine: str = platform.machine()) -> dict[str, int]:
    """Download size per package for this machine (CPython 3.12), from uv.lock, so the progress
    bar can count bytes. Best effort: picks this OS and CPU's wheel (macOS arm64 / universal2,
    Windows x64, manylinux x86_64 / aarch64), else the pure-Python one, else the sdist."""
    lock = tomllib.loads(lock_text)
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    arm = machine.lower() in ("arm64", "aarch64")
    if os_name == "darwin":
        def native(f):
            return "macosx" in f and ("universal2" in f or ("arm64" if arm else "x86_64") in f)
    elif os_name == "win32":
        def native(f):
            return ("win_arm64" if arm else "win_amd64") in f
    else:
        def native(f):
            return "manylinux" in f and ("aarch64" if arm else "x86_64") in f
    sizes = {}
    for pkg in lock.get("package", []):
        wheels = pkg.get("wheels", [])

        def rank(w):
            f = w.get("url", "").rsplit("/", 1)[-1]
            py = tag in f or "abi3" in f or "-py3-" in f or "-py2.py3-" in f
            return (native(f) and py) * 2 + ("none-any" in f)
        best = max(wheels, key=rank, default=None)
        size = (best or pkg.get("sdist") or {}).get("size")
        if size:
            sizes[pkg["name"]] = int(size)
    return sizes


class Setup:
    """Runs the steps, reports into a Progress, can be cancelled and retried."""

    def __init__(self):
        self.repair = bool((read_marker() or {}).get("repair"))
        lock_text = (app_dir() / "uv.lock").read_text()
        self.sizes = lock_sizes(lock_text)
        self.pkg_bytes = float(sum(self.sizes.values()))
        self.progress = self._fresh()
        self.proc: subprocess.Popen | None = None
        self.cancel_flag = threading.Event()
        self.thread: threading.Thread | None = None

    def _fresh(self) -> Progress:
        return Progress([("Python packages", self.pkg_bytes),
                         ("AI models", float(sum(m[3] for m in MODELS))),
                         ("Finishing", 1.0)])

    def start(self, on_done) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.progress = self._fresh()
        self.cancel_flag.clear()
        self.thread = threading.Thread(target=self._run, args=(on_done,), daemon=True, name="autodeck-setup")
        self.thread.start()

    def cancel(self) -> None:
        self.cancel_flag.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def _run(self, on_done) -> None:
        p = self.progress
        try:
            self._packages(0)
            after_packages()  # ffmpeg on PATH, certifi for the model downloads
            self._models(1)
            p.step(2, status="running", detail="Checking")
            m = {**want(), "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            config.write_json_atomic(marker_file(), m)  # last: only a complete setup counts
            p.step(2, status="done", detail="")
            p.status = "done"
            on_done()
        except Cancelled:
            p.status, p.error = "cancelled", "Setup was cancelled. Retry continues where it stopped."
        except Exception as e:  # network, disk full, bad hash: show it, Retry resumes
            p.status, p.error = "failed", str(e) or e.__class__.__name__
            p.add_log(f"ERROR: {e!r}")
        for s in p.steps:
            if s["status"] == "running":
                s["status"] = "failed" if p.status == "failed" else "waiting"
        print(f"setup {p.status}: {p.error}", flush=True)

    def _packages(self, i: int) -> None:
        p = self.progress
        p.step(i, status="running", detail="Checking what's needed")
        args = ["--no-dev", "--no-install-project"] + (["--reinstall"] if self.repair else [])
        # What this sync will install (a dry run reads only the lock and the venv): the bar's
        # total is the download size of exactly those packages.
        dry = subprocess.run(uv_cmd(*args, "--dry-run"), env=uv_env(), capture_output=True, text=True, **QUIET)
        todo = would_install(dry.stdout + dry.stderr)
        self.pkg_bytes = float(sum(self.sizes.get(n, 0) for n in todo)) or 1.0
        p.step(i, weight=self.pkg_bytes)
        p.add_log("$ uv sync " + " ".join(args))
        self.proc = subprocess.Popen(uv_cmd(*args), env=uv_env(), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1, **QUIET)
        done = 0.0
        active: dict[str, float] = {}
        for line in self.proc.stdout:
            p.add_log(line)
            print(line.rstrip(), flush=True)
            ev = parse_uv_line(line)
            if ev is None:
                continue
            kind, pkg, size = ev
            if kind == "start":
                active[pkg] = self.sizes.get(pkg, size)
                p.step(i, detail=f"Downloading {pkg} ({size / 1e6:.0f} MB)")
            elif kind == "done":
                done += active.pop(pkg, self.sizes.get(pkg, 0))
                p.step(i, done=done, detail=f"Downloaded {pkg}" if not active else f"Downloading {next(iter(active))}")
            elif kind == "prepared":
                p.step(i, done=self.pkg_bytes * 0.97, detail="Installing")
        rc = self.proc.wait()
        if self.cancel_flag.is_set():
            raise Cancelled()
        if rc != 0:
            raise RuntimeError(f"Installing packages failed (uv exit {rc}). Is this Mac online? See the log.")
        p.step(i, status="done", done=self.pkg_bytes, detail="")

    def _models(self, i: int) -> None:
        p = self.progress
        p.step(i, status="running")
        done = 0.0
        for rel, url, sha, size in MODELS:
            dest = models_dir() / rel
            if dest.exists() and dest.stat().st_size == size and not self.repair:
                done += size
                p.step(i, done=done)
                continue
            p.step(i, detail=f"Downloading {rel}")
            part = dest.with_name(dest.name + ".part")
            resume = part.stat().st_size if part.exists() else 0
            p.add_log(f"GET {url}" + (f" (resuming at {resume} bytes)" if resume else ""))
            print(p.log[-1], flush=True)
            base = done
            download(url, dest, sha, size, lambda n: p.step(i, done=base + n), self.cancel_flag)
            done += size
            p.step(i, done=done)
        p.step(i, status="done", detail="")


def download(url: str, dest: Path, sha256: str, size: int, on_bytes, cancel: threading.Event) -> None:
    """Resumable, verified download: continues a .part file with an HTTP Range request, and
    only a file whose sha256 matches the pinned one is renamed into place."""
    if dest.exists():  # repair: keep it if it's intact
        if _sha256(dest) == sha256:
            return
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    if have < size:
        req = urllib.request.Request(url, headers={"User-Agent": "autodeck-setup", **({"Range": f"bytes={have}-"} if have else {})})
        with urllib.request.urlopen(req, timeout=60) as r:
            if have and r.status != 206:  # server ignored the Range: start over
                have = 0
            with open(part, "ab" if have else "wb") as f:
                while chunk := r.read(1 << 20):
                    if cancel.is_set():
                        raise Cancelled()
                    f.write(chunk)
                    have += len(chunk)
                    on_bytes(have)
    if _sha256(part) != sha256:
        part.unlink()
        raise RuntimeError(f"{dest.name}: download corrupted (sha256 mismatch); Retry downloads it again")
    os.replace(part, dest)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()
