"""Build the STANDALONE autodeck for Windows: a portable folder, zipped, that installs its heavy
parts on first launch. The Windows counterpart of scripts/build_standalone.sh (same idea, no
signing: Windows asks once whether to run an unsigned download).

    uv run python scripts/build_standalone_windows.py     # -> dist/autodeck/ + dist/autodeck-<version>-windows-x64.zip

What goes in the folder:
    autodeck.cmd                 launcher -> autodeck.standalone.bootstrap() on the bundled Python (pythonw: no console)
    Resources/python/            python-build-standalone CPython 3.12 (the Python uv manages), trimmed
    Resources/bin/uv.exe         official uv (PyPI wheel, sha256 checked against PyPI)
    Resources/bin/rubberband.exe + sndfile.dll   official Rubber Band CLI (breakfastquay.com, pinned sha256)
    Resources/app/               autodeck/, web/, design/icon, pyproject.toml, uv.lock, .python-version
First launch installs the rest (packages from uv.lock, models) into %LOCALAPPDATA%\\autodeck,
see autodeck/standalone.py. Needs on the build PC only: uv with its managed CPython 3.12
(`uv python install 3.12`) and this repo synced (`uv sync`).
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "build"
DIST = REPO / "dist"
APP = DIST / "autodeck"
RES = APP / "Resources"
# Pinned third-party binaries. Bump deliberately; the hashes make a tampered download fail the build.
UV_VERSION = "0.13.7"
RUBBERBAND_URL = "https://breakfastquay.com/files/releases/rubberband-4.0.0-gpl-executable-windows.zip"
RUBBERBAND_SHA256 = "f2d47fc64dbb42f6cc62edf7933ac4fa89d8f0ef8b9cf97b6afc263a7fe05644"
# What autodeck never uses, dropped from the bundled Python: GUI toolkits, tests, pip (uv installs), headers.
PYTHON_TRIM = ["Lib/test", "Lib/idlelib", "Lib/tkinter", "Lib/turtledemo", "Lib/ensurepip", "Lib/lib2to3",
               "tcl", "include", "Scripts", "DLLs/_tkinter.pyd", "DLLs/tcl86t.dll", "DLLs/tk86t.dll"]

LAUNCHER = r"""@echo off
rem autodeck standalone launcher: first-launch setup + start, see autodeck/standalone.py.
rem -I ignores PYTHON* env vars and user site-packages; bytecode goes to the caches folder,
rem never into this folder. pythonw: no console window (the app logs to %LOCALAPPDATA%\autodeck\logs).
setlocal
set "RES=%~dp0Resources"
set "AUTODECK_RESOURCES=%RES%"
if defined AUTODECK_CACHES (set "PYC=%AUTODECK_CACHES%\pycache") else (set "PYC=%LOCALAPPDATA%\autodeck\cache\pycache")
start "" "%RES%\python\pythonw.exe" -I -X "pycache_prefix=%PYC%" -c "import sys; sys.path.insert(0, sys.argv[1]); from autodeck import standalone; standalone.bootstrap()" "%RES%\app"
"""


def step(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def run(*cmd, **kw) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout.strip()


def download(url: str, sha256: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "autodeck-build"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise SystemExit(f"{url}: sha256 mismatch")
    return data


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("This builds the Windows app; run it on Windows (scripts/build_standalone.sh is the Mac build).")
    uv = os.environ.get("UV") or shutil.which("uv")  # `uv run` sets UV to its own path
    if not uv:
        raise SystemExit("uv not found: run this with `uv run python scripts/build_standalone_windows.py`")
    os.chdir(REPO)
    version = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    run(uv, "lock", "--check")  # the bundled uv.lock must match pyproject.toml, or first-launch `uv sync --frozen` lies
    shutil.rmtree(APP, ignore_errors=True)
    for d in (APP, RES / "bin", RES / "app", BUILD):
        d.mkdir(parents=True, exist_ok=True)

    step("Python (python-build-standalone, as managed by uv)")
    # --no-project: not this repo's .venv. base_prefix: the real install dir behind the version link.
    # UV_PYTHON, when set, names the interpreter outright (for a machine where uv's version link is unusable).
    py = os.environ.get("UV_PYTHON") or run(uv, "python", "find", "3.12", "--managed-python", "--no-project")
    pyhome = Path(run(py, "-c", "import os,sys;print(os.path.realpath(sys.base_prefix))"))
    print(f"from {pyhome}")
    shutil.copytree(pyhome, RES / "python")
    for rel in PYTHON_TRIM:
        p = RES / "python" / rel
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    for p in list((RES / "python").rglob("__pycache__")):
        shutil.rmtree(p, ignore_errors=True)
    # Empty site-packages: everything is installed into the venv. (A managed Python on a dev PC can
    # have had things pip-installed into it; none of that may ship.)
    for p in (RES / "python" / "Lib" / "site-packages").iterdir():
        if p.name != "README.txt":
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
    print(run(str(RES / "python" / "python.exe"), "-I", "-B", "-c", "import ssl, sqlite3, ctypes, zlib; print('python ok')"))

    step(f"uv {UV_VERSION} (official wheel from PyPI, sha256 checked)")
    rel = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/uv/{UV_VERSION}/json", timeout=60))
    w = next(u for u in rel["urls"] if u["filename"] == f"uv-{UV_VERSION}-py3-none-win_amd64.whl")
    data = download(w["url"], w["digests"]["sha256"])
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        (RES / "bin" / "uv.exe").write_bytes(z.read(f"uv-{UV_VERSION}.data/scripts/uv.exe"))
    print(run(str(RES / "bin" / "uv.exe"), "--version"))

    step("rubberband CLI (official build, pinned sha256)")
    tb = BUILD / Path(RUBBERBAND_URL).name
    if not tb.exists():
        tb.write_bytes(download(RUBBERBAND_URL, RUBBERBAND_SHA256))
    elif hashlib.sha256(tb.read_bytes()).hexdigest() != RUBBERBAND_SHA256:
        raise SystemExit(f"{tb}: sha256 mismatch; delete it and build again")
    with zipfile.ZipFile(tb) as z:
        for f in ("rubberband.exe", "sndfile.dll", "COPYING.txt"):
            member = next(m for m in z.namelist() if m.endswith("/" + f))
            (RES / "bin" / ("rubberband-COPYING.txt" if f == "COPYING.txt" else f)).write_bytes(z.read(member))

    step("autodeck source + lock")
    shutil.copytree(REPO / "autodeck", RES / "app" / "autodeck", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO / "web", RES / "app" / "web", ignore=shutil.ignore_patterns("*.test.mjs"))
    shutil.copytree(REPO / "design" / "icon", RES / "app" / "design" / "icon")
    for f in ("pyproject.toml", "uv.lock", ".python-version", "LICENSE"):
        shutil.copy2(REPO / f, RES / "app" / f)

    step("launcher")
    (APP / "autodeck.cmd").write_text(LAUNCHER, encoding="ascii", newline="\r\n")
    (APP / "README.txt").write_text(
        f"autodeck {version} for Windows (x64)\n\n"
        "Double-click autodeck.cmd. The first launch downloads about 1.5 GB of audio libraries and models\n"
        "into %LOCALAPPDATA%\\autodeck (a progress window shows it; it resumes if interrupted).\n"
        "Windows may ask once whether to run this unsigned download: More info > Run anyway.\n"
        "Log: %LOCALAPPDATA%\\autodeck\\logs\\autodeck.log. Source: https://github.com/bvdr/autodeck\n",
        encoding="utf-8", newline="\r\n")

    step("zip")
    out = DIST / f"autodeck-{version}-windows-x64"
    Path(str(out) + ".zip").unlink(missing_ok=True)
    shutil.make_archive(str(out), "zip", root_dir=DIST, base_dir="autodeck")

    step("done")
    size = sum(p.stat().st_size for p in APP.rglob("*") if p.is_file())
    print(f"{APP}  {size / 1e6:.0f} MB unpacked\n{out}.zip  {Path(str(out) + '.zip').stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
