"""Fetch the Rubber Band command-line tool autodeck renders with (pyrubberband calls `rubberband`).

macOS gets it from Homebrew (`brew install rubberband`) and the standalone app ships it. Windows has
no package-manager source, so this downloads the official build from breakfastquay.com (GPL; the
archive's sha256 is pinned here, a tampered download fails) into config.tools_dir()
(%LOCALAPPDATA%\\autodeck\\bin), which autodeck puts on PATH at startup (config.ensure_tools_on_path).

    uv run python scripts/get_rubberband.py            # installs rubberband.exe + sndfile.dll, once
    uv run python scripts/get_rubberband.py --force    # re-downloads
"""
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autodeck import config  # noqa: E402

URL = "https://breakfastquay.com/files/releases/"
# Per OS: archive, its sha256, the files to take out of it (the GPL text goes in beside them).
BUILDS = {
    "win32": ("rubberband-4.0.0-gpl-executable-windows.zip",
              "f2d47fc64dbb42f6cc62edf7933ac4fa89d8f0ef8b9cf97b6afc263a7fe05644",
              ("rubberband.exe", "sndfile.dll", "COPYING.txt")),
}


def fetch(url: str, sha256: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "autodeck-setup"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise SystemExit(f"{url}: sha256 mismatch, not installing it")
    return data


def main() -> int:
    if sys.platform not in BUILDS:
        print("Nothing to fetch on this OS: install Rubber Band with your package manager "
              "(macOS: brew install rubberband; Debian/Ubuntu: apt install rubberband-cli).")
        return 0
    archive, sha, files = BUILDS[sys.platform]
    dest = config.tools_dir()
    have = [f for f in files if (dest / f).exists()]
    if len(have) == len(files) and "--force" not in sys.argv:
        print(f"Rubber Band is already in {dest} (use --force to download it again)")
        return 0
    print(f"Downloading {URL + archive} ...")
    data = fetch(URL + archive, sha)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for f in files:
            member = next(m for m in z.namelist() if m.endswith("/" + f))
            out = dest / ("rubberband-COPYING.txt" if f == "COPYING.txt" else f)
            out.write_bytes(z.read(member))
    print(f"Installed {', '.join(f for f in files if f != 'COPYING.txt')} into {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
