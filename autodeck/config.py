"""Paths and spec constants. Everything tunable lives here."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Spec values (see docs/superpowers/specs/2026-10-30-autodeck-design.md)
WINDOW_SIZE = 10          # songs analysed/rendered ahead in the rolling window
# Hard ceiling on ready-but-unplayed songs of ANY genre. WINDOW_SIZE counts only the active
# genre, and a song's genre is unknown until it's downloaded and analysed, so a rare genre
# made the worker download the whole pool hunting for matches (140 songs / 9 GB in real use).
MAX_READY_AHEAD = 20
FIXED_NEXT = 2            # upcoming songs whose order is locked
STEMS_AHEAD = 3           # Up next songs (after the ON AIR one) that get vocal/instrumental stems
# Up next is the owner's list (kept songs); below this many ready songs the engine tops it up
# with auto picks from the active source so the music never stops.
MIN_UPCOMING = 3
BEATMATCH_MAX_PCT = 0.08  # max tempo stretch to beatmatch (8%)
EASE_BARS = 16            # bars over which tempo eases back to the song's own
PHRASE_BARS = 8           # a phrase boundary = any multiple of 8 bars from the first downbeat; prefer 16

DEFAULT_GENRES = ["Pop", "Hip-hop", "Romanian hip-hop", "Romanian sârbe"]
ANY_GENRE = "Any"  # pseudo-genre: no filtering


# Where the app keeps its data: each OS's own convention, not a loose folder in the home dir,
# and outside the repo on purpose (audio must never be committed).
#   data    library/ (songs, analyses, moods, session, history, models), settings.json, the
#           standalone app's runtime/ and models/
#   caches  Last.fm / Deezer lookups, uv's package cache, numba/torch caches (safe to delete)
#   logs    autodeck.log of the desktop window and the standalone app
#   macOS    ~/Library/Application Support/autodeck   ~/Library/Caches/autodeck    ~/Library/Logs
#   Windows  %LOCALAPPDATA%\autodeck                  %LOCALAPPDATA%\autodeck\cache  %LOCALAPPDATA%\autodeck\logs
#   Linux    $XDG_DATA_HOME/autodeck (~/.local/share) $XDG_CACHE_HOME/autodeck      $XDG_STATE_HOME/autodeck


def _platform_dirs() -> tuple[Path, Path, Path]:
    """(data, caches, logs) for this OS."""
    home = Path.home()
    if sys.platform == "darwin":
        lib = home / "Library"
        return lib / "Application Support" / "autodeck", lib / "Caches" / "autodeck", lib / "Logs"
    if sys.platform == "win32":
        local = Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
        return local / "autodeck", local / "autodeck" / "cache", local / "autodeck" / "logs"
    data = Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share")
    cache = Path(os.environ.get("XDG_CACHE_HOME") or home / ".cache")
    state = Path(os.environ.get("XDG_STATE_HOME") or home / ".local" / "state")
    return data / "autodeck", cache / "autodeck", state / "autodeck"


APP_SUPPORT, CACHES, LOGS = _platform_dirs()
LEGACY_LIBRARY = Path.home() / "autodeck-library"  # before 2026-10-31


def library_root() -> Path:
    # Read the env var on every call (not at import) so tests can monkeypatch it.
    return Path(os.environ.get("AUTODECK_LIBRARY") or APP_SUPPORT / "library")


def cache_root() -> Path:
    """Re-fetchable lookups. With AUTODECK_LIBRARY set (tests, experiments) they stay beside
    that library instead of mixing into the real ~/Library/Caches."""
    if os.environ.get("AUTODECK_LIBRARY"):
        return library_root() / "cache"
    return CACHES


def migrate_legacy_library() -> str | None:
    """Move a pre-2026-10-31 ~/autodeck-library to the app's data folder, once, at startup.
    A rename on the same disk: instant, no copy, nothing duplicated. The lookup caches go to
    the caches folder. Returns a message when something moved."""
    if os.environ.get("AUTODECK_LIBRARY") or not LEGACY_LIBRARY.is_dir():
        return None
    new = library_root()
    if new.exists():
        return None  # never merge two libraries silently; the new one wins
    new.parent.mkdir(parents=True, exist_ok=True)
    os.rename(LEGACY_LIBRARY, new)
    for name in ("lastfm-cache", "deezer-cache"):
        old = new / name
        if old.is_dir() and not (CACHES / name).exists():
            CACHES.mkdir(parents=True, exist_ok=True)
            os.rename(old, CACHES / name)
    return f"Moved your library from {LEGACY_LIBRARY} to {new}"


def data_dir() -> Path:
    """App-level files (settings.json, lastfm.env, pylib/): Application Support, or beside a
    test library when AUTODECK_LIBRARY is set so tests never touch the owner's real files."""
    return library_root() if os.environ.get("AUTODECK_LIBRARY") else APP_SUPPORT


def app_home() -> Path:
    """The standalone app's own files: runtime/ (Python + packages), models/, setup marker.
    AUTODECK_HOME moves it (clean-machine tests); read live so tests can monkeypatch it."""
    return Path(os.environ.get("AUTODECK_HOME") or APP_SUPPORT)


def caches_dir() -> Path:
    """Re-downloadable caches of the standalone app (uv's package cache, numba, pycache).
    AUTODECK_CACHES moves it, for the same reason as AUTODECK_HOME."""
    return Path(os.environ.get("AUTODECK_CACHES") or CACHES)


def log_file() -> Path:
    """autodeck.log: the desktop window and the standalone app log there (no terminal)."""
    return LOGS / "autodeck.log"


def tools_dir() -> Path:
    """Command-line tools autodeck needs that no package manager provides on this OS, put
    there by scripts/get_rubberband.py (Windows: rubberband.exe + sndfile.dll). The standalone
    app ships its own copy in its resources instead."""
    return app_home() / "bin"


def ensure_tools_on_path() -> None:
    """Prepend tools_dir() to PATH once, so pyrubberband's `rubberband` call finds it."""
    d = tools_dir()
    if d.is_dir() and str(d) not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def ensure_ca_bundle() -> None:
    """Windows: verify HTTPS (urllib: model downloads, Last.fm, Deezer, PyPI) through the OS trust
    store. The uv-managed Python verifies with OpenSSL against the roots Windows has cached (and
    certifi's list, in the standalone app), and the Discogs model download from essentia.upf.edu
    failed that way: its chain ends in a HARICA root neither resolved, while Windows' own stack
    (SChannel) fetches what it is missing. truststore, what pip uses, makes Python's ssl do the
    same. A no-op elsewhere: macOS works as before."""
    if sys.platform != "win32":
        return
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:  # first launch of the standalone app, before the packages: nothing to download yet
        pass
    # The Hugging Face cache can't symlink without developer mode and says so twice, loudly; it works regardless.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def installed_model(name: str) -> Path | None:
    """A model the standalone app's setup downloaded (models/ in app_home()), so analysis never
    downloads at song time. None when absent (dev): callers then download as before."""
    p = app_home() / "models" / name
    return p if p.exists() else None


# Module-level alias requested by the spec; resolved at import time, so code that
# must honor AUTODECK_LIBRARY overrides at runtime should call library_root().
LIBRARY: Path = library_root()


def write_json_atomic(path: Path, data) -> None:
    """Write JSON via temp file + os.replace so a crash never leaves truncated JSON.

    The background worker writes these files constantly; readers must only ever
    see the old or the new complete file. Temp lives in the same dir so the
    replace stays on one filesystem (atomic).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp name per call: the worker thread and the server thread can write
    # the same song.json at once, and a shared "<name>.tmp" would let them clobber
    # each other's half-written file. os.replace makes the last writer win, whole.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        for attempt in range(50):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                # Windows: replacing a file that another thread is reading or replacing at that very
                # moment is refused (no FILE_SHARE_DELETE), not queued. Both are over in a moment.
                if attempt == 49:
                    raise
                time.sleep(0.01)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _genres_file() -> Path:
    return library_root() / "genres.json"


def genres() -> list[str]:
    """Owner-editable genre list (without "Any"); falls back to the defaults."""
    try:
        return json.loads(_genres_file().read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):  # missing or corrupt -> defaults
        return list(DEFAULT_GENRES)


def save_genres(items: list[str]) -> None:
    write_json_atomic(_genres_file(), items)


# --- owner settings (settings.json) -------------------------------------------------
# Where yt-dlp reads the YouTube login from. Keys are yt-dlp's --cookies-from-browser names.
BROWSERS = {"chrome": "Chrome", "safari": "Safari", "firefox": "Firefox", "brave": "Brave",
            "edge": "Edge", "none": "none"}
DEFAULT_BROWSER = "chrome"


def _settings_file() -> Path:
    return data_dir() / "settings.json"


def settings() -> dict:
    try:
        d = json.loads(_settings_file().read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (FileNotFoundError, ValueError):  # missing or corrupt -> defaults
        return {}


def save_setting(key: str, value) -> None:
    write_json_atomic(_settings_file(), {**settings(), key: value})


def youtube_browser() -> str:
    """AUTODECK_BROWSER (env) > Settings > chrome. An unknown saved value falls back to chrome."""
    env = os.environ.get("AUTODECK_BROWSER")
    if env:
        return env.lower()  # free-form on purpose: any yt-dlp browser name (vivaldi, opera, ...)
    saved = settings().get("youtube_browser")
    return saved if saved in BROWSERS else DEFAULT_BROWSER
