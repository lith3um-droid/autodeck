"""autodeck.compat.danceability against essentia's Danceability.

Part A (always): sanity on synthetic signals and the edge cases, with the values the C++ gives.
Part B (where `import essentia.standard` works, i.e. the Mac/Linux run): the port against real
essentia on the same signals, so the upstream test run validates what Windows relies on.
"""

import time

import numpy as np
import pytest

from autodeck.compat.danceability import danceability, tau_frames

try:
    import essentia.standard as es
except ImportError:  # Windows: no wheels, part B is skipped
    es = None

SR = 44100  # the call site loads at 44.1 kHz, so synthesize there (conftest's make_click_signal is 22050)


# --- synthetic signals --------------------------------------------------------

def _kick(dur=0.25, amp=1.0):
    t = np.arange(int(dur * SR)) / SR
    f = 45 + 115 * np.exp(-t * 30)  # pitch drop 160 -> 45 Hz
    return amp * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 12)


def _burst(rng, dur, tau, amp):
    n = int(dur * SR)
    return amp * rng.standard_normal(n) * np.exp(-np.arange(n) / (tau * SR))


def groove(bpm, seconds, swing=0.0, seed=0):
    """4/4 drums: kick on 1 and 3, snare on 2 and 4, hats on every eighth; `swing` delays the
    off-beat hats by that fraction of an eighth (0.33 is a triplet feel)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    y = np.zeros(n)
    beat = 60.0 / bpm
    kick, snare, hat = _kick(), _burst(rng, 0.2, 0.03, 0.6), _burst(rng, 0.05, 0.01, 0.25)

    def add(at, w):
        s = int(at * SR)
        seg = w[: max(0, n - s)]
        y[s : s + len(seg)] += seg

    i = 0
    while i * beat < seconds:
        add(i * beat, kick if i % 2 == 0 else snare)
        add(i * beat, hat)
        add((i + 0.5 + 0.5 * swing) * beat, hat)
        i += 1
    return np.clip(y, -1, 1).astype(np.float32)


def band(seconds, bpm=120):
    """Drums plus a chord progression (triads with 3 harmonics, one chord per bar) and an
    eighth-note bass line on the root."""
    y = groove(bpm, seconds).astype(np.float64)
    t = np.arange(len(y)) / SR
    beat = 60.0 / bpm
    roots = np.array([110.0, 146.83, 164.81, 123.47])  # A D E B
    root = roots[((t // (4 * beat)) % 4).astype(int)]
    for ratio, amp in ((1.0, 0.12), (1.25, 0.08), (1.5, 0.08)):
        for h, ha in ((1, 1.0), (2, 0.5), (3, 0.25)):
            y += amp * ha * np.sin(2 * np.pi * np.cumsum(root * ratio * h) / SR)
    env = np.exp(-(t % (beat / 2)) * 10)
    y += 0.25 * env * np.sin(2 * np.pi * np.cumsum(root / 2) / SR)
    return np.clip(y, -1, 1).astype(np.float32)


def white_noise(seconds, seed=1):
    return (0.3 * np.random.default_rng(seed).standard_normal(int(seconds * SR))).astype(np.float32)


def am_sine(seconds, carrier=220.0, rate=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * (1 + 0.8 * np.sin(2 * np.pi * rate * t)) * np.sin(2 * np.pi * carrier * t)).astype(np.float32)


def padded(seconds, pad):
    z = np.zeros(int(pad * SR), dtype=np.float32)
    return np.concatenate([z, groove(120, seconds), z])


SIGNALS = {
    "groove_120": lambda: groove(120, 30),
    "groove_95_swing": lambda: groove(95, 30, swing=0.33, seed=3),
    "band": lambda: band(30),
    "white_noise": lambda: white_noise(20),
    "am_sine": lambda: am_sine(30),
    "silence_both_ends": lambda: padded(20, 6),
    "long_240s": lambda: groove(124, 240, seed=5),
    # edge cases
    "shorter_than_min_tau": lambda: groove(120, 0.2),  # 20 frames < 31
    "all_zero": lambda: np.zeros(10 * SR, dtype=np.float32),
    "half_second": lambda: groove(120, 0.5),  # 50 frames: 6 taus fit
}
EDGE = ("shorter_than_min_tau", "all_zero", "half_second")

_cache = {}


def signal(name):
    if name not in _cache:
        _cache[name] = SIGNALS[name]()
    return _cache[name]


# --- part A: always -----------------------------------------------------------

def test_tau_list_matches_configure():
    taus = tau_frames()
    assert taus[0] == 31 and taus[-1] == 871 and len(taus) == 36  # float32 1.1 steps, int() truncation
    assert all(b > a for a, b in zip(taus, taus[1:]))


def test_groove_scores_as_danceable():
    # analyze.danceability maps raw 0.9..1.6 to 0..1 (real music, measured). A machine-steady
    # synthetic groove scores above that band (~3.8: its loudness curve is perfectly periodic, so
    # DFA sees a tiny exponent) and clamps to 1.0 there, the right end for a drum loop.
    d, dfa = danceability(signal("groove_120"), SR)
    assert d > 1.6
    assert dfa.shape == (len(tau_frames()) - 1,) and dfa.dtype == np.float32
    assert d == pytest.approx(1 / np.mean(dfa), rel=1e-5)


def test_drone_below_noise_below_groove():
    g, _ = danceability(signal("groove_120"), SR)
    n, _ = danceability(signal("white_noise"), SR)
    a, _ = danceability(signal("am_sine"), SR)
    # white noise integrates to a random walk (DFA exponent ~0.5 -> raw ~2.0), a drone's slow
    # swell to a smooth curve (exponent > 1 -> raw < 0.9, maps to 0), the groove sits highest
    assert n == pytest.approx(2.0, abs=0.15)
    assert a < 0.9 < n < g


def test_silence_at_both_ends_lowers_the_score():
    # the silent stretches put a long-range trend into the integrated loudness: larger exponent
    d, _ = danceability(signal("groove_120"), SR)
    p, _ = danceability(signal("silence_both_ends"), SR)
    assert 0.9 < p < d


@pytest.mark.parametrize("name", ["shorter_than_min_tau", "all_zero"])
def test_edge_cases_give_zero(name):
    d, dfa = danceability(signal(name), SR)
    assert d == 0.0
    assert dfa.shape == (len(tau_frames()) - 1,) and not dfa.any()


def test_half_second_uses_the_taus_that_fit():
    d, dfa = danceability(signal("half_second"), SR)
    assert np.isfinite(d) and d > 0
    assert dfa[5:].tolist() == [0.0] * (len(dfa) - 5)  # 50 frames: taus 31..49 fit, 54 does not
    assert np.all(np.isfinite(dfa[:5]))
    assert d == pytest.approx(0.423943, abs=1e-5)  # what essentia 2.1b6.dev1389 gives (port is bit-exact)


def test_accepts_float64_and_2d():
    y = signal("groove_120")
    d, dfa = danceability(y, SR)
    d64, dfa64 = danceability(y.astype(np.float64), SR)
    assert d64 == d and np.array_equal(dfa64, dfa)
    d2, _ = danceability(y[:, None], SR)
    assert d2 == d


def test_four_minutes_is_fast():
    y = signal("long_240s")
    t0 = time.perf_counter()
    danceability(y, SR)
    assert time.perf_counter() - t0 < 10


# --- part B: against essentia -------------------------------------------------

@pytest.mark.skipif(es is None, reason="essentia not installed (Windows): part B needs the reference")
@pytest.mark.parametrize("name", list(SIGNALS))
def test_matches_essentia(name):
    y = signal(name)
    ref_d, ref_dfa = es.Danceability(sampleRate=SR)(y)
    d, dfa = danceability(y, SR)
    assert dfa.shape == ref_dfa.shape
    if name in EDGE:
        assert d == ref_d
        assert np.array_equal(dfa, ref_dfa)
    else:
        assert d == pytest.approx(ref_d, rel=1e-3)
        assert np.max(np.abs(dfa - ref_dfa)) <= 1e-3
