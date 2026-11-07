"""autodeck.compat.key (numpy port of essentia's KeyExtractor) against essentia itself.

Part A runs everywhere: sanity checks on clean synthetic chords (expected keys are what essentia
2.1b6 returns for them, re-checked by part B). Part B runs only where `import essentia.standard`
works (macOS/Linux) and asserts the port returns the same (key, scale) as essentia and a strength
within 1e-3 on every validation signal: chord progressions in a dozen keys and modes with
harmonics, melodies, bass lines, drums, vibrato and detuning, plus white noise, a bare click
track, a 3 s and a 95 s signal. The port is not bit-exact (float64 numpy FFT vs essentia's float32
KISS FFT), hence the strength tolerance; measured gaps are <= 2e-5 and no signal needed a looser
one. Single frames can differ more (a bass peak barely over the 1e-4 threshold has a nearly flat
parabola, so its interpolated frequency moves ~0.02 Hz under FFT rounding and the HPCP cosine
weight amplifies that), so the per-frame HPCP check only asks that most frames agree and that
the mean, which Key consumes, agrees to 1e-3. Known unmatchable inputs, not tested: a lone impulse
or a bare Nyquist tone, whose spectrum is flat where essentia's "peaks" are its FFT's rounding
noise (the port finds none and reports A minor 0.0).
"""
import numpy as np
import pytest

from autodeck.compat import key as compat_key
from autodeck.music import to_camelot

try:
    import essentia.standard as es
except ImportError:  # Windows: no essentia wheels; part B is skipped
    es = None

SR = 22050
NOTE = {"C": 0, "C#": 1, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7,
        "G#": 8, "Ab": 8, "A": 9, "Bb": 10, "B": 11}


# --- synthetic signals ---------------------------------------------------------------------------

def _midi_hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _chord_midi(root: str, quality: str, octave: int = 4) -> list[int]:
    base = NOTE[root] + 12 * (octave + 1)
    return [base + i for i in {"maj": (0, 4, 7), "min": (0, 3, 7)}[quality]]


def _tone(freq: float, n: int, harmonics: int = 6, rolloff: float = 0.6, vibrato: float = 0.0,
          decay: float = 0.8) -> np.ndarray:
    """Harmonic tone: `harmonics` partials with amplitude rolloff**k, 10 ms attack, exponential
    decay; `vibrato` is the relative frequency modulation depth at 5.5 Hz."""
    t = np.arange(n) / SR
    if vibrato:
        phase = 2 * np.pi * np.cumsum(freq * (1 + vibrato * np.sin(2 * np.pi * 5.5 * t))) / SR
    else:
        phase = 2 * np.pi * freq * t
    y = sum(rolloff ** k * np.sin((k + 1) * phase + 0.37 * k) for k in range(harmonics))
    return y * np.minimum(1.0, t / 0.01) * np.exp(-decay * t)


def _drums(n: int, bpm: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    t = np.arange(int(0.25 * SR)) / SR
    kick = np.sin(2 * np.pi * np.cumsum(45 + 110 * np.exp(-t * 30)) / SR) * np.exp(-t * 12)
    burst = lambda d, tau: rng.standard_normal(int(d * SR)) * np.exp(-np.arange(int(d * SR)) / (tau * SR))  # noqa: E731
    snare, hat = 0.5 * burst(0.15, 0.04), 0.15 * burst(0.04, 0.008)
    y = np.zeros(n)
    for i in range(int(n / SR / beat)):
        s = int(i * beat * SR)
        for off, w in [(0, kick if i % 2 == 0 else snare), (int(beat / 2 * SR), hat)]:
            seg = w[: max(0, n - s - off)]
            y[s + off : s + off + len(seg)] += seg
    return y


def progression(chords: list[tuple[str, str]], seconds: float, bpm: float = 120.0, beats_per_chord: int = 4,
                octave: int = 4, melody: bool = False, bass: bool = False, drums: bool = False,
                vibrato: float = 0.0, detune_cents: float = 0.0, seed: int = 0) -> np.ndarray:
    """Chord progression of harmonic-rich tones, optionally with an arpeggiated melody an octave
    up, a bass line two octaves down and a kick/snare/hat pattern. float32, peak 0.8."""
    n = int(seconds * SR)
    y = np.zeros(n)
    chord_len = int(60.0 / bpm * beats_per_chord * SR)
    ratio = 2 ** (detune_cents / 1200)
    pos, k = 0, 0
    while pos < n:
        root, quality = chords[k % len(chords)]
        seg = min(chord_len, n - pos)
        notes = _chord_midi(root, quality, octave)
        for m in notes:
            y[pos:pos + seg] += 0.2 * _tone(_midi_hz(m) * ratio, seg, vibrato=vibrato)
        if bass:
            y[pos:pos + seg] += 0.3 * _tone(_midi_hz(notes[0] - 24) * ratio, seg, harmonics=2, decay=0.3)
        if melody:
            step = int(60.0 / bpm / 2 * SR)
            for j, p in enumerate(range(pos, pos + seg, step)):
                m = notes[(j * 2) % 3] + 12 + (2 if j % 4 == 3 else 0)
                y[p:p + min(step, pos + seg - p)] += 0.25 * _tone(_midi_hz(m) * ratio, min(step, pos + seg - p),
                                                                 harmonics=4, decay=3.0)
        pos += seg
        k += 1
    if drums:
        y += _drums(n, bpm, seed)
    return (0.8 * y / max(np.abs(y).max(), 1e-9)).astype(np.float32)


def triad(root: str, quality: str, seconds: float = 20.0, octave: int = 3) -> np.ndarray:
    """Plain sine triad like tests/conftest.py's a_minor_triad fixture."""
    t = np.arange(int(seconds * SR)) / SR
    y = sum(np.sin(2 * np.pi * _midi_hz(m) * t) for m in _chord_midi(root, quality, octave)) / 3
    return (0.5 * y).astype(np.float32)


def a_minor_triad_signal() -> np.ndarray:
    """Exactly the conftest a_minor_triad fixture's samples (220, 261.63, 329.63 Hz, 20 s)."""
    t = np.arange(20 * SR) / SR
    y = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 261.63, 329.63)) / 3
    return (0.5 * y).astype(np.float32)


_I_IV_V = lambda a, b, c, d: [(a, "maj"), (b, "maj"), (c, "maj"), (d, "maj")]  # noqa: E731

VALIDATION_SIGNALS = {
    # name: builder. Keys and modes: C D E F F# G A Bb B Eb major/minor across the set.
    "a_minor_triad": a_minor_triad_signal,
    "c_major_triad": lambda: triad("C", "maj"),
    "g_major_triad": lambda: triad("G", "maj"),
    "e_minor_triad": lambda: triad("E", "min"),
    "f_minor_triad": lambda: triad("F", "min"),
    "c_major": lambda: progression(_I_IV_V("C", "F", "G", "C"), 30),
    "a_minor": lambda: progression([("A", "min"), ("D", "min"), ("E", "maj"), ("A", "min")], 30),
    "g_major_melody": lambda: progression([("G", "maj"), ("E", "min"), ("C", "maj"), ("D", "maj")], 30, melody=True),
    "e_minor_bass": lambda: progression([("E", "min"), ("A", "min"), ("B", "maj"), ("E", "min")], 30, bass=True),
    "d_major_drums": lambda: progression([("D", "maj"), ("A", "maj"), ("B", "min"), ("G", "maj")], 30, bass=True, drums=True),
    "f_minor": lambda: progression([("F", "min"), ("Bb", "min"), ("C", "maj"), ("F", "min")], 30, octave=3),
    "bb_major_melody": lambda: progression(_I_IV_V("Bb", "Eb", "F", "Bb"), 30, melody=True, bpm=100),
    "fsharp_minor_vibrato": lambda: progression([("F#", "min"), ("B", "min"), ("C#", "maj"), ("F#", "min")], 30, vibrato=0.006),
    "eb_major_detuned": lambda: progression(_I_IV_V("Eb", "Ab", "Bb", "Eb"), 30, detune_cents=12.0),
    "csharp_minor_drums": lambda: progression([("C#", "min"), ("F#", "min"), ("G#", "maj"), ("C#", "min")], 30, drums=True, seed=3),
    "e_major_melody_bass": lambda: progression(_I_IV_V("E", "A", "B", "E"), 30, melody=True, bass=True, bpm=128),
    "a_major_vibrato_detuned": lambda: progression(_I_IV_V("A", "D", "E", "A"), 30, vibrato=0.004, detune_cents=-9.0),
    "d_minor_short": lambda: progression([("D", "min"), ("G", "min"), ("A", "maj"), ("D", "min")], 3.0, bpm=180),
    "b_minor_long_full": lambda: progression([("B", "min"), ("E", "min"), ("F#", "maj"), ("B", "min")], 95,
                                             melody=True, bass=True, drums=True, seed=7),
    "white_noise": lambda: (0.3 * np.random.default_rng(1).standard_normal(30 * SR)).astype(np.float32),
    "click_track": lambda: _drums(30 * SR, 120, 5).astype(np.float32),
}

_cache: dict[str, np.ndarray] = {}


def signal(name: str) -> np.ndarray:
    if name not in _cache:
        _cache[name] = VALIDATION_SIGNALS[name]()
    return _cache[name]


# --- part A: always runs -------------------------------------------------------------------------

def test_a_minor_triad_fixture(a_minor_triad):
    import soundfile as sf

    y, sr = sf.read(a_minor_triad, dtype="float32")
    assert sr == SR
    key, scale, strength = compat_key.key_extractor(y, SR)
    assert (key, scale) == ("A", "minor")
    assert np.isfinite(strength) and -1.0 <= strength <= 1.0


@pytest.mark.parametrize("name, expected", [
    ("a_minor_triad", ("A", "minor")),
    ("c_major_triad", ("C", "major")),
    ("g_major_triad", ("G", "major")),
    ("e_minor_triad", ("E", "minor")),
    ("f_minor_triad", ("F", "minor")),
    ("c_major", ("C", "major")),
    ("a_minor", ("A", "minor")),
    ("e_minor_bass", ("E", "minor")),
    ("f_minor", ("F", "minor")),
    ("bb_major_melody", ("Bb", "major")),
    ("d_minor_short", ("D", "minor")),
])
def test_clean_progressions(name, expected):
    """The keys here are what essentia 2.1b6 itself returns for these signals (checked in part B)."""
    key, scale, strength = compat_key.key_extractor(signal(name), SR)
    assert (key, scale) == expected
    assert np.isfinite(strength) and -1.0 <= strength <= 1.0


def test_key_names_work_with_to_camelot():
    for k in compat_key.KEY_NAMES:
        for scale in ("major", "minor"):
            assert to_camelot(f"{k} {scale}")


def test_empty_signal_raises_like_essentia():
    # essentia: EssentiaException("KeyExtractor: empty input signal"); analyze._key catches it
    with pytest.raises(ValueError, match="empty input signal"):
        compat_key.key_extractor(np.zeros(0, dtype=np.float32), SR)


def test_silence_and_short_signals_do_not_raise():
    # silence: no spectral peak clears the 1e-4 threshold, the HPCP is flat and key.cpp's
    # correlation returns 0 everywhere, which it resolves to ("A", "minor", 0.0)
    assert compat_key.key_extractor(np.zeros(5 * SR, dtype=np.float32), SR) == ("A", "minor", 0.0)
    for n in (1, 100, 2048, 2049, 4096, 6144, 6145):
        key, scale, strength = compat_key.key_extractor(signal("c_major")[:n], SR)
        assert key in compat_key.KEY_NAMES and scale in ("major", "minor")
        assert np.isfinite(strength)


# --- part B: against essentia ---------------------------------------------------------------------

needs_essentia = pytest.mark.skipif(es is None, reason="essentia is not installed (no Windows wheels)")


def _essentia_hpcp_frames(y: np.ndarray) -> np.ndarray:
    """The HPCP stream KeyExtractor's inner network feeds its Key, from the same essentia algorithms."""
    import essentia
    import essentia.streaming as ess

    pool = essentia.Pool()
    gen = ess.VectorInput(y)
    cut = ess.FrameCutter(frameSize=4096, hopSize=4096)
    win = ess.Windowing(size=4096, type="hann")
    spec = ess.Spectrum()
    peaks = ess.SpectralPeaks(orderBy="magnitude", magnitudeThreshold=0.0001, minFrequency=25, maxFrequency=3500,
                              maxPeaks=60, sampleRate=SR)
    white = ess.SpectralWhitening(maxFrequency=3500, sampleRate=SR)
    hpcp = ess.HPCP(bandPreset=False, harmonics=4, maxFrequency=3500, minFrequency=25, nonLinear=False,
                    normalized="none", referenceFrequency=440, sampleRate=SR, size=12, weightType="cosine",
                    windowSize=1.0, maxShifted=False)
    gen.data >> cut.signal
    cut.frame >> win.frame
    win.frame >> spec.frame
    spec.spectrum >> peaks.spectrum
    spec.spectrum >> white.spectrum
    peaks.magnitudes >> white.magnitudes
    peaks.frequencies >> white.frequencies
    white.magnitudes >> hpcp.magnitudes
    peaks.frequencies >> hpcp.frequencies
    hpcp.hpcp >> (pool, "hpcp")
    essentia.run(gen)
    return np.asarray(pool["hpcp"], dtype=np.float32)


@needs_essentia
@pytest.mark.parametrize("name", list(VALIDATION_SIGNALS))
def test_matches_essentia(name):
    y = signal(name)
    ref = es.KeyExtractor(profileType="edma", sampleRate=SR)(y)
    key, scale, strength = compat_key.key_extractor(y, SR)
    assert (key, scale) == (ref[0], ref[1]), f"{name}: port {key} {scale} {strength:.5f} vs essentia {ref}"
    assert abs(strength - ref[2]) <= 1e-3, f"{name}: strength {strength:.6f} vs essentia {ref[2]:.6f}"
    assert to_camelot(f"{key} {scale}") == to_camelot(f"{ref[0]} {ref[1]}")


@needs_essentia
@pytest.mark.parametrize("name", ["c_major", "bb_major_melody", "d_major_drums", "b_minor_long_full",
                                  "white_noise", "click_track", "d_minor_short"])
def test_hpcp_stream_matches_essentia(name):
    """Stage check: the HPCP stream (everything before Key) agrees up to FFT rounding: same
    frame count, the mean within 1e-3 of its peak, and at least 85 % of the frames within 1e-3
    (measured: 94-100 %; see the module docstring for the frames that differ more)."""
    y = signal(name)
    ref = _essentia_hpcp_frames(y)
    out = compat_key.hpcp_frames(y, SR)
    assert out.shape == ref.shape
    scale = max(float(ref.max()), 1e-12)
    assert np.abs(out.mean(axis=0) - ref.mean(axis=0)).max() / scale < 1e-3
    assert np.mean(np.abs(out - ref).max(axis=1) / scale <= 1e-3) >= 0.85


@needs_essentia
def test_empty_signal_raises_in_essentia_too():
    with pytest.raises(Exception, match="empty input signal"):
        es.KeyExtractor(profileType="edma", sampleRate=SR)(np.zeros(0, dtype=np.float32))
