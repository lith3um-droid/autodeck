"""autodeck.compat.musicnn vs essentia's TensorflowInputMusiCNN.

Part A runs everywhere (shape, frame count, range, silence). Part B needs real essentia (macOS /
Linux wheels) and checks the port frame by frame on synthetic music-like signals; on Windows,
where there is no essentia, only part A runs.
"""
import numpy as np
import pytest

from autodeck.compat import musicnn as m

try:
    import essentia.standard as es
except ImportError:  # Windows
    es = None

needs_essentia = pytest.mark.skipif(es is None, reason="essentia has no Windows wheels; the port is validated on macOS/Linux")

SR = m.SR  # 16000


# --- signals -------------------------------------------------------------------------------
def chord_track(seconds, seed=0):
    """Chord progression (5 harmonics per note), bass an octave below, noise-burst percussion."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    y = np.zeros_like(t)
    roots = [110.0, 146.83, 196.0, 174.61]
    for k, root in enumerate(roots):
        on = ((t // 2) % len(roots)) == k  # one chord per 2 s
        for ratio in (1, 1.25, 1.5, 2):
            for h in range(1, 6):
                y += on * np.sin(2 * np.pi * root * ratio * h * t + rng.uniform(0, 2 * np.pi)) * 0.15 / h
        y += on * 0.3 * np.sin(2 * np.pi * root / 2 * t)
    for s in range(0, len(t), SR // 2):  # 120 bpm hits
        n = min(2000, len(t) - s)
        y[s : s + n] += rng.standard_normal(n) * 0.4 * np.exp(-np.arange(n) / 300)
    return np.clip(y, -1, 1).astype(np.float32)


def click_track_16k(seconds=30):
    import librosa
    from conftest import SR as CLICK_SR, make_click_signal

    return librosa.resample(make_click_signal(120, seconds), orig_sr=CLICK_SR, target_sr=SR).astype(np.float32)


def pink_noise(seconds, seed=3):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    spec = np.fft.rfft(rng.standard_normal(n)) / np.sqrt(np.arange(n // 2 + 1) + 1.0)
    y = np.fft.irfft(spec, n)
    return (0.3 * y / np.abs(y).max()).astype(np.float32)


def sine_sweep(seconds=10, f0=20.0, f1=8000.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.8 * np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * seconds) * t**2))).astype(np.float32)


VALIDATION = ["chord_progression", "click_track", "white_noise", "pink_noise", "sine_sweep", "silence_both_ends", "short_1.5s", "long_65s"]


def validation_signal(name):
    z = np.zeros(2 * SR, np.float32)
    return {
        "chord_progression": lambda: chord_track(20),
        "click_track": lambda: click_track_16k(30),
        "white_noise": lambda: (0.3 * np.random.default_rng(1).standard_normal(10 * SR)).astype(np.float32),
        "pink_noise": lambda: pink_noise(10),
        "sine_sweep": lambda: sine_sweep(),
        "silence_both_ends": lambda: np.concatenate([z, chord_track(5, 2), z]),
        "short_1.5s": lambda: chord_track(1.5, 4),
        "long_65s": lambda: chord_track(65, 7),
    }[name]()


# --- part A: always ------------------------------------------------------------------------
@pytest.mark.parametrize("n,expected", [(0, 0), (100, 0), (511, 0), (512, 1), (513, 1), (767, 1), (768, 2), (800, 2), (1024, 3), (2 * SR, (2 * SR - 512) // 256 + 1)])
def test_frame_count(n, expected):
    assert m.frame_count(n) == expected
    y = np.random.default_rng(n).standard_normal(n).astype(np.float32)
    assert m.mel_frames(y).shape == (expected, 96)


def test_shape_dtype_finite():
    out = m.mel_frames(chord_track(3))
    assert out.shape == (m.frame_count(3 * SR), 96)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_silence_is_zero():
    assert np.array_equal(m.mel_frames(np.zeros(3 * SR, np.float32)), np.zeros((m.frame_count(3 * SR), 96), np.float32))


def test_plausible_range():
    # log10(1 + 10000 * power): non-negative, and a full-scale signal stays well under 10
    for y in (chord_track(3), sine_sweep(3), np.ones(SR, np.float32)):
        out = m.mel_frames(y)
        assert out.min() >= 0
        assert out.max() < 10
    assert m.mel_frames(chord_track(3)).max() > 1  # music is clearly above the floor


def test_sine_lands_in_its_band():
    t = np.arange(2 * SR) / SR
    edges = m.filter_frequencies()
    for hz in (440.0, 1500.0, 5000.0):
        prof = m.mel_frames((0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)).mean(axis=0)
        expected = np.argmin(np.abs(edges[1:-1] - hz))  # band i has its apex at edge i+1
        assert abs(int(np.argmax(prof)) - expected) <= 1


def test_filter_edges_are_slaney_mel():
    librosa = pytest.importorskip("librosa")
    # essentia accumulates the mel step in float32; librosa linspaces in float64. ~0.01 Hz apart at most.
    assert np.allclose(m.filter_frequencies(), librosa.mel_frequencies(98, fmin=0.0, fmax=8000.0, htk=False), atol=0.05)


def test_filterbank_shape_and_support():
    bank = m.filterbank()
    assert bank.shape == (96, 257) and bank.dtype == np.float32
    assert (bank >= 0).all()
    assert ((bank > 0).sum(axis=1) >= 1).all()  # every band touches at least one bin


def test_accepts_float64_and_2d_input():
    y = chord_track(2).astype(np.float64)
    assert np.array_equal(m.mel_frames(y), m.mel_frames(y.astype(np.float32)))
    assert m.mel_frames(y.reshape(1, -1)).shape == m.mel_frames(y).shape


# --- part B: against real essentia ---------------------------------------------------------
def essentia_frames(y):
    mel = es.TensorflowInputMusiCNN()
    out = np.array([mel(f) for f in es.FrameGenerator(y, frameSize=512, hopSize=256, startFromZero=True, validFrameThresholdRatio=1)], dtype=np.float32)
    return out.reshape(-1, 96)


@needs_essentia
@pytest.mark.parametrize("n", [0, 100, 511, 512, 513, 767, 768, 769, 1024, 1025, 3 * SR + 17])
def test_frame_count_matches_framegenerator(n):
    y = np.random.default_rng(n).standard_normal(n).astype(np.float32)
    n_es = sum(1 for _ in es.FrameGenerator(y, frameSize=512, hopSize=256, startFromZero=True, validFrameThresholdRatio=1))
    assert m.frame_count(n) == n_es


@needs_essentia
def test_window_matches_essentia():
    frame = np.random.default_rng(1).standard_normal(512).astype(np.float32)
    ours = frame * m.hann_window()
    ours = np.concatenate([ours[256:], ours[:256]])  # zeroPhase=True swaps the halves
    assert np.array_equal(es.Windowing(type="hann", normalized=False, size=512)(frame), ours)


@needs_essentia
def test_filterbank_matches_essentia_melbands():
    mb = es.MelBands(inputSize=257, numberBands=96, sampleRate=16000, highFrequencyBound=8000, warpingFormula="slaneyMel", weighting="linear", normalize="unit_tri")
    ref = np.zeros((96, 257), np.float32)
    for j in range(257):  # impulse responses: column j of essentia's filterbank
        e = np.zeros(257, np.float32)
        e[j] = 1.0
        ref[:, j] = mb(e)
    assert np.abs(ref - m.filterbank()).max() <= 1e-6


@needs_essentia
@pytest.mark.parametrize("name", VALIDATION)
def test_matches_essentia(name):
    y = validation_signal(name)
    ref, out = essentia_frames(y), m.mel_frames(y)
    assert out.shape == ref.shape, (name, out.shape, ref.shape)
    err = np.abs(ref - out).max() if len(ref) else 0.0
    print(f"{name}: {len(y)} samples, {len(ref)} frames, max abs err {err:.2e}")
    assert err <= 1e-3, (name, err)
