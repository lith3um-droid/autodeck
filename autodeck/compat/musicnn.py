"""Port of essentia's TensorflowInputMusiCNN, the mel front-end the Discogs-EffNet models were trained with.

essentia hard-codes the chain (tensorflowinputmusicnn.cpp) so it matches the training set-up:

    FrameCutter(frameSize=512, hopSize=256, startFromZero=True, validFrameThresholdRatio=1)
    > Windowing(type="hann", size=512, normalized=False; defaults: zeroPhase=True, symmetric=True)
    > Spectrum(size=512)                      magnitude of the 257-bin real FFT
    > MelBands(inputSize=257, numberBands=96, sampleRate=16000, lowFrequencyBound=0,
               highFrequencyBound=8000, warpingFormula="slaneyMel", weighting="linear",
               normalize="unit_tri"; defaults: type="power", log=False)
    > UnaryOperator(shift=1, scale=10000)     1 + 10000 * bands
    > UnaryOperator(type="log10")

The numbers below follow the C++ step by step rather than librosa, because the filterbank is not
librosa's: MelBands walks the 98 band edges by *accumulating* the mel increment in float32
(melbands.cpp calculateFilterFrequencies), which drifts ~0.01 Hz from a linspace by the top band,
and TriangularBands (triangularbands.cpp createFilters) evaluates the triangles only on the FFT
bins inside [edge_i, edge_i+2] with "unit_tri" scaling by the triangle's theoretical area. Those
float32 steps are reproduced here with numpy float32 scalars, so the filterbank, the window and
the frame positions match essentia to the bit.

What cannot be bit-exact: essentia's FFT (KissFFT/FFTW in float32) and its sequential float32
band sums; this port uses numpy's float64 FFT and a float64 matmul. The remaining difference is
essentia's own rounding noise (observed max abs error ~1e-5 on log-mel values; the tests allow 1e-3).
"""
import math

import numpy as np

SR = 16000
FRAME_SIZE = 512
HOP_SIZE = 256
N_BANDS = 96
N_BINS = FRAME_SIZE // 2 + 1  # Spectrum output: 257 bins
LOW_HZ, HIGH_HZ = 0.0, SR / 2  # MelBands lowFrequencyBound default / highFrequencyBound = sampleRate/2

_f32 = np.float32


# --- essentiamath.h Slaney mel warping, in float32 like essentia's Real -----------------------
def _hz2mel_slaney(hz: np.float32) -> np.float32:
    min_log_hz, lin_slope = _f32(1000.0), _f32(3 / 200.0)
    if hz < min_log_hz:
        return hz * lin_slope
    min_log_mel = min_log_hz * lin_slope
    log_step = _f32(math.log(6.4) / 27.0)  # computed in double, stored as Real
    return min_log_mel + np.log(hz / min_log_hz) / log_step


def _mel2hz_slaney(mel: np.float32) -> np.float32:
    min_log_hz, lin_slope = _f32(1000.0), _f32(3 / 200.0)
    min_log_mel = min_log_hz * lin_slope
    if mel < min_log_mel:
        return mel / lin_slope
    log_step = _f32(math.log(6.4) / 27.0)
    return min_log_hz * np.exp((mel - min_log_mel) * log_step)


def filter_frequencies() -> np.ndarray:
    """The 98 band edges (band i spans edges i..i+2, apex at i+1): MelBands::calculateFilterFrequencies.

    The increment is accumulated in float32, not linspace'd; that is what makes the edges essentia's.
    """
    low = _hz2mel_slaney(_f32(LOW_HZ))
    high = _hz2mel_slaney(_f32(HIGH_HZ))
    inc = (high - low) / _f32(N_BANDS + 1)
    edges = np.empty(N_BANDS + 2, np.float32)
    mel = low
    for i in range(N_BANDS + 2):
        edges[i] = _mel2hz_slaney(mel)
        mel = mel + inc
    return edges


def filterbank() -> np.ndarray:
    """(96, 257) float32 weights: TriangularBands::createFilters with weighting="linear", normalize="unit_tri".

    Each triangle is sampled only at the FFT bins inside [edge_i, edge_i+2] (bin spacing 31.25 Hz,
    so the lowest bands touch just two or three bins) and divided by its theoretical area
    (edge_i+2 - edge_i) / 2, the "unit_tri" rule. All arithmetic in float32 as in essentia.
    """
    edges = filter_frequencies()
    scale = _f32((SR / 2.0) / (N_BINS - 1))  # frequencyScale = 31.25 Hz per bin, exact
    bank = np.zeros((N_BANDS, N_BINS), np.float32)
    for i in range(N_BANDS):
        f0, f1, f2 = edges[i], edges[i + 1], edges[i + 2]
        step1, step2 = f1 - f0, f2 - f1
        j0, j1 = math.ceil(f0 / scale), math.floor(f2 / scale)
        if j1 >= N_BINS:
            raise ValueError("band edge above Nyquist")  # essentia throws the same way
        for j in range(j0, j1 + 1):
            binfreq = _f32(j) * scale
            bank[i, j] = (binfreq - f0) / step1 if binfreq < f1 else (f2 - binfreq) / step2
        area = _f32(float(step1 + step2) / 2.0)
        bank[i, j0 : j1 + 1] /= area
    return bank


def hann_window() -> np.ndarray:
    """Windowing type="hann", symmetric=True, normalized=False: 0.5 - 0.5 cos(2 pi i / (N - 1)), double then Real."""
    i = np.arange(FRAME_SIZE, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * i / (FRAME_SIZE - 1.0))).astype(np.float32)


_WINDOW = hann_window()
_BANK_T = filterbank().T.astype(np.float64)  # (257, 96) for the matmul


def frame_count(n_samples: int) -> int:
    """How many frames FrameCutter(startFromZero=True, validFrameThresholdRatio=1) yields.

    Only full 512-sample frames at 0, 256, 512, ... are kept: a frame that would run past the end
    is dropped (threshold = frameSize), and one that ends exactly at the end is the last.
    """
    return 0 if n_samples < FRAME_SIZE else (n_samples - FRAME_SIZE) // HOP_SIZE + 1


def mel_frames(y: np.ndarray) -> np.ndarray:
    """Log-mel frames of a 16 kHz mono signal, float32 (n_frames, 96), as essentia's

        [TensorflowInputMusiCNN()(f) for f in FrameGenerator(y, frameSize=512, hopSize=256,
                                                              startFromZero=True, validFrameThresholdRatio=1)]

    Fewer than 512 samples give shape (0, 96). Digital silence gives exactly 0 (log10(1 + 0)).
    """
    y = np.ascontiguousarray(y, dtype=np.float32).ravel()
    n = frame_count(len(y))
    if n == 0:
        return np.zeros((0, N_BANDS), np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(y, FRAME_SIZE)[: (n - 1) * HOP_SIZE + 1 : HOP_SIZE]
    windowed = frames * _WINDOW  # float32 * float32, same rounding as Windowing::compute
    # zeroPhase=True swaps the two halves of the windowed frame; the magnitude spectrum is unchanged
    # by that circular shift, so it is left out.
    power = np.abs(np.fft.rfft(windowed.astype(np.float64), axis=1)) ** 2  # Spectrum then MelBands type="power"
    bands = (power @ _BANK_T).astype(np.float32)
    shifted = _f32(1) + _f32(10000) * bands  # UnaryOperator shift/scale, float32 like essentia
    return np.log10(shifted)  # >= 1 always, so UnaryOperator's 1e-30 clip never applies
