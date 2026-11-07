"""numpy port of essentia's KeyExtractor, for Windows where essentia has no wheels.

analyze._key calls `KeyExtractor(profileType="edma", sampleRate=22050)(y)`; this module is a
stage-by-stage port of that streaming network with the exact parameters keyextractor.cpp
configures, not a chroma approximation:

    FrameCutter        frameSize 4096, hopSize 4096, startFromZero false (first frame centred on
                       sample 0, zero padded), validFrameThresholdRatio 0, silentFrames "noise"
                       (-100 dB power, 1e-10: can never lift a peak over the threshold, so skipped)
    Windowing          hann, size 4096, symmetric, normalized (area 2), zeroPhase
    Spectrum           |rfft|
    SpectralPeaks      PeakDetection: parabolic interpolation, minFrequency 25, maxFrequency 3500,
                       magnitudeThreshold 1e-4, maxPeaks 60, orderBy magnitude
    SpectralWhitening  maxFrequency 3500 (x1.2 inside), 100 Hz break-point envelope
    HPCP               size 12, referenceFrequency 440, harmonics 4, weightType cosine,
                       windowSize 1.0, normalized none, nonLinear/bandPreset/maxShifted false
    Key (streaming)    mean of all HPCP frames, normalize, pcpThreshold 0.2 gate,
                       averageDetuningCorrection (a no-op at pcpSize 12), profile "edma",
                       usePolyphony false, useThreeChords false, numHarmonics 4, slope 0.6

Not bit-exact: essentia runs KISS FFT in float32, this uses numpy's float64 rfft rounded to
float32, so spectra differ by ~1e-7 relative and the strength by ~1e-6 (2e-5 at worst over the
validation set); a peak sitting exactly on the 1e-4 threshold or an HPCP bin on the 0.2 gate
could in theory flip. The one input class that cannot match at all is a spectrum that is flat or
stationary over a run of bins (a lone impulse, a bare Nyquist tone): essentia's peaks there are
its FFT's rounding noise, the port sees a plateau and no peak (silence and such signals give
"A minor" with strength 0.0, which is also what essentia returns for silence). Verified against
essentia 2.1b6 in tests/test_compat_key.py (same key/scale, strength within 1e-3).
"""
import math

import numpy as np

FRAME_SIZE = 4096
HOP_SIZE = 4096
MIN_FREQUENCY = 25.0
MAX_FREQUENCY = 3500.0
PEAK_THRESHOLD = 0.0001
MAX_PEAKS = 60
HPCP_SIZE = 12
TUNING_FREQUENCY = 440.0
PCP_THRESHOLD = 0.2
HPCP_HARMONICS = 4
WHITENING_RESOLUTION = 100.0   # SpectralWhitening::bpfResolution
SILENCE_CUTOFF = 1e-10         # essentiamath lin2db: below this -> -100 dB

# key.cpp: bin 0 of the HPCP is the reference frequency (A4 = 440 Hz), hence this order and spelling.
KEY_NAMES = ["A", "Bb", "B", "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab"]

# key.cpp profile tables (major, minor); usePolyphony=false so they are used as they are.
# "weichai" needs a second mode-detection pass and is not ported.
PROFILES = {
    "diatonic": ([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1]),
    "krumhansl": ([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
                  [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
    "temperley": ([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
                  [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]),
    "tonictriad": ([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
    "temperley2005": ([0.748, 0.060, 0.488, 0.082, 0.67, 0.46, 0.096, 0.715, 0.104, 0.366, 0.057, 0.4],
                      [0.712, 0.084, 0.474, 0.618, 0.049, 0.46, 0.105, 0.747, 0.404, 0.067, 0.133, 0.33]),
    "thpcp": ([0.95162, 0.20742, 0.71758, 0.22007, 0.71341, 0.48841, 0.31431, 1.00000, 0.20957, 0.53657, 0.22585, 0.55363],
              [0.94409, 0.21742, 0.64525, 0.63229, 0.27897, 0.57709, 0.26428, 1.0000, 0.26428, 0.30633, 0.45924, 0.35929]),
    "shaath": ([6.6, 2.0, 3.5, 2.3, 4.6, 4.0, 2.5, 5.2, 2.4, 3.7, 2.3, 3.4],
               [6.5, 2.7, 3.5, 5.4, 2.6, 3.5, 2.5, 5.2, 4.0, 2.7, 4.3, 3.2]),
    "gomez": ([0.82, 0.00, 0.55, 0.00, 0.53, 0.30, 0.08, 1.00, 0.00, 0.38, 0.00, 0.47],
              [0.81, 0.00, 0.53, 0.54, 0.00, 0.27, 0.07, 1.00, 0.27, 0.07, 0.10, 0.36]),
    "noland": ([0.0629, 0.0146, 0.061, 0.0121, 0.0623, 0.0414, 0.0248, 0.0631, 0.015, 0.0521, 0.0142, 0.0478],
               [0.0682, 0.0138, 0.0543, 0.0519, 0.0234, 0.0544, 0.0176, 0.067, 0.0349, 0.0297, 0.0401, 0.027]),
    "edmm": ([0.083] * 12,
             [0.17235348, 0.04, 0.0761009, 0.12, 0.05621498, 0.08527853, 0.0497915, 0.13451001, 0.07458916, 0.05003023, 0.09187879, 0.05545106]),
    "bgate": ([1.00, 0.00, 0.42, 0.00, 0.53, 0.37, 0.00, 0.77, 0.00, 0.38, 0.21, 0.30],
              [1.00, 0.00, 0.36, 0.39, 0.00, 0.38, 0.00, 0.74, 0.27, 0.00, 0.42, 0.23]),
    "braw": ([1.0000, 0.1573, 0.4200, 0.1570, 0.5296, 0.3669, 0.1632, 0.7711, 0.1676, 0.3827, 0.2113, 0.2965],
             [1.0000, 0.2330, 0.3615, 0.3905, 0.2925, 0.3777, 0.1961, 0.7425, 0.2701, 0.2161, 0.4228, 0.2272]),
    "edma": ([1.00, 0.29, 0.50, 0.40, 0.60, 0.56, 0.32, 0.80, 0.31, 0.45, 0.42, 0.39],
             [1.00, 0.31, 0.44, 0.58, 0.33, 0.49, 0.29, 0.78, 0.43, 0.29, 0.53, 0.32]),
}


def key_extractor(y: np.ndarray, sample_rate: float, profile_type: str = "edma") -> tuple[str, str, float]:
    """essentia KeyExtractor(profileType=profile_type, sampleRate=sample_rate)(y).

    Returns (key, scale, strength) with essentia's spellings ("Eb", "F#", ...; "major"/"minor")
    and the raw correlation strength (float32-rounded). Raises ValueError on an empty signal,
    as essentia throws "KeyExtractor: empty input signal" (analyze._key catches it).
    """
    y = np.asarray(y, dtype=np.float32).ravel()
    if y.size == 0:
        raise ValueError("KeyExtractor: empty input signal")
    if profile_type not in PROFILES:
        raise ValueError(f"Key: Unsupported profile type: {profile_type}")
    hpcp = hpcp_frames(y, float(sample_rate))
    return key_from_hpcp(hpcp.mean(axis=0, dtype=np.float64).astype(np.float32), profile_type)


# --- stages: each takes/returns what the essentia algorithm of the same name does ---------------

def frames(y: np.ndarray, frame_size: int = FRAME_SIZE, hop_size: int = HOP_SIZE) -> np.ndarray:
    """FrameCutter(startFromZero=false): frame k starts at -(frameSize+1)//2 + k*hopSize,
    zero padded at both ends; frames are produced while their start is inside the signal
    (the streaming cutter keeps producing until no sample is left, validFrameThresholdRatio 0).
    Shape (n_frames, frame_size), float32."""
    y = np.asarray(y, dtype=np.float32)
    start0 = -((frame_size + 1) // 2)
    n_frames = -(-(len(y) - start0) // hop_size)           # ceil
    padded = np.zeros(max((n_frames - 1) * hop_size + frame_size, len(y) - start0), dtype=np.float32)
    padded[-start0:-start0 + len(y)] = y
    idx = np.arange(n_frames)[:, None] * hop_size + np.arange(frame_size)[None, :]
    return padded[idx]


def hann_window(size: int = FRAME_SIZE) -> np.ndarray:
    """Windowing(type=hann, normalized=true): symmetric hann scaled so its float32 sum is 2
    (the sum is accumulated in float32 in sequence, as windowing.cpp does)."""
    i = np.arange(size, dtype=np.float64)
    w = (0.5 - 0.5 * np.cos(2.0 * np.pi * i / (size - 1.0))).astype(np.float32)
    total = np.add.accumulate(np.abs(w), dtype=np.float32)[-1]
    return (w * np.float32(2.0 / float(total))).astype(np.float32)


def windowed(fr: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Windowing(zeroPhase=true): multiply, then swap the halves (the FFT magnitude does not
    change, it is kept so the rounding path matches)."""
    out = (fr.astype(np.float32) * window.astype(np.float32)).astype(np.float32)
    half = out.shape[-1] // 2
    return np.concatenate([out[..., half:], out[..., :half]], axis=-1)


def spectrum(wf: np.ndarray) -> np.ndarray:
    """Spectrum: magnitude of the one-sided FFT. float64 rfft rounded to float32 (essentia: KISS in
    float32; this is the one stage that cannot match bit for bit)."""
    return np.abs(np.fft.rfft(wf.astype(np.float64), axis=-1)).astype(np.float32)


def spectral_peaks(spec: np.ndarray, sample_rate: float, min_freq: float = MIN_FREQUENCY,
                   max_freq: float = MAX_FREQUENCY, threshold: float = PEAK_THRESHOLD,
                   max_peaks: int = MAX_PEAKS) -> tuple[np.ndarray, np.ndarray]:
    """SpectralPeaks on one spectrum: PeakDetection(interpolate=true, orderBy=amplitude).

    Returns (frequencies, magnitudes), at most max_peaks, sorted by magnitude descending (ties:
    lower frequency first). Follows peakdetection.cpp: the scan starts at ceil(minFreq/scale),
    a peak there is taken un-interpolated, interior strict maxima get parabolic interpolation
    of position and value, and the scan stops at the first peak past maxFreq. Plateaus (equal
    neighbouring bins over the threshold) are handed to a literal port of the C++ loop."""
    a = np.asarray(spec, dtype=np.float32)
    size = a.shape[-1]
    scale = np.float32(np.float32(sample_rate / 2.0) / np.float32(size - 1))
    i0 = max(0, int(math.ceil(np.float32(min_freq) / scale)))
    thr = np.float32(threshold)
    over = a > thr
    if i0 + 1 < size and np.any(over[i0:-1] & (a[i0:-1] == a[i0 + 1:])):
        pos, val = _peak_scan_literal(a, scale, i0, max_freq, thr)
    else:
        pos, val = [], []
        if i0 + 1 < size and a[i0] > a[i0 + 1] and over[i0]:
            pos.append(np.float32(i0) * scale)
            val.append(a[i0])
        m = np.flatnonzero(over[i0 + 1:size - 1] & (a[i0:size - 2] < a[i0 + 1:size - 1])
                           & (a[i0 + 2:size] < a[i0 + 1:size - 1])) + (i0 + 1)
        if len(m):
            left, mid, right = a[m - 1].astype(np.float64), a[m].astype(np.float64), a[m + 1].astype(np.float64)
            delta = 0.5 * ((left - right) / (left - 2.0 * mid + right))
            p = ((m + delta).astype(np.float32) * scale).astype(np.float32)
            v = (mid - 0.25 * (left - right) * delta).astype(np.float32)
            # the loop breaks at the first interior peak beyond maxFreq; a peak at the second to last
            # bin is only reached (and then kept regardless of maxFreq) when nothing broke before
            interior = m <= size - 3
            beyond = np.flatnonzero(interior & (p > np.float32(max_freq)))
            keep = np.ones(len(m), dtype=bool)
            if len(beyond):
                keep[beyond[0]:] = False
            pos.extend(p[keep])
            val.extend(v[keep])
    pos, val = np.asarray(pos, dtype=np.float32), np.asarray(val, dtype=np.float32)
    order = np.lexsort((pos, -val.astype(np.float64)))[:max_peaks]
    return pos[order], val[order]


def _peak_scan_literal(a, scale, i0, max_freq, thr):
    """peakdetection.cpp's scan loop as written, for spectra with plateaus."""
    size = len(a)
    pos, val = [], []
    i = i0
    if i + 1 < size and a[i] > a[i + 1] and a[i] > thr:
        pos.append(np.float32(i) * scale)
        val.append(a[i])
    while True:
        while i + 1 < size - 1 and a[i] >= a[i + 1]:
            i += 1
        while i + 1 < size - 1 and a[i] < a[i + 1]:
            i += 1
        j = i
        while j + 1 < size - 1 and a[j] == a[j + 1]:
            j += 1
        if j + 1 < size - 1 and a[j + 1] < a[j] and a[j] > thr:
            if j != i:
                rb, rv = np.float32((i + j) * 0.5), a[i]
            else:
                rv, rb = _interpolate(a[j - 1], a[j], a[j + 1], j)
            p = np.float32(rb * scale)
            if p > np.float32(max_freq):
                break
            pos.append(p)
            val.append(rv)
        i = j
        if i + 1 >= size - 1:
            if i == size - 2 and a[i - 1] < a[i] and a[i + 1] < a[i] and a[i] > thr:
                rv, rb = _interpolate(a[i - 1], a[i], a[i + 1], j)
                pos.append(np.float32(rb * scale))
                val.append(rv)
            break
    return pos, val


def _interpolate(left, mid, right, cur):
    left, mid, right = float(left), float(mid), float(right)
    delta = 0.5 * ((left - right) / (left - 2.0 * mid + right))
    return np.float32(mid - 0.25 * (left - right) * delta), np.float32(cur + delta)


def _lin2db(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x < SILENCE_CUTOFF, -100.0, 10.0 * np.log10(np.maximum(x, SILENCE_CUTOFF)))


class Whitening:
    """SpectralWhitening(maxFrequency, sampleRate): the per-frame envelope needs the same bin
    windows every frame, so they are built once (spectralwhitening.cpp, the loop over `freq`)."""

    def __init__(self, sample_rate: float, spec_size: int, max_freq: float = MAX_FREQUENCY):
        self.max_freq = float(np.float32(float(np.float32(max_freq)) * 1.2))   # the 1.2 "magic number"
        self.spectral_range = np.float32(sample_rate / 2.0)
        xs = []
        windows = []
        freq = 0.0
        while freq <= self.max_freq and freq <= self.spectral_range:
            bf = np.float32(freq - max(50.0, freq * 0.34))
            ef = np.float32(freq + max(50.0, freq * 0.58))
            b = int(float(np.float32(bf / self.spectral_range)) * (spec_size - 1.0) + 0.5)
            e = int(float(np.float32(ef / self.spectral_range)) * (spec_size - 1.0) + 0.5)
            b = min(spec_size - 1, max(b, 0))
            e = min(spec_size, max(e, b + 1))
            c = b / 2.0 + e / 2.0
            half = e - c
            w = np.zeros(spec_size)
            i = np.arange(b, e)
            w[b:e] = (1.0 - np.abs(i - c) / half) ** 4
            windows.append(w)
            xs.append(freq)
            freq = float(np.float32(freq + WHITENING_RESOLUTION))
        self.xs = np.asarray(xs)
        self.windows = np.asarray(windows)                 # (n_points, spec_size)

    def envelope(self, spec: np.ndarray) -> np.ndarray:
        """Break-point values in dB (amplitude dB, i.e. 2*lin2db), one row per spectrum: the
        energy-weighted mean energy in each window (weight = window^4 * energy)."""
        energy = np.atleast_2d(np.asarray(spec, dtype=np.float64)) ** 2
        n = energy @ self.windows.T
        wavg = (energy * energy) @ self.windows.T
        wavg = np.where(n != 0.0, wavg / np.where(n != 0.0, n, 1.0), wavg)
        wavg[:, -1] = wavg[:, -2]
        return 2.0 * _lin2db(np.sqrt(wavg))

    def __call__(self, ys: np.ndarray, freqs: np.ndarray, mags: np.ndarray) -> np.ndarray:
        """Whitened linear magnitudes (float32) of one frame's peaks given its envelope row."""
        if len(mags) == 0:
            return np.zeros(0, dtype=np.float32)
        freqs = np.asarray(freqs, dtype=np.float32).astype(np.float64)
        amp = 2.0 * _lin2db(np.asarray(mags, dtype=np.float32))
        env = np.interp(freqs, self.xs, ys)
        white = np.where(amp > env, 0.0, np.where(amp > env - 30.0, amp - env, -200.0))
        white = white - 20.0 * freqs / 4000.0
        white = np.where(freqs > self.max_freq - WHITENING_RESOLUTION, amp, white)
        return (10.0 ** (white / 20.0)).astype(np.float32)


def _harmonic_table(n_harmonics: int = HPCP_HARMONICS) -> list[tuple[float, float]]:
    """HPCP::initHarmonicContributionTable: (semitone offset, summed weight) per distinct pitch
    class of harmonics 1..n+1; with 4 harmonics: [(0, 3), (7.02, 1), (3.86, 0.861)]."""
    table: list[list[float]] = []
    for i in range(n_harmonics + 1):
        semitone = 12.0 * math.log2(i + 1.0)
        octweight = max(1.0, (semitone / 12.0) * 0.5)
        while semitone >= 12.0 - 1e-5:
            semitone -= 12.0
        for entry in table:
            if semitone - 1e-5 < entry[0] < semitone + 1e-5:
                entry[1] += 1.0 / octweight
                break
        else:
            table.append([semitone, 1.0 / octweight])
    return [(s, w) for s, w in table]


def hpcp(freqs: np.ndarray, mags: np.ndarray, size: int = HPCP_SIZE, reference: float = TUNING_FREQUENCY,
         min_freq: float = MIN_FREQUENCY, max_freq: float = MAX_FREQUENCY,
         harmonics: int = HPCP_HARMONICS, window_size: float = 1.0) -> np.ndarray:
    """HPCP(weightType=cosine, normalized=none, nonLinear=false, bandPreset=false, maxShifted=false)
    of one frame's peaks: every peak and each hypothesised sub-harmonic fundamental adds
    cos(pi*distance) * mag^2 * weight^2 to the bins within +-windowSize/2 of its pitch class."""
    out = np.zeros(size, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float32).astype(np.float64)
    mags = np.asarray(mags, dtype=np.float32).astype(np.float64)
    keep = (freqs >= np.float32(min_freq)) & (freqs <= np.float32(max_freq))
    freqs, mags = freqs[keep], mags[keep]
    if len(freqs) == 0:
        return out.astype(np.float32)
    resolution = size // 12
    for semitone, strength in _harmonic_table(harmonics):
        f = freqs * (2.0 ** (-semitone / 12.0))
        bin_f = np.log2(f / reference) * size
        left = np.ceil(bin_f - resolution * window_size / 2.0).astype(int)
        right = np.floor(bin_f + resolution * window_size / 2.0).astype(int)
        contrib = mags * mags * strength * strength
        # with windowSize 1.0 the window covers one bin, two only when bin_f is exactly x.5
        for bins in (left, right):
            sel = np.ones(len(bins), dtype=bool) if bins is left else right != left
            weight = np.cos(np.pi * (np.abs(bin_f[sel] - bins[sel]) / resolution) / window_size)
            np.add.at(out, bins[sel] % size, weight * contrib[sel])
    return out.astype(np.float32)


def hpcp_frames(y: np.ndarray, sample_rate: float) -> np.ndarray:
    """The HPCP of every frame of the signal, shape (n_frames, 12) float32: the stream the
    streaming Key accumulates."""
    fr = frames(y)
    spec = spectrum(windowed(fr, hann_window(fr.shape[1])))
    whiten = Whitening(sample_rate, spec.shape[1])
    envelopes = whiten.envelope(spec)
    out = np.zeros((len(spec), HPCP_SIZE), dtype=np.float32)
    for k in range(len(spec)):
        f, m = spectral_peaks(spec[k], sample_rate)
        out[k] = hpcp(f, whiten(envelopes[k], f, m))
    return out


def key_from_hpcp(pcp: np.ndarray, profile_type: str = "edma", pcp_threshold: float = PCP_THRESHOLD,
                  detuning_correction: bool = True) -> tuple[str, str, float]:
    """streaming Key::process on the averaged HPCP, then standard Key::compute.

    pcpThreshold > 0: normalize to unit max and zero the bins below the threshold.
    averageDetuningCorrection: rotate so the max lands on a tempered bin; with 12 bins per octave
    every bin is tempered, so it only renormalizes. Then the Pearson correlation of the pcp with
    each rotation of the major and minor profile; the best wins, minor on a tie, strength is
    that correlation (0 when either side has no variance, e.g. silence -> "A minor", 0.0)."""
    pcp = np.asarray(pcp, dtype=np.float32).astype(np.float64)
    size = len(pcp)
    if size < 12 or size % 12:
        raise ValueError("Key: input PCP size is not a positive multiple of 12")
    if pcp_threshold > 0:
        pcp = _normalize(pcp)
        pcp[pcp < pcp_threshold] = 0.0
    if detuning_correction:
        pcp = _normalize(pcp)
        res = size // 12
        shift = int(np.argmax(pcp)) % res
        pcp = np.roll(pcp, -(shift - res if shift > res // 2 else shift))
    major, minor = (np.asarray(p, dtype=np.float64) for p in PROFILES[profile_type])
    best = []
    for prof in (major, minor):
        prof = _resize_profile(prof, size)
        corr = [_correlation(pcp, prof, s) for s in range(size)]
        k = int(np.argmax(corr))                     # first max wins, as the strict > in key.cpp
        best.append((corr[k], k))
    (c_major, k_major), (c_minor, k_minor) = best
    if c_major > c_minor:
        strength, idx, scale = c_major, k_major, "major"
    else:
        strength, idx, scale = c_minor, k_minor, "minor"
    return KEY_NAMES[idx * 12 // size], scale, float(np.float32(strength))  # C int division


def _normalize(v):
    m = v.max()
    return v / m if m != 0 else v


def _resize_profile(prof, size):
    """Key::resize: linear interpolation between the 12 profile values when size > 12."""
    n = size // 12
    out = np.zeros(size)
    for i in range(12):
        nxt = prof[0] if i == 11 else prof[i + 1]
        incr = (prof[i] - nxt) / n
        for j in range(n):
            out[i * n + j] = prof[i] - j * incr
    return out


def _correlation(v1, v2, shift):
    """Key::correlation: Pearson correlation of v1 with v2 rotated by `shift`, 0 if either is flat."""
    d1 = v1 - v1.mean()
    d2 = v2 - v2.mean()
    s1, s2 = math.sqrt(float(d1 @ d1)), math.sqrt(float(d2 @ d2))
    if s1 == 0.0 or s2 == 0.0:
        return 0.0
    return float(d1 @ np.roll(d2, shift)) / (s1 * s2)
