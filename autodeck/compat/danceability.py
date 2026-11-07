"""Port of essentia's Danceability (detrended fluctuation analysis, Streich & Herrera 2005).

Stands in for `essentia.standard.Danceability(sampleRate=44100)(y)` in analyze.danceability where
essentia has no wheels (Windows). Ported from src/algorithms/rhythm/danceability.cpp + .h with the
parameters autodeck uses: minTau 310 ms, maxTau 8800 ms, tauMultiplier 1.1.

The C++ works in float32 (essentia's Real) with plain in-order loops, and its residual-error step
subtracts two large float32 sums, so a float64 closed form would not give essentia's numbers, only
close ones. The port therefore keeps essentia's dtype and summation order: np.cumsum accumulates
in order (np.sum does not, it is pairwise), so the last column of a float32 cumsum is the C++ sum.
Only log10 can differ, by an ulp of float32, when numpy's log10f rounds differently from glibc's.
"""

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

F32 = np.float32

MIN_TAU_MS = 310.0
MAX_TAU_MS = 8800.0
TAU_MULTIPLIER = 1.1


def tau_frames(min_tau_ms: float = MIN_TAU_MS, max_tau_ms: float = MAX_TAU_MS,
               multiplier: float = TAU_MULTIPLIER) -> list[int]:
    """Block sizes in 10 ms frames as Danceability::configure builds them: a float32 geometric
    series from minTau up to and including maxTau, each value truncated with int(tau / 10)."""
    if min_tau_ms > max_tau_ms:
        raise ValueError("Danceability: minTau cannot be larger than maxTau")
    if not (min_tau_ms > 0 and multiplier > 1):  # essentia's parameter ranges (0,inf) and (1,inf)
        raise ValueError("Danceability: minTau must be > 0 and tauMultiplier > 1")
    taus = []
    tau, top, mult = F32(min_tau_ms), F32(max_tau_ms), F32(multiplier)
    while tau <= top:
        taus.append(int(float(tau) / 10.0))
        tau = F32(tau * mult)
    return taus


def _frame_stddev(frames: np.ndarray) -> np.ndarray:
    """Danceability::stddev of every row: float32 mean and sum of squared deviations accumulated
    in order, then sqrt(var / (n - 1)) in double (the C++ divides by a double) back to float32."""
    n = frames.shape[1]
    mean = np.cumsum(frames, axis=1, dtype=F32)[:, -1] / F32(n)
    d = frames - mean[:, None]
    var = np.cumsum(d * d, axis=1, dtype=F32)[:, -1]
    return np.sqrt(var.astype(np.float64) / (n - 1.0)).astype(F32)


def _fluctuation(s: np.ndarray, tau: int, rows_per_chunk: int = 1 << 21) -> np.float32:
    """F(tau) of Danceability::compute: residualError over the tau-frame blocks of s starting at
    0, jump, 2*jump, ... below len(s) - tau (jump = max(tau / 50, 1)), summed in order, divided by
    (len(s) - tau) / jump and square-rooted, all in float32. Each residual is
    (ssyy - ssxy^2 / ssxx) / tau with the three sums accumulated in order (the C++ unrolls by 8
    but keeps the order). Blocks are independent, so they are processed in bounded chunks."""
    n = len(s)
    if n == tau:
        return F32(0.0)
    if tau == 0:  # minTau below 10 ms: the C++ divides an empty block's sums 0/0
        return F32("nan")
    jump = max(tau // 50, 1)
    blocks = sliding_window_view(s, tau)[: n - tau : jump]  # row k: s[k*jump : k*jump + tau]
    dx = np.arange(tau, dtype=F32) - F32((tau - 1.0) * 0.5)
    ssxx = np.cumsum(dx * dx, dtype=F32)[-1]
    residuals = np.empty(len(blocks), dtype=F32)
    step = max(1, rows_per_chunk // tau)
    with np.errstate(divide="ignore", invalid="ignore"):  # tau == 1: ssxx == 0, nan as in the C++
        for i in range(0, len(blocks), step):
            b = blocks[i : i + step]
            mean_y = np.cumsum(b, axis=1, dtype=F32)[:, -1] / F32(tau)
            dy = b - mean_y[:, None]
            ssxy = np.cumsum(dx * dy, axis=1, dtype=F32)[:, -1]
            ssyy = np.cumsum(dy * dy, axis=1, dtype=F32)[:, -1]
            residuals[i : i + step] = (ssyy - ssxy * ssxy / ssxx) / F32(tau)
        total = np.cumsum(residuals, dtype=F32)[-1]
        return np.sqrt(total / (F32(n - tau) / F32(jump)))


def danceability(y: np.ndarray, sample_rate: float = 44100.0, *, min_tau_ms: float = MIN_TAU_MS,
                 max_tau_ms: float = MAX_TAU_MS, tau_multiplier: float = TAU_MULTIPLIER,
                 ) -> tuple[float, np.ndarray]:
    """essentia's (danceability, dfa) for a mono signal already at sample_rate.

    10 ms frames -> per-frame stddev -> mean removal -> cumulative sum -> for each tau the
    detrended fluctuation F(tau) -> DFA exponents between consecutive taus (with the +3 the C++
    adds to each tau) -> 1 / their mean. Zero danceability and an all-zero dfa when the signal is
    shorter than two taus, when some F is zero (silence, or len(s) == tau) or when the mean
    exponent is not positive, as in the C++. dfa has len(taus) - 1 entries, as in essentia.
    """
    y = np.ascontiguousarray(y, dtype=F32).ravel()
    taus = tau_frames(min_tau_ms, max_tau_ms, tau_multiplier)
    dfa = np.zeros(max(len(taus) - 1, 0), dtype=F32)
    frame_size = int(0.01 * float(F32(sample_rate)))
    n_frames = len(y) // frame_size
    if n_frames == 0 or n_frames < taus[0]:
        return 0.0, dfa  # too short for even the smallest tau

    s = _frame_stddev(y[: n_frames * frame_size].reshape(n_frames, frame_size))
    s = s - np.cumsum(s, dtype=F32)[-1] / F32(n_frames)
    s = np.cumsum(s, dtype=F32)

    fluct = []
    for tau in taus:
        if n_frames < tau:
            break
        fluct.append(_fluctuation(s, tau))

    total = F32(0.0)
    # inf/nan where the C++ gets them (F[i] == 0, or equal taus from a multiplier near 1) rather
    # than raising: they end in a zero danceability below, as in essentia
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(len(fluct) - 1):
            if fluct[i + 1] == 0.0:
                return 0.0, np.zeros_like(dfa)
            num = np.log10(fluct[i + 1] / fluct[i])  # float32 log10 of a float32 ratio
            den = math.log10((taus[i + 1] + 3.0) / (taus[i] + 3.0))  # double, as in the C++
            dfa[i] = F32(np.float64(num) / np.float64(den))
            total = F32(total + dfa[i])
    if len(fluct) <= 1:
        return 0.0, dfa
    mean_alpha = F32(total / F32(len(fluct) - 1))
    if not mean_alpha > 0.0:
        return 0.0, dfa
    return float(F32(1.0 / float(mean_alpha))), dfa
