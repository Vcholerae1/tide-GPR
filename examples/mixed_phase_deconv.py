from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
TaperKind = Literal[
    "hamming",
    "hann",
    "hanning",
    "bartlett",
    "triang",
    "blackman",
    "boxcar",
    "cos",
]


@dataclass(frozen=True, slots=True)
class WaveletEstimate:
    """A trace-dependent wavelet estimate.

    Attributes
    ----------
    coordinate:
        Coordinate vector of the wavelet samples.
    values:
        Wavelet samples with shape ``(n_wavelet_samples, n_traces)``.
    """

    coordinate: FloatArray
    values: FloatArray


@dataclass(frozen=True, slots=True)
class SpikingFilterResult:
    """Estimated spiking inverse filter."""

    inverse_filter: FloatArray
    delay: int


@dataclass(frozen=True, slots=True)
class MixedPhaseResult:
    """Outputs from mixed-phase spiking deconvolution."""

    mixed: FloatArray
    minimum_phase: FloatArray
    inverse_filters: FloatArray
    minimum_phase_wavelets: FloatArray
    minimum_phase_wavelet: WaveletEstimate
    mixed_phase_wavelet: WaveletEstimate
    phase_radians: float
    phase_degrees: float
    window_indices: NDArray[np.int_]
    window_mask: NDArray[np.bool_]
    filter_length: int
    delay: int


def _as_trace_matrix(values: ArrayLike) -> FloatArray:
    """Return ``values`` as ``(n_samples, n_traces)`` float64 data."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim == 2:
        return array
    raise ValueError("values must be a scalar, vector, or 2-D matrix")


def _as_row_major_vector(values: ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _toeplitz_from_first_column(column: ArrayLike) -> FloatArray:
    values = np.asarray(column, dtype=np.float64).reshape(-1)
    offsets = np.abs(np.subtract.outer(np.arange(values.size), np.arange(values.size)))
    return values[offsets]


def _solve_symmetric_system(matrix: FloatArray, rhs: FloatArray) -> FloatArray:
    """Solve a symmetric system, using Cholesky when the matrix is SPD."""
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.solve(matrix, rhs)

    y = np.linalg.solve(factor, rhs)
    return np.linalg.solve(factor.T, y)


def _bounded_argmax(
    objective: Callable[[float], float],
    lower: float,
    upper: float,
    tolerance: float,
    *,
    max_iter: int = 1000,
) -> float:
    """Golden-section search for a bounded scalar maximum."""
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    inv_phi_sq = (3.0 - np.sqrt(5.0)) / 2.0

    lower = float(lower)
    upper = float(upper)
    width = upper - lower
    if width <= tolerance:
        return (lower + upper) / 2.0

    left = lower + inv_phi_sq * width
    right = lower + inv_phi * width
    f_left = objective(left)
    f_right = objective(right)

    for _ in range(max_iter):
        if abs(upper - lower) <= tolerance:
            break

        f_left = f_left if np.isfinite(f_left) else -np.inf
        f_right = f_right if np.isfinite(f_right) else -np.inf

        if f_left < f_right:
            lower = left
            left = right
            f_left = f_right
            width = upper - lower
            right = lower + inv_phi * width
            f_right = objective(right)
        else:
            upper = right
            right = left
            f_right = f_left
            width = upper - lower
            left = lower + inv_phi_sq * width
            f_left = objective(left)

    return (lower + upper) / 2.0


def hamming_window(length: int) -> FloatArray:
    """Return a Hamming window as float64."""
    length = int(length)
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    return np.hamming(length).astype(np.float64)


def pad_edges(
    values: ArrayLike,
    pad_samples: int,
    pad_traces: int | None = None,
    *,
    constant: bool = False,
) -> FloatArray:
    """Pad trace data along sample and trace axes.

    Parameters
    ----------
    values:
        Scalar, vector, or matrix. Vectors are treated as one trace.
    pad_samples:
        Number of samples to add above and below.
    pad_traces:
        Number of traces to add on each side. Defaults to ``pad_samples``.
    constant:
        If ``True``, pad with zeros. Otherwise repeat edge values.
    """
    matrix = _as_trace_matrix(values)
    pad_samples = int(pad_samples)
    pad_traces = pad_samples if pad_traces is None else int(pad_traces)
    if pad_samples < 0 or pad_traces < 0:
        raise ValueError("pad widths must be non-negative")

    mode = "constant" if constant else "edge"
    return np.pad(
        matrix,
        ((pad_samples, pad_samples), (pad_traces, pad_traces)),
        mode=mode,
    )


def rms_normalize_traces(values: ArrayLike) -> FloatArray:
    """Normalize each trace by its RMS amplitude."""
    traces = _as_trace_matrix(values).copy()
    n_samples = traces.shape[0]
    energy = np.sum(traces * traces, axis=0)
    denominator = max(n_samples - 1, 1)
    scale = np.sqrt(energy / denominator)
    valid = np.isfinite(scale) & (scale != 0.0)
    traces[:, valid] /= scale[valid]
    return traces


def _cosine_taper(length: int, fraction: float) -> FloatArray:
    if not 0.0 <= fraction <= 0.5:
        raise ValueError("fraction must be between 0 and 0.5")

    weights = np.ones(length, dtype=np.float64)
    if fraction == 0.0 or length <= 1:
        return weights

    samples = np.arange(length, dtype=np.float64)
    denominator = length - 1
    left = samples / length <= fraction
    right = (1.0 - fraction) <= samples / denominator
    weights[left] = 0.5 * (
        1.0 - np.cos(np.pi * (samples[left] / denominator / fraction))
    )
    weights[right] = 0.5 * (
        1.0 - np.cos(np.pi * ((1.0 - samples[right] / denominator) / fraction))
    )
    return weights


def window_taper(
    length: int,
    *,
    kind: TaperKind = "hamming",
    half: bool = False,
    reverse: bool = False,
    cosine_fraction: float = 0.1,
) -> FloatArray:
    """Create a common 1-D taper window."""
    length = int(length)
    if length <= 0:
        return np.empty(0, dtype=np.float64)

    output_length = length
    full_length = 2 * length + 1 if half else length
    windows: dict[str, Callable[[int], FloatArray]] = {
        "hamming": hamming_window,
        "hanning": lambda n: np.hanning(n).astype(np.float64),
        "hann": lambda n: np.hanning(n).astype(np.float64),
        "bartlett": lambda n: np.bartlett(n).astype(np.float64),
        "triang": lambda n: np.bartlett(n).astype(np.float64),
        "blackman": lambda n: np.blackman(n).astype(np.float64),
        "boxcar": lambda n: np.ones(n, dtype=np.float64),
        "cos": lambda n: _cosine_taper(n, cosine_fraction),
    }

    try:
        weights = windows[kind](full_length)
    except KeyError as exc:
        supported = ", ".join(sorted(windows))
        raise ValueError(
            f"unsupported taper kind {kind!r}; choose one of: {supported}"
        ) from exc

    if not half:
        return weights
    if reverse:
        return weights[output_length : 2 * output_length]
    return weights[:output_length]


def convolve_traces(values: ArrayLike, filters: ArrayLike) -> FloatArray:
    """Convolve each trace with its corresponding FIR filter.

    If only one filter is supplied, it is broadcast to all traces.
    """
    traces = _as_trace_matrix(values)
    kernels = _as_trace_matrix(filters)
    n_samples, n_traces = traces.shape

    if kernels.shape[1] == 1 and n_traces > 1:
        kernels = np.tile(kernels, (1, n_traces))
    elif kernels.shape[1] < n_traces:
        raise ValueError(
            "filters must have either one column or at least as many columns as values"
        )
    else:
        kernels = kernels[:, :n_traces]

    filter_length = kernels.shape[0]
    padded = pad_edges(traces, filter_length, 0, constant=True)
    impulse_response = np.zeros_like(padded)
    impulse_response[:filter_length, :] = kernels

    filtered = np.fft.ifft(
        np.fft.fft(padded, axis=0) * np.fft.fft(impulse_response, axis=0),
        axis=0,
    ).real
    return filtered[filter_length : filter_length + n_samples, :]


def normalized_autocorrelation(values: ArrayLike, max_lag: int) -> FloatArray:
    """Return normalized autocorrelation from lag 0 to ``max_lag``."""
    samples = _as_row_major_vector(values)
    n_samples = samples.size
    if n_samples == 0:
        raise ValueError("values must not be empty")

    max_lag = int(min(max_lag, n_samples - 1))
    centered = samples - np.mean(samples)
    lag_zero = float(np.dot(centered, centered))
    if lag_zero == 0.0:
        raise ValueError("lag-zero autocorrelation is zero")

    return np.array(
        [
            np.dot(centered[: n_samples - lag], centered[lag:]) / lag_zero
            for lag in range(max_lag + 1)
        ],
        dtype=np.float64,
    )


def estimate_spiking_inverse_filter(
    values: ArrayLike,
    *,
    length: int = 35,
    delay: int | None = None,
    prewhitening: float = 0.001,
    taper_kind: TaperKind = "hamming",
) -> SpikingFilterResult:
    """Estimate a minimum-phase spiking inverse filter."""
    length = int(length)
    if length <= 0:
        raise ValueError("length must be positive")
    if delay is None:
        raise NotImplementedError("automatic delay selection is not implemented")

    first_trace = _as_trace_matrix(values)[:, 0]
    acf = normalized_autocorrelation(first_trace, max_lag=length - 1)
    effective_length = acf.size
    acf *= window_taper(
        effective_length,
        kind=taper_kind,
        half=True,
        reverse=True,
        cosine_fraction=0.1,
    )
    acf /= acf[0]
    acf[0] += float(prewhitening) ** 2

    delay = int(delay)
    if delay < 1 or delay > effective_length:
        raise ValueError(f"delay must be in [1, {effective_length}], got {delay}")

    spike = np.zeros(effective_length, dtype=np.float64)
    spike[delay - 1] = 1.0
    inverse_filter = _solve_symmetric_system(_toeplitz_from_first_column(acf), spike)
    return SpikingFilterResult(inverse_filter=np.asarray(inverse_filter), delay=delay)


def hilbert_transform(
    values: ArrayLike,
    pad: int = 10,
    *,
    npad: int | None = None,
) -> ComplexArray:
    """Return the Hilbert-transform quadrature signal used for phase rotation.

    Matrix inputs are transformed column-wise in one FFT call.
    """
    if npad is not None:
        pad = npad
    array = np.asarray(values, dtype=np.float64)
    is_vector = array.ndim <= 1
    signal = _as_trace_matrix(array)
    n_original = signal.shape[0]
    pad = int(pad)
    if pad < 0:
        raise ValueError("pad must be non-negative")
    if n_original == 0:
        return np.empty(0, dtype=np.complex128)
    if pad > n_original:
        raise ValueError("pad must not exceed the vector length")

    if pad == 0:
        padded = signal.copy()
        start = 0
    else:
        padded = np.concatenate(
            (signal[:pad][::-1, :], signal, signal[-pad:][::-1, :]),
            axis=0,
        )
        start = pad

    spectrum = np.fft.fft(padded, axis=0)
    frequencies = np.arange(padded.shape[0], dtype=np.float64)
    frequencies -= np.mean(frequencies)
    multiplier = (-np.sign(frequencies - 0.5))[:, None]
    transformed = np.fft.ifft(-1j * spectrum * multiplier, axis=0)
    result = transformed[start : start + n_original].astype(np.complex128)
    return result[:, 0] if is_vector else result


def phase_rotate(
    values: ArrayLike,
    phase_radians: float,
    pad: int = 20,
    *,
    npad: int | None = None,
) -> FloatArray:
    """Apply constant phase rotation to a vector or each trace of a matrix."""
    if npad is not None:
        pad = npad
    array = np.asarray(values, dtype=np.float64)
    is_vector = array.ndim <= 1
    if array.ndim > 2:
        raise ValueError("values must be a vector or a 2-D matrix")

    signal = _as_trace_matrix(array)
    quadrature = np.real(hilbert_transform(signal, pad=pad))
    rotated = signal * np.cos(phase_radians) - quadrature * np.sin(phase_radians)
    return rotated[:, 0] if is_vector else rotated


def excess_kurtosis(values: ArrayLike) -> float:
    """Return excess kurtosis using the same normalization as the original code."""
    samples = _as_row_major_vector(values)
    n_samples = samples.size
    if n_samples <= 1:
        return float("nan")

    centered_sq = (samples - np.mean(samples)) ** 2
    numerator = np.sum(centered_sq**2) / n_samples
    denominator = (np.sum(centered_sq) / (n_samples - 1)) ** 2
    return float(numerator / denominator - 3.0)


def find_optimal_phase_rotation(
    values: ArrayLike, *, tol: float | None = None
) -> float:
    """Find the phase rotation that maximizes excess kurtosis."""
    samples = _as_row_major_vector(values)
    tolerance = float(np.finfo(float).eps ** 0.25) if tol is None else float(tol)
    return float(
        _bounded_argmax(
            lambda theta: excess_kurtosis(phase_rotate(samples, theta)),
            0.0,
            np.pi,
            tolerance,
        )
    )


def _window_indices(
    coordinate: FloatArray, window: Sequence[float]
) -> NDArray[np.int_]:
    bounds = np.sort(np.asarray(window, dtype=np.float64))
    if bounds.size != 2:
        raise ValueError("window must contain exactly two values")

    selected = np.flatnonzero((coordinate > bounds[0]) & (coordinate < bounds[1]))
    if selected.size == 0:
        raise ValueError("window selects no samples")
    return selected


def _neighbor_indices(center: int, radius: int, trace_count: int) -> NDArray[np.int_]:
    start = max(0, center - radius)
    stop = min(trace_count, center + radius + 1)
    return np.arange(start, stop)


def _wavelet_coordinate(
    coordinate: FloatArray,
    filter_length: int,
    sample_spacing: float | None,
) -> FloatArray:
    if sample_spacing is None:
        sample_spacing = (
            float(np.median(np.diff(coordinate))) if coordinate.size >= 2 else 1.0
        )
    pre_samples = int(np.round(filter_length / 3.0))
    return np.arange(-pre_samples, filter_length + 1, dtype=np.float64) * sample_spacing


def mixed_phase_deconvolution(
    data: ArrayLike,
    coordinate: ArrayLike,
    window: tuple[float, float] | Sequence[float],
    trace_radius: int,
    filter_length: int,
    prewhitening: float,
    *,
    delay: int = 1,
    sample_spacing: float | None = None,
    taper_kind: TaperKind = "hamming",
) -> MixedPhaseResult:
    """Run mixed-phase spiking deconvolution.

    Parameters
    ----------
    data:
        Input traces with shape ``(n_samples, n_traces)``. A 1-D vector is
        treated as one trace.
    coordinate:
        Sample coordinate vector, usually time/depth, with length ``n_samples``.
    window:
        Two coordinate bounds used to estimate the inverse filters and optimal
        phase rotation.
    trace_radius:
        Number of neighboring traces on each side used to build each supertrace.
    filter_length:
        Requested inverse-filter length. It is clipped to the number of samples
        selected by ``window``.
    prewhitening:
        Stabilization factor added to the autocorrelation zero lag as
        ``prewhitening ** 2``.
    delay:
        One-based spike delay.
    sample_spacing:
        Wavelet coordinate spacing. If omitted, the median coordinate spacing is
        used.
    taper_kind:
        Taper applied to the autocorrelation before solving the inverse filter.
    """
    traces_in = _as_trace_matrix(data)
    coordinate_array = np.asarray(coordinate, dtype=np.float64).reshape(-1)
    if coordinate_array.size != traces_in.shape[0]:
        raise ValueError("coordinate length must match data.shape[0]")

    trace_radius = int(trace_radius)
    filter_length = int(filter_length)
    delay = int(delay)
    if trace_radius < 0:
        raise ValueError("trace_radius must be non-negative")
    if filter_length <= 0:
        raise ValueError("filter_length must be positive")
    if delay < 1:
        raise ValueError("delay must be a one-based index >= 1")

    window_indices = _window_indices(coordinate_array, window)
    filter_length = min(filter_length, int(window_indices.size))
    if delay > filter_length:
        raise ValueError("delay cannot be larger than the effective filter_length")

    traces = rms_normalize_traces(traces_in)
    n_samples, n_traces = traces.shape
    deconvolved = np.empty((n_samples, n_traces), dtype=np.float64)
    inverse_filters = np.empty((filter_length, n_traces), dtype=np.float64)

    for trace_index in range(n_traces):
        neighbors = _neighbor_indices(trace_index, trace_radius, n_traces)
        supertrace = traces[np.ix_(window_indices, neighbors)].reshape(-1)
        inverse_filter = estimate_spiking_inverse_filter(
            supertrace,
            length=filter_length,
            delay=delay,
            prewhitening=prewhitening,
            taper_kind=taper_kind,
        ).inverse_filter
        inverse_filters[:, trace_index] = inverse_filter
        deconvolved[:, trace_index] = convolve_traces(
            traces[:, trace_index], inverse_filter
        )[:, 0]

    wavelets = np.fft.ifft(1.0 / np.fft.fft(inverse_filters, axis=0), axis=0).real
    wavelet_coordinate = _wavelet_coordinate(
        coordinate_array, filter_length, sample_spacing
    )
    pre_samples = int(np.round(filter_length / 3.0))
    wavelet_values = np.vstack(
        (
            np.zeros((pre_samples, n_traces), dtype=np.float64),
            wavelets,
            np.zeros((1, n_traces), dtype=np.float64),
        )
    )
    minimum_phase_wavelet = WaveletEstimate(wavelet_coordinate, wavelet_values)

    phase = find_optimal_phase_rotation(deconvolved[window_indices, :])
    mixed = phase_rotate(deconvolved, phase)
    mixed_phase_wavelet = WaveletEstimate(
        wavelet_coordinate.copy(), phase_rotate(wavelet_values, -phase)
    )

    window_mask = np.zeros(n_samples, dtype=bool)
    window_mask[window_indices] = True

    return MixedPhaseResult(
        mixed=mixed,
        minimum_phase=deconvolved,
        inverse_filters=inverse_filters,
        minimum_phase_wavelets=wavelets,
        minimum_phase_wavelet=minimum_phase_wavelet,
        mixed_phase_wavelet=mixed_phase_wavelet,
        phase_radians=phase,
        phase_degrees=float(np.degrees(phase)),
        window_indices=window_indices,
        window_mask=window_mask,
        filter_length=filter_length,
        delay=delay,
    )


def mean_normalized_wavelet(wavelet: FloatArray | WaveletEstimate) -> FloatArray:
    """Average trace-dependent wavelets and normalize their peak amplitude."""
    if isinstance(wavelet, WaveletEstimate):
        values = wavelet.values
    else:
        values = wavelet

    mean_wavelet = np.mean(_as_trace_matrix(values), axis=1)
    peak = np.max(np.abs(mean_wavelet))
    return mean_wavelet if peak == 0 else mean_wavelet / peak


__all__ = [
    "ComplexArray",
    "FloatArray",
    "MixedPhaseResult",
    "SpikingFilterResult",
    "TaperKind",
    "WaveletEstimate",
    "convolve_traces",
    "estimate_spiking_inverse_filter",
    "excess_kurtosis",
    "find_optimal_phase_rotation",
    "hamming_window",
    "hilbert_transform",
    "mean_normalized_wavelet",
    "mixed_phase_deconvolution",
    "normalized_autocorrelation",
    "pad_edges",
    "phase_rotate",
    "rms_normalize_traces",
    "window_taper",
]
