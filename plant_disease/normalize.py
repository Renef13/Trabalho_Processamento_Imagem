"""Normalizacao experimental de cor para diagnosticos HSV.

Este modulo nao faz parte do pipeline padrao das fases F1-F5. Ele existe
para experimentos controlados da F6/correcao, mantendo o canal H intacto
e ajustando apenas S e V por histogram matching.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "calculate_channel_cdf",
    "match_channel_to_cdf",
    "match_sv_histogram",
]


def calculate_channel_cdf(values: np.ndarray) -> Optional[np.ndarray]:
    """Calcula a CDF normalizada de valores uint8 no intervalo 0-255."""
    if values is None:
        return None

    try:
        arr = np.asarray(values, dtype=np.uint8).ravel()
        if arr.size == 0:
            return None

        hist = np.bincount(arr, minlength=256).astype(np.float64)
        cdf = np.cumsum(hist)
        if cdf[-1] <= 0:
            return None

        return cdf / cdf[-1]

    except Exception:
        return None


def match_channel_to_cdf(channel: np.ndarray, reference_cdf: np.ndarray) -> Optional[np.ndarray]:
    """Ajusta um canal uint8 para seguir a CDF de referencia."""
    if channel is None or reference_cdf is None:
        return None

    try:
        src = np.asarray(channel, dtype=np.uint8)
        ref = np.asarray(reference_cdf, dtype=np.float64).ravel()
        if src.size == 0 or ref.size != 256:
            return None

        src_cdf = calculate_channel_cdf(src)
        if src_cdf is None:
            return None

        lut = np.interp(src_cdf, ref, np.arange(256))
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        return lut[src]

    except Exception:
        return None


def match_sv_histogram(
    hsv_source: np.ndarray,
    hsv_reference: Optional[np.ndarray] = None,
    reference_cdfs: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Optional[np.ndarray]:
    """Faz histogram matching apenas nos canais S e V.

    O canal H e preservado exatamente como recebido. A referencia pode ser
    uma imagem HSV (`hsv_reference`) ou um par de CDFs pre-calculadas
    `(s_cdf, v_cdf)`.
    """
    if hsv_source is None:
        return None

    try:
        source = np.asarray(hsv_source)
        if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
            return None

        if reference_cdfs is None:
            if hsv_reference is None:
                return None

            reference = np.asarray(hsv_reference)
            if (
                reference.ndim != 3
                or reference.shape[2] != 3
                or reference.dtype != np.uint8
            ):
                return None

            s_cdf = calculate_channel_cdf(reference[:, :, 1])
            v_cdf = calculate_channel_cdf(reference[:, :, 2])
        else:
            s_cdf, v_cdf = reference_cdfs

        if s_cdf is None or v_cdf is None:
            return None

        matched_s = match_channel_to_cdf(source[:, :, 1], s_cdf)
        matched_v = match_channel_to_cdf(source[:, :, 2], v_cdf)
        if matched_s is None or matched_v is None:
            return None

        result = source.copy()
        result[:, :, 1] = matched_s
        result[:, :, 2] = matched_v
        return result

    except Exception:
        return None
