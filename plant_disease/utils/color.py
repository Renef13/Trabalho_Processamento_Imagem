"""Auxiliares para conversão de espaço de cores e limiarização.

Implementações solicitadas:

- F1.2: conversão BGR -> RGB (para visualização com matplotlib)
- F1.3: conversão BGR -> HSV (espaço principal de análise)

As funções recebem um numpy.ndarray (imagem no formato BGR, como lido
pelo OpenCV) e retornam uma nova imagem no espaço de cor desejado ou
None em caso de input inválido.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

__all__ = ["bgr_to_rgb", "bgr_to_hsv"]


def bgr_to_rgb(imagem: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Converte uma imagem de BGR (OpenCV) para RGB.

    Retorna um novo numpy.ndarray com os canais convertidos ou None
    caso o input seja inválido.
    """
    if imagem is None:
        return None

    if not isinstance(imagem, np.ndarray):
        return None

    # Espera imagens 2D (tons de cinza) ou 3D.
    # Para tons de cinza, retorna uma cópia.
    try:
        if imagem.ndim == 2:
            return imagem.copy()

        if imagem.shape[-1] == 3:
            return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        # Quantidade de canais inesperada:
        # retorna uma cópia para evitar efeitos colaterais.
        return imagem.copy()

    except Exception:
        return None


def bgr_to_hsv(imagem: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Converte uma imagem de BGR (OpenCV) para HSV.

    Retorna um novo numpy.ndarray no espaço HSV ou None caso o input
    seja inválido. Se a imagem for monocromática (2D), retorna None,
    pois HSV requer 3 canais.
    """
    if imagem is None:
        return None

    if not isinstance(imagem, np.ndarray):
        return None

    try:
        if imagem.ndim == 2:
            # Não é possível converter imagem de canal único para HSV.
            return None

        if imagem.shape[-1] == 3:
            return cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

        return None

    except Exception:
        return None