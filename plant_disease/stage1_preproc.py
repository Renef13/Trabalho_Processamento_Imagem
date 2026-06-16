"""Rotinas de pré-processamento (Etapa 1).

Implementações:

- F1.4: suavização com GaussianBlur
- F1.5: redimensionamento para 256x256

As funções são pequenos wrappers das utilidades do OpenCV com validação
básica de input.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = ["gaussian_blur", "resize_to_size"]


def _ensure_odd_positive(x: int) -> int:
    """Retorna o inteiro ímpar positivo mais próximo >= 1 para kernels."""
    if x is None:
        return 1

    try:
        xi = int(x)
    except Exception:
        return 1

    if xi <= 0:
        xi = 1

    if xi % 2 == 0:
        xi += 1

    return xi


def gaussian_blur(
    imagem: Optional[np.ndarray],
    ksize: Tuple[int, int] = (5, 5),
    sigmaX: float = 0.0,
) -> Optional[np.ndarray]:
    """Aplica Gaussian blur a uma imagem.

    Parâmetros
    - imagem: imagem BGR ou em tons de cinza como numpy.ndarray
    - ksize: tamanho do kernel como (width, height). Os valores serão
      convertidos para inteiros ímpares positivos quando necessário
      (requisito do OpenCV).
    - sigmaX: desvio padrão do kernel gaussiano na direção X
      (consulte a documentação de cv2.GaussianBlur)

    Retorna a imagem com blur aplicado ou None para input inválido.
    """
    if imagem is None:
        return None

    if not isinstance(imagem, np.ndarray):
        return None

    try:
        kx = _ensure_odd_positive(ksize[0])
        ky = _ensure_odd_positive(ksize[1])
        k = (kx, ky)

        return cv2.GaussianBlur(imagem, k, sigmaX)

    except Exception:
        return None


def resize_to_size(
    imagem: Optional[np.ndarray],
    size: Tuple[int, int] = (256, 256),
    interpolation: int = cv2.INTER_AREA,
) -> Optional[np.ndarray]:
    """Redimensiona uma imagem para o tamanho especificado.

    O padrão é 256x256 (F1.5). Retorna None para input inválido.
    """
    if imagem is None:
        return None

    if not isinstance(imagem, np.ndarray):
        return None

    try:
        width = int(size[0])
        height = int(size[1])

        if width <= 0 or height <= 0:
            return None

        # cv2.resize recebe (width, height)
        return cv2.resize(
            imagem,
            (width, height),
            interpolation=interpolation,
        )

    except Exception:
        return None