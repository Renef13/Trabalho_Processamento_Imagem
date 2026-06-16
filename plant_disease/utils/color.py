"""Color space conversion and thresholding helpers.

Implementações solicitadas:
- F1.2: conversão BGR -> RGB (para visualização com matplotlib)
- F1.3: conversão BGR -> HSV (espaço principal de análise)

As funções aceitam um numpy.ndarray (imagem no formato BGR, como lido
por OpenCV) e retornam uma nova imagem no espaço de cor desejado ou
None em caso de entrada inválida.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

__all__ = ["bgr_to_rgb", "bgr_to_hsv"]


def bgr_to_rgb(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
	"""Convert an image from BGR (OpenCV) to RGB (matplotlib-friendly).

	Returns a new numpy array with channels converted or None if the
	input is invalid.
	"""
	if image is None:
		return None
	if not isinstance(image, np.ndarray):
		return None
	# Expect 2D (grayscale) or 3D images. For grayscale, return a copy.
	try:
		if image.ndim == 2:
			return image.copy()
		if image.shape[-1] == 3:
			return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# Unexpected channel number: try to return a copy to avoid side-effects
		return image.copy()
	except Exception:
		return None


def bgr_to_hsv(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
	"""Convert an image from BGR (OpenCV) to HSV.

	Returns a new numpy array in HSV color space or None if the input is
	invalid. If the input is grayscale (2D), returns None because HSV
	requires 3 channels.
	"""
	if image is None:
		return None
	if not isinstance(image, np.ndarray):
		return None
	try:
		if image.ndim == 2:
			# cannot convert single-channel to HSV directly
			return None
		if image.shape[-1] == 3:
			return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		return None
	except Exception:
		return None

