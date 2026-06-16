"""Stage 1: preprocessing routines.

Implements:
- F1.4: smoothing with GaussianBlur
- F1.5: resizing to 256x256

The functions are small wrappers around OpenCV utilities with basic
input validation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = ["gaussian_blur", "resize_to_size"]


def _ensure_odd_positive(x: int) -> int:
	"""Return the nearest odd positive integer >= 1 for kernel sizes."""
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


def gaussian_blur(image: Optional[np.ndarray], ksize: Tuple[int, int] = (5, 5), sigmaX: float = 0.0) -> Optional[np.ndarray]:
	"""Apply Gaussian blur to an image.

	Parameters
	- image: BGR or grayscale image as numpy.ndarray
	- ksize: kernel size as (width, height). Values will be converted to
	  odd positive integers if needed (OpenCV requirement).
	- sigmaX: Gaussian kernel standard deviation in X direction (see
	  cv2.GaussianBlur docs)

	Returns the blurred image or None for invalid input.
	"""
	if image is None:
		return None
	if not isinstance(image, np.ndarray):
		return None
	try:
		kx = _ensure_odd_positive(ksize[0])
		ky = _ensure_odd_positive(ksize[1])
		k = (kx, ky)
		return cv2.GaussianBlur(image, k, sigmaX)
	except Exception:
		return None


def resize_to_size(image: Optional[np.ndarray], size: Tuple[int, int] = (256, 256), interpolation: int = cv2.INTER_AREA) -> Optional[np.ndarray]:
	"""Resize image to given size (width, height).

	Defaults to 256x256 (F1.5). Returns None for invalid input.
	"""
	if image is None:
		return None
	if not isinstance(image, np.ndarray):
		return None
	try:
		width = int(size[0])
		height = int(size[1])
		if width <= 0 or height <= 0:
			return None
		# cv2.resize takes (width, height)
		return cv2.resize(image, (width, height), interpolation=interpolation)
	except Exception:
		return None
