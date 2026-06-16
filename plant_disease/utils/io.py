"""File I/O helpers for image datasets.

Implementações:
- load_image(path, flags=cv2.IMREAD_COLOR) -> Optional[np.ndarray]
- load_batch(paths_or_dir, extensions=(...), recursive=False, flags=...)
- save_result(output_path, image, params=None) -> bool

As funções usam OpenCV (cv2.imread / cv2.imwrite). Apenas este arquivo foi
modificado conforme solicitado (F1).
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np


def load_image(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
	"""Load a single image using cv2.imread.

	Returns the image (numpy array) or None if the file doesn't exist or
	couldn't be read.
	"""
	if not path:
		return None
	if not os.path.exists(path):
		return None
	img = cv2.imread(path, flags)
	return img


def load_batch(
	paths_or_dir: Union[str, List[str]],
	extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
	recursive: bool = False,
	flags: int = cv2.IMREAD_COLOR,
) -> List[Tuple[str, np.ndarray]]:
	"""Load multiple images.

	- If `paths_or_dir` is a directory path, it will collect files matching
	  `extensions` and load them. If `recursive` is False only the top-level
	  directory is scanned.
	- If `paths_or_dir` is a string that is a file path it will try to load
	  that single file and return a list with one tuple (path, image) if
	  successful.
	- If `paths_or_dir` is a list of file paths, each existing and readable
	  image will be returned as (path, image).

	Returns a list of tuples (path, image) for successfully read images.
	"""
	results: List[Tuple[str, np.ndarray]] = []

	def _try_append(p: str) -> None:
		try:
			img = cv2.imread(p, flags)
		except Exception:
			img = None
		if img is not None:
			results.append((p, img))

	if isinstance(paths_or_dir, str):
		if os.path.isdir(paths_or_dir):
			# walk directory
			for root, dirs, files in os.walk(paths_or_dir):
				for f in files:
					if f.lower().endswith(tuple(ext.lower() for ext in extensions)):
						_try_append(os.path.join(root, f))
				if not recursive:
					break
			return results
		else:
			# single file path
			if os.path.exists(paths_or_dir):
				_try_append(paths_or_dir)
			return results

	# assume iterable of paths
	for p in paths_or_dir:
		if not p:
			continue
		if os.path.exists(p):
			_try_append(p)
	return results


def save_result(output_path: str, image: np.ndarray, params: Optional[List[int]] = None) -> bool:
	"""Save image to disk using cv2.imwrite.

	Creates parent directories if needed. Returns True on success, False
	otherwise.
	"""
	if image is None:
		return False
	out_dir = os.path.dirname(output_path)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)
	try:
		# cv2.imwrite expects params as a list of ints or None
		return cv2.imwrite(output_path, image, params or [])
	except Exception:
		return False
