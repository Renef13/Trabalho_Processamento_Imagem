"""Auxiliares de I/O para datasets de imagens.

Implementações:

- load_image(path, flags=cv2.IMREAD_COLOR) -> Optional[np.ndarray]
- load_batch(paths_or_dir, extensions=(...), recursive=False, flags=...)
- save_result(output_path, imagem, params=None) -> bool

As funções usam OpenCV (cv2.imread / cv2.imwrite). Apenas este arquivo foi
modificado conforme solicitado (F1).
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np


def load_image(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Carrega uma única imagem usando cv2.imread.

    Retorna a imagem (numpy.ndarray) ou None caso o arquivo não exista
    ou não possa ser lido.
    """
    if not path:
        return None

    if not os.path.exists(path):
        return None

    img = cv2.imread(path, flags)
    return img


def load_batch(
    paths_or_dir: Union[str, List[str]],
    extensions: Tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    ),
    recursive: bool = False,
    flags: int = cv2.IMREAD_COLOR,
) -> List[Tuple[str, np.ndarray]]:
    """Carrega múltiplas imagens.

    - Se `paths_or_dir` for o path de um diretório, coleta arquivos
      compatíveis com `extensions` e os carrega. Se `recursive` for
      False, apenas o nível superior do diretório será analisado.
    - Se `paths_or_dir` for uma string apontando para um arquivo,
      tenta carregar esse único arquivo e retorna uma lista contendo
      uma tupla (path, imagem) em caso de sucesso.
    - Se `paths_or_dir` for uma lista de paths, cada imagem existente
      e legível será retornada como (path, imagem).

    Retorna uma lista de tuplas (path, imagem) para as imagens lidas
    com sucesso.
    """
    resultados: List[Tuple[str, np.ndarray]] = []

    def _try_append(path_arquivo: str) -> None:
        try:
            img = cv2.imread(path_arquivo, flags)
        except Exception:
            img = None

        if img is not None:
            resultados.append((path_arquivo, img))

    if isinstance(paths_or_dir, str):
        if os.path.isdir(paths_or_dir):
            for root, dirs, files in os.walk(paths_or_dir):
                for file in files:
                    if file.lower().endswith(
                        tuple(ext.lower() for ext in extensions)
                    ):
                        _try_append(os.path.join(root, file))

                if not recursive:
                    break

            return resultados

        else:
            if os.path.exists(paths_or_dir):
                _try_append(paths_or_dir)

            return resultados

    for path in paths_or_dir:
        if not path:
            continue

        if os.path.exists(path):
            _try_append(path)

    return resultados


def save_result(
    output_path: str,
    imagem: np.ndarray,
    params: Optional[List[int]] = None,
) -> bool:
    """Salva uma imagem em disco usando cv2.imwrite.

    Cria diretórios pais quando necessário. Retorna True em caso de
    sucesso e False caso contrário.
    """
    if imagem is None:
        return False

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        # cv2.imwrite espera params como lista de inteiros ou None.
        return cv2.imwrite(output_path, imagem, params or [])
    except Exception:
        return False