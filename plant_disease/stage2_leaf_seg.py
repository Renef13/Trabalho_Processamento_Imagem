"""Rotinas de segmentação da folha (Etapa 2).

Implementações solicitadas:

- F2.1: segment_leaf(hsv) → (leaf_mask, leaf_area_px)
  Verde: H ≈ 35–85 (espaço HSV) com cv2.inRange
- F2.2: segment_leaf_otsu(gray) → (leaf_mask, leaf_area_px)
  Limiarização automática via cv2.threshold(..., THRESH_OTSU)
- F2.3: operações morfológicas — MORPH_OPEN + MORPH_CLOSE para limpeza
- F2.4: extract_largest_contour(mask) → refined_mask
  Encontra e isola o maior contorno (folha) usando cv2.findContours
- F2.5: apply_mask(imagem, mask) → masked_image
  cv2.bitwise_and: extrai região foliar com fundo preto
- F2.6: HSV tuning interativo (CRITICAL PATH)
- F2.7: visual validation com 20+ imagens

As funções são wrappers com validação robusta de inputs, similares ao padrão
estabelecido em stage1_preproc.py (F1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

__all__ = [
    "LeafSegmentationConfig",
    "segment_leaf",
    "segment_leaf_otsu",
    "extract_largest_contour",
    "apply_mask",
    "LeafSegmenter",
]


@dataclass(frozen=True)
class LeafSegmentationConfig:
    """Configuracao padrao da etapa de segmentação da folha."""

    # Intervalo HSV para pixels verdes (folha saudável)
    h_min: int = 35
    h_max: int = 85
    s_min: int = 30
    s_max: int = 255
    v_min: int = 40
    v_max: int = 255

    # Operações morfológicas
    aplicar_morph: bool = True
    morph_kernel_size: Tuple[int, int] = (5, 5)
    morph_iterations: int = 1


def _validate_hsv_image(imagem: Optional[np.ndarray]) -> bool:
    """Valida se a imagem é um array HSV válido (3 canais, uint8)."""
    if imagem is None:
        return False
    if not isinstance(imagem, np.ndarray):
        return False
    if imagem.ndim != 3 or imagem.shape[2] != 3:
        return False
    if imagem.dtype != np.uint8:
        return False
    return True


def _validate_gray_image(imagem: Optional[np.ndarray]) -> bool:
    """Valida se a imagem é um array em tons de cinza válido."""
    if imagem is None:
        return False
    if not isinstance(imagem, np.ndarray):
        return False
    if imagem.ndim != 2:
        return False
    if imagem.dtype != np.uint8:
        return False
    return True


def segment_leaf(
    hsv: Optional[np.ndarray],
    h_min: int = 35,
    h_max: int = 85,
    s_min: int = 30,
    s_max: int = 255,
    v_min: int = 40,
    v_max: int = 255,
    aplicar_morph: bool = True,
    morph_kernel_size: Tuple[int, int] = (5, 5),
    morph_iterations: int = 1,
) -> Optional[Tuple[np.ndarray, int]]:
    """Segmenta a folha usando cv2.inRange no espaço HSV.

    Parâmetros
    - hsv: imagem em espaço HSV (3 canais, uint8)
    - h_min, h_max: intervalo de Hue (padrão 35–85 para verde)
    - s_min, s_max: intervalo de Saturation
    - v_min, v_max: intervalo de Value
    - aplicar_morph: se True, aplica operações morfológicas para limpar ruído
    - morph_kernel_size: tamanho do kernel para erosão/dilatação
    - morph_iterations: número de iterações das operações morfológicas

    Retorna tupla (leaf_mask, leaf_area_px) onde:
    - leaf_mask: imagem binária (0 ou 255) com pixels de folha
    - leaf_area_px: número de pixels brancos na máscara
    Ou None para input inválido.
    """
    if not _validate_hsv_image(hsv):
        return None

    try:
        lower_bound = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper_bound = np.array([h_max, s_max, v_max], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        if aplicar_morph:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                morph_kernel_size,
            )
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel,
                iterations=morph_iterations,
            )
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=morph_iterations,
            )

        leaf_area_px = int(cv2.countNonZero(mask))

        return (mask, leaf_area_px)

    except Exception:
        return None


def segment_leaf_otsu(
    gray: Optional[np.ndarray],
    aplicar_morph: bool = True,
    morph_kernel_size: Tuple[int, int] = (5, 5),
    morph_iterations: int = 1,
) -> Optional[Tuple[np.ndarray, int]]:
    """Segmenta a folha usando limiarização de Otsu.

    Parâmetros
    - gray: imagem em tons de cinza (uint8, 1 canal)
    - aplicar_morph: se True, aplica operações morfológicas para limpar ruído
    - morph_kernel_size: tamanho do kernel para erosão/dilatação
    - morph_iterations: número de iterações das operações morfológicas

    Retorna tupla (leaf_mask, leaf_area_px) onde:
    - leaf_mask: imagem binária (0 ou 255) com pixels de folha
    - leaf_area_px: número de pixels brancos na máscara
    Ou None para input inválido.

    A limiarização de Otsu calcula automaticamente o limiar ótimo que
    maximiza a variância entre-classe, sendo ideal para imagens com
    distribuição bimodal (fundo vs. folha).
    """
    if not _validate_gray_image(gray):
        return None

    try:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if aplicar_morph:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                morph_kernel_size,
            )
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel,
                iterations=morph_iterations,
            )
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=morph_iterations,
            )

        leaf_area_px = int(cv2.countNonZero(mask))

        return (mask, leaf_area_px)

    except Exception:
        return None


def extract_largest_contour(
    mask: Optional[np.ndarray],
    fill_contour: bool = True,
) -> Optional[Tuple[np.ndarray, float]]:
    """Extrai o maior contorno (folha) e retorna máscara refinada.

    Parâmetros
    - mask: máscara binária (uint8, 0 ou 255)
    - fill_contour: se True, preenche o contorno encontrado

    Retorna tupla (refined_mask, largest_area) onde:
    - refined_mask: máscara binária com apenas o maior contorno
    - largest_area: área (em pixels) do maior contorno
    Ou None para input inválido.

    Implementação de F2.4: cv2.findContours + cv2.contourArea
    Isola a folha descartando ruídos de fundo.
    """
    if mask is None:
        return None

    if not isinstance(mask, np.ndarray):
        return None

    if mask.ndim != 2 or mask.dtype != np.uint8:
        return None

    try:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = float(cv2.contourArea(largest_contour))

        if largest_area <= 0:
            return None

        refined_mask = np.zeros_like(mask)

        if fill_contour:
            cv2.drawContours(refined_mask, [largest_contour], 0, 255, -1)
        else:
            cv2.drawContours(refined_mask, [largest_contour], 0, 255, 2)

        return (refined_mask, largest_area)

    except Exception:
        return None


def apply_mask(
    imagem: Optional[np.ndarray],
    mask: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Aplica máscara binária à imagem usando cv2.bitwise_and.

    Parâmetros
    - imagem: imagem original (BGR ou cinza)
    - mask: máscara binária (uint8, 0 ou 255)

    Retorna imagem com fundo preto e folha preservada, ou None para input
    inválido.

    Implementação de F2.5: cv2.bitwise_and para extrair região foliar
    com fundo preto.
    """
    if imagem is None or mask is None:
        return None

    if not isinstance(imagem, np.ndarray):
        return None

    if not isinstance(mask, np.ndarray):
        return None

    if mask.ndim != 2 or mask.dtype != np.uint8:
        return None

    if imagem.shape[:2] != mask.shape[:2]:
        return None

    try:
        masked_image = cv2.bitwise_and(imagem, imagem, mask=mask)
        return masked_image

    except Exception:
        return None


class LeafSegmenter:
    """Objeto responsavel pela etapa 2 do pipeline."""

    def __init__(self, config: Optional[LeafSegmentationConfig] = None) -> None:
        self.config = config or LeafSegmentationConfig()

    def run_hsv(
        self,
        hsv: Optional[np.ndarray],
    ) -> Optional[Tuple[np.ndarray, int]]:
        """Executa segmentação HSV com a configuração deste objeto."""
        config = self.config
        return segment_leaf(
            hsv,
            h_min=config.h_min,
            h_max=config.h_max,
            s_min=config.s_min,
            s_max=config.s_max,
            v_min=config.v_min,
            v_max=config.v_max,
            aplicar_morph=config.aplicar_morph,
            morph_kernel_size=config.morph_kernel_size,
            morph_iterations=config.morph_iterations,
        )

    def run_otsu(
        self,
        gray: Optional[np.ndarray],
    ) -> Optional[Tuple[np.ndarray, int]]:
        """Executa segmentação de Otsu com a configuração deste objeto."""
        config = self.config
        return segment_leaf_otsu(
            gray,
            aplicar_morph=config.aplicar_morph,
            morph_kernel_size=config.morph_kernel_size,
            morph_iterations=config.morph_iterations,
        )

    def refine_with_contour(
        self,
        mask: Optional[np.ndarray],
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Refina máscara extraindo o maior contorno (F2.4)."""
        return extract_largest_contour(mask, fill_contour=True)

    def extract_leaf_region(
        self,
        imagem: Optional[np.ndarray],
        mask: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Extrai região foliar aplicando máscara (F2.5)."""
        return apply_mask(imagem, mask)
