"""Rotinas de detecção de lesões (Etapa 3).

Implementações:

- F3.1: LesionDetectionConfig — ranges HSV para amarelos e marrons (H circular)
- F3.2: segment_lesions_hsv(hsv, config) -> raw_lesion_mask
    multi-range cv2.inRange + cv2.bitwise_or
- F3.3: intersect_with_leaf(lesion_mask, leaf_mask) -> lesion_inside_leaf_mask
    cv2.bitwise_and, garante lesao sempre dentro da folha
- F3.4: refine_lesion_mask(mask, kernel_size, iterations)
    MORPH_OPEN + MORPH_CLOSE para limpar ruído
- F3.5: find_lesion_contours(lesion_mask, min_area) -> list[contours]
    cv2.findContours + filtro por área mínima
- F3.6: draw_lesion_contours(image_bgr, contours) -> imagem com contornos

Função central: detect_lesions(hsv, leaf_mask) -> (lesion_mask, contours)
Integra F3.2 -> F3.3 -> F3.4 -> F3.5 em um único passo, seguindo o mesmo
padrão de validação de input estabelecido em stage1_preproc.py e
stage2_leaf_seg.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "LesionDetectionConfig",
    "segment_lesions_hsv",
    "intersect_with_leaf",
    "refine_lesion_mask",
    "find_lesion_contours",
    "detect_lesions",
    "draw_lesion_contours",
    "LesionDetector",
]


@dataclass(frozen=True)
class LesionDetectionConfig:
    """Configuração padrão da etapa de detecção de lesões (F3.1).

    O canal H do OpenCV é circular (0-179). Tons "marrons/avermelhados"
    ficam perto das duas pontas do canal (próximo de 0 e próximo de 179),
    por isso são necessários DOIS ranges (BROWN_LOW e BROWN_HIGH) unidos
    com cv2.bitwise_or, em vez de um único range.
    """

    # Amarelo — lesões iniciais / clorose
    yellow_lower: Tuple[int, int, int] = (15, 40, 40)
    yellow_upper: Tuple[int, int, int] = (35, 255, 255)

    # Marrom — parte baixa do canal H (0-15)
    brown_low_lower: Tuple[int, int, int] = (0, 40, 20)
    brown_low_upper: Tuple[int, int, int] = (15, 255, 220)

    # Marrom — parte alta do canal H (160-179), pois H é circular
    brown_high_lower: Tuple[int, int, int] = (160, 40, 20)
    brown_high_upper: Tuple[int, int, int] = (179, 255, 220)

    # Morfologia (F3.4)
    aplicar_morph: bool = True
    morph_kernel_size: Tuple[int, int] = (3, 3)
    morph_iterations: int = 1

    # Contornos (F3.5)
    min_contour_area: float = 10.0


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


def _validate_mask(mask: Optional[np.ndarray]) -> bool:
    """Valida se é uma máscara binária válida (uint8, 2D)."""
    if mask is None:
        return False
    if not isinstance(mask, np.ndarray):
        return False
    if mask.ndim != 2:
        return False
    if mask.dtype != np.uint8:
        return False
    return True


def segment_lesions_hsv(
    hsv: Optional[np.ndarray],
    config: Optional[LesionDetectionConfig] = None,
) -> Optional[np.ndarray]:
    """Segmenta possíveis regiões de lesão no espaço HSV (F3.2).

    Combina três ranges (amarelo, marrom-baixo, marrom-alto) com
    cv2.bitwise_or, pois o canal H é circular e o marrom fica dividido
    entre as duas pontas do canal.

    Parâmetros
    - hsv: imagem em espaço HSV (3 canais, uint8)
    - config: LesionDetectionConfig com os ranges. Usa o padrão se None.

    Retorna `raw_lesion_mask` (uint8, 0 ou 255) ou None para input inválido.
    """
    if not _validate_hsv_image(hsv):
        return None

    cfg = config or LesionDetectionConfig()

    try:
        yellow_mask = cv2.inRange(
            hsv,
            np.array(cfg.yellow_lower, dtype=np.uint8),
            np.array(cfg.yellow_upper, dtype=np.uint8),
        )
        brown_low_mask = cv2.inRange(
            hsv,
            np.array(cfg.brown_low_lower, dtype=np.uint8),
            np.array(cfg.brown_low_upper, dtype=np.uint8),
        )
        brown_high_mask = cv2.inRange(
            hsv,
            np.array(cfg.brown_high_lower, dtype=np.uint8),
            np.array(cfg.brown_high_upper, dtype=np.uint8),
        )

        raw_lesion_mask = cv2.bitwise_or(yellow_mask, brown_low_mask)
        raw_lesion_mask = cv2.bitwise_or(raw_lesion_mask, brown_high_mask)

        return raw_lesion_mask

    except Exception:
        return None


def intersect_with_leaf(
    lesion_mask: Optional[np.ndarray],
    leaf_mask: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Restringe a máscara de lesão à área da folha (F3.3).

    Garante que nenhum pixel de lesão exista fora da folha, requisito
    obrigatório antes de qualquer cálculo de área/percentual (F4).

    Parâmetros
    - lesion_mask: máscara bruta de lesão (uint8, 0 ou 255)
    - leaf_mask: máscara final da folha (uint8, 0 ou 255)

    Retorna `lesion_inside_leaf_mask` ou None para input inválido.
    """
    if not _validate_mask(lesion_mask) or not _validate_mask(leaf_mask):
        return None

    if lesion_mask.shape != leaf_mask.shape:
        return None

    try:
        return cv2.bitwise_and(lesion_mask, leaf_mask)
    except Exception:
        return None


def refine_lesion_mask(
    mask: Optional[np.ndarray],
    kernel_size: Tuple[int, int] = (3, 3),
    iterations: int = 1,
) -> Optional[np.ndarray]:
    """Refinamento morfológico da máscara de lesão (F3.4).

    Aplica MORPH_OPEN (remove ruído pequeno) seguido de MORPH_CLOSE
    (fecha pequenos buracos), preservando lesões reais.

    Parâmetros
    - mask: máscara binária de lesão (uint8, 0 ou 255)
    - kernel_size: tamanho do kernel elíptico. Recomendado começar 3x3.
    - iterations: número de iterações das operações morfológicas.

    Retorna a máscara refinada ou None para input inválido.
    """
    if not _validate_mask(mask):
        return None

    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

        refined = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, iterations=iterations
        )
        refined = cv2.morphologyEx(
            refined, cv2.MORPH_CLOSE, kernel, iterations=iterations
        )

        return refined

    except Exception:
        return None


def find_lesion_contours(
    lesion_mask: Optional[np.ndarray],
    min_area: float = 10.0,
) -> Optional[List[np.ndarray]]:
    """Detecta contornos individuais de lesão (F3.5).

    Parâmetros
    - lesion_mask: máscara binária refinada de lesão (uint8, 0 ou 255)
    - min_area: área mínima (em pixels) para um contorno ser considerado
      uma lesão real, descartando ruído residual.

    Retorna lista de contornos (possivelmente vazia caso não haja
    lesões) ou None para input inválido.
    """
    if not _validate_mask(lesion_mask):
        return None

    try:
        contours, _ = cv2.findContours(
            lesion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        filtrados = [c for c in contours if cv2.contourArea(c) >= min_area]

        return filtrados

    except Exception:
        return None


def detect_lesions(
    hsv: Optional[np.ndarray],
    leaf_mask: Optional[np.ndarray],
    config: Optional[LesionDetectionConfig] = None,
) -> Optional[Tuple[np.ndarray, List[np.ndarray]]]:
    """Função central da Etapa 3: detecta lesões dentro da folha.

    Integra, em ordem, F3.2 -> F3.3 -> F3.4 -> F3.5:
    1. segment_lesions_hsv   — segmentação HSV multi-range
    2. intersect_with_leaf   — restringe à área da folha
    3. refine_lesion_mask    — limpeza morfológica
    4. find_lesion_contours  — contornos individuais filtrados por área

    Parâmetros
    - hsv: imagem HSV (uint8, 3 canais)
    - leaf_mask: máscara binária da folha (uint8, 0 ou 255)
    - config: LesionDetectionConfig. Usa o padrão se None.

    Retorna tupla (lesion_mask, contours) ou None para input inválido ou
    falha em qualquer etapa interna.
    """
    if not _validate_hsv_image(hsv) or not _validate_mask(leaf_mask):
        return None

    cfg = config or LesionDetectionConfig()

    raw_mask = segment_lesions_hsv(hsv, cfg)
    if raw_mask is None:
        return None

    inside_leaf_mask = intersect_with_leaf(raw_mask, leaf_mask)
    if inside_leaf_mask is None:
        return None

    if cfg.aplicar_morph:
        refined_mask = refine_lesion_mask(
            inside_leaf_mask,
            kernel_size=cfg.morph_kernel_size,
            iterations=cfg.morph_iterations,
        )
        if refined_mask is None:
            return None
    else:
        refined_mask = inside_leaf_mask

    contours = find_lesion_contours(refined_mask, min_area=cfg.min_contour_area)
    if contours is None:
        return None

    return (refined_mask, contours)


def draw_lesion_contours(
    image_bgr: Optional[np.ndarray],
    contours: Optional[List[np.ndarray]],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> Optional[np.ndarray]:
    """Desenha os contornos de lesão sobre a imagem original (F3.6).

    Parâmetros
    - image_bgr: imagem original ou pré-processada (BGR, uint8)
    - contours: lista de contornos das lesões (saída de find_lesion_contours
      ou detect_lesions)
    - color: cor BGR usada para desenhar os contornos (padrão vermelho)
    - thickness: espessura da linha do contorno

    Retorna uma cópia da imagem com os contornos desenhados, ou None para
    input inválido. Se `contours` for uma lista vazia, retorna uma cópia
    da imagem original sem desenhos (folha saudável não deve exibir
    contornos relevantes).
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        return None

    if contours is None:
        return None

    try:
        resultado = image_bgr.copy()

        if len(contours) > 0:
            cv2.drawContours(resultado, contours, -1, color, thickness)

        return resultado

    except Exception:
        return None


class LesionDetector:
    """Objeto responsável pela etapa 3 do pipeline."""

    def __init__(self, config: Optional[LesionDetectionConfig] = None) -> None:
        self.config = config or LesionDetectionConfig()

    def run(
        self,
        hsv: Optional[np.ndarray],
        leaf_mask: Optional[np.ndarray],
    ) -> Optional[Tuple[np.ndarray, List[np.ndarray]]]:
        """Executa a detecção completa de lesões (F3.2-F3.5)."""
        return detect_lesions(hsv, leaf_mask, config=self.config)

    def draw(
        self,
        image_bgr: Optional[np.ndarray],
        contours: Optional[List[np.ndarray]],
    ) -> Optional[np.ndarray]:
        """Desenha os contornos de lesão sobre a imagem (F3.6)."""
        return draw_lesion_contours(image_bgr, contours)
