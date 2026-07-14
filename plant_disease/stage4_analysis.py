"""Rotinas de análise quantitativa (Etapa 4).

Implementações:

- F4.1: count_mask_pixels(mask) -> int
    Contagem de pixels brancos de uma máscara binária.
- F4.2: calculate_affected_percentage(leaf_px, lesion_px) -> float
    % afetada = lesao/folha x 100, sem divisao por zero.
- F4.3: classify_severity(pct_affected) -> str
    Classificador de severidade por limiar fixo, 4 niveis.
- F4.4: calculate_h_histogram(hsv, mask=None) -> np.ndarray
    Histograma do canal H (cv2.calcHist) normalizado.
- F4.5: create_result_panel(image_rgb, leaf_mask, lesion_mask, hist_h, metrics)
    Painel de visualizacao final com 4 subplots (matplotlib).

Função central: analyze(leaf_mask, lesion_mask, hsv=None) -> dict
Integra F4.1 -> F4.2 -> F4.3 -> F4.4 (opcional) em uma única chamada,
seguindo o mesmo padrão de validação estabelecido nas etapas 1-3
(stage1_preproc.py, stage2_leaf_seg.py, stage3_lesion.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

__all__ = [
    "SeverityThresholds",
    "count_mask_pixels",
    "calculate_affected_percentage",
    "classify_severity",
    "calculate_h_histogram",
    "create_result_panel",
    "analyze",
    "ResultAnalyzer",
]


@dataclass(frozen=True)
class SeverityThresholds:
    """Limiares fixos de classificação de severidade (F4.3).

    Regras (percentual de área foliar afetada por lesão):
    - < leve_min             -> "Saudavel"
    - [leve_min, moderada_min)  -> "Leve"
    - [moderada_min, grave_min) -> "Moderada"
    - >= grave_min           -> "Grave"

    Os valores padrão seguem exatamente o especificado no plano do
    projeto: Saudável<5% / Leve 5-20% / Moderada 20-50% / Grave>=50%.
    """

    leve_min: float = 5.0
    moderada_min: float = 20.0
    grave_min: float = 50.0


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


def count_mask_pixels(mask: Optional[np.ndarray]) -> int:
    """Conta pixels não-nulos (brancos) de uma máscara binária (F4.1).

    Parâmetros
    - mask: máscara binária (uint8, valores 0 ou 255)

    Retorna a contagem de pixels brancos como int. Retorna 0 para
    máscara inválida ou vazia — a função nunca lança exceção, pois é
    usada como base para todo o restante da análise (F4.2+).
    """
    if not _validate_mask(mask):
        return 0

    try:
        return int(np.sum(mask > 0))
    except Exception:
        return 0


def calculate_affected_percentage(leaf_px: int, lesion_px: int) -> float:
    """Calcula o percentual de área foliar afetada por lesão (F4.2).

    Parâmetros
    - leaf_px: total de pixels da folha (área de referência — NUNCA a
      área total da imagem, apenas usar a máscara da folha)
    - lesion_px: total de pixels classificados como lesão

    Retorna (lesion_px / leaf_px) * 100, sempre limitado ao intervalo
    [0, 100]. Retorna 0.0 quando leaf_px <= 0, evitando divisão por
    zero (folha não detectada).
    """
    try:
        leaf_px_i = int(leaf_px)
        lesion_px_i = int(lesion_px)
    except (TypeError, ValueError):
        return 0.0

    if leaf_px_i <= 0:
        return 0.0

    if lesion_px_i < 0:
        lesion_px_i = 0

    pct = (lesion_px_i / leaf_px_i) * 100.0

    return max(0.0, min(100.0, pct))


def classify_severity(
    pct_affected: float,
    thresholds: Optional[SeverityThresholds] = None,
) -> str:
    """Classifica a severidade da doença por limiar fixo, 4 níveis (F4.3).

    Regras (thresholds padrão, documentados em SeverityThresholds):
    - < 5%       -> "Saudavel"
    - 5% a <20%  -> "Leve"
    - 20% a <50% -> "Moderada"
    - >= 50%     -> "Grave"

    Parâmetros
    - pct_affected: percentual de área afetada (0-100)
    - thresholds: SeverityThresholds customizado. Usa o padrão se None.

    Retorna a classificação como string. A classificação é puramente
    determinística — mesma entrada sempre produz a mesma saída.
    """
    th = thresholds or SeverityThresholds()

    try:
        pct = float(pct_affected)
    except (TypeError, ValueError):
        pct = 0.0

    if pct < th.leve_min:
        return "Saudavel"
    if pct < th.moderada_min:
        return "Leve"
    if pct < th.grave_min:
        return "Moderada"
    return "Grave"


def calculate_h_histogram(
    hsv: Optional[np.ndarray],
    mask: Optional[np.ndarray] = None,
    bins: int = 180,
) -> Optional[np.ndarray]:
    """Calcula o histograma normalizado do canal H (Hue) (F4.4).

    Parâmetros
    - hsv: imagem em espaço HSV (3 canais, uint8)
    - mask: máscara opcional (uint8, 0 ou 255) restringindo a região
      analisada. Em analyze(), usamos a leaf_mask inteira (não apenas a
      lesão), pois o objetivo é comparar o pico verde (folha saudável)
      contra o pico amarelo/marrom (folha doente) dentro da mesma folha.
      Se None, calcula sobre a imagem inteira.
    - bins: número de bins do histograma (padrão 180 — um por valor
      possível de H no OpenCV, que vai de 0 a 179)

    Retorna um np.ndarray 1D normalizado (soma ≈ 1.0) ou None para
    input inválido.
    """
    if not _validate_hsv_image(hsv):
        return None

    if mask is not None and not _validate_mask(mask):
        return None

    try:
        h_channel = hsv[:, :, 0]

        hist = cv2.calcHist([h_channel], [0], mask, [bins], [0, 180])
        hist = hist.flatten()

        total = hist.sum()
        if total > 0:
            hist = hist / total

        return hist

    except Exception:
        return None


def create_result_panel(
    image_rgb: Optional[np.ndarray],
    leaf_mask: Optional[np.ndarray],
    lesion_mask: Optional[np.ndarray],
    hist_h: Optional[np.ndarray],
    metrics: Optional[Dict] = None,
) -> Optional[Figure]:
    """Cria o painel final de visualização com 4 subplots (F4.5).

    Painel:
    1. Imagem original (RGB).
    2. Máscara da folha (F2).
    3. Máscara de lesão (F3).
    4. Histograma do canal H — ou aviso textual se indisponível.

    Parâmetros
    - image_rgb: imagem original já convertida para RGB (usar
      bgr_to_rgb antes de chamar esta função; matplotlib espera RGB)
    - leaf_mask: máscara binária da folha (pode ser None)
    - lesion_mask: máscara binária da lesão (pode ser None)
    - hist_h: histograma do canal H, saída de calculate_h_histogram
      (pode ser None)
    - metrics: dict opcional com "pct_affected" e "severity" para
      compor o título do painel (tipicamente a saída de analyze())

    Retorna a Figure do matplotlib pronta, ou None para input
    inválido. A função NÃO chama plt.show() — quem chama decide se
    exibe (notebook) ou salva em disco (pipeline/batch).
    """
    if image_rgb is None or not isinstance(image_rgb, np.ndarray):
        return None

    metrics = metrics or {}
    pct = metrics.get("pct_affected")
    severity = metrics.get("severity", "N/A")

    try:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(image_rgb)
        axes[0].set_title("Imagem Original")
        axes[0].axis("off")

        if leaf_mask is not None:
            axes[1].imshow(leaf_mask, cmap="gray")
        axes[1].set_title("Máscara da Folha")
        axes[1].axis("off")

        if lesion_mask is not None:
            axes[2].imshow(lesion_mask, cmap="gray")
        axes[2].set_title("Máscara de Lesão")
        axes[2].axis("off")

        if hist_h is not None and len(hist_h) > 0:
            axes[3].plot(hist_h, color="darkgreen", linewidth=1.2)
            axes[3].fill_between(range(len(hist_h)), hist_h, alpha=0.2, color="darkgreen")
            axes[3].set_xlim([0, len(hist_h)])
            axes[3].set_title("Histograma Canal H")
            axes[3].set_xlabel("Hue (0-179)")
            axes[3].set_ylabel("Frequência normalizada")
        else:
            axes[3].axis("off")
            axes[3].text(
                0.5, 0.5, "Histograma\nindisponível",
                ha="center", va="center", fontsize=10, color="gray",
            )

        if isinstance(pct, (int, float)):
            titulo = f"Área afetada: {pct:.1f}%  —  Severidade: {severity}"
        else:
            titulo = f"Severidade: {severity}"

        fig.suptitle(titulo, fontsize=14, fontweight="bold")
        plt.tight_layout()

        return fig

    except Exception:
        plt.close("all")
        return None


def analyze(
    leaf_mask: Optional[np.ndarray],
    lesion_mask: Optional[np.ndarray],
    hsv: Optional[np.ndarray] = None,
    thresholds: Optional[SeverityThresholds] = None,
) -> Optional[Dict]:
    """Função central da Etapa 4: análise quantitativa completa.

    Integra, em ordem, F4.1 -> F4.2 -> F4.3 -> F4.4 (opcional):
    1. count_mask_pixels        — conta pixels da folha e da lesão
    2. calculate_affected_percentage — calcula % afetada
    3. classify_severity        — classifica em 4 níveis
    4. calculate_h_histogram    — histograma do canal H sobre a folha
       inteira (somente se `hsv` for fornecido)

    Parâmetros
    - leaf_mask: máscara binária da folha (uint8, 0 ou 255)
    - lesion_mask: máscara binária da lesão, já restrita à folha —
      espera-se a saída de stage3_lesion.detect_lesions, nunca a
      máscara bruta antes da interseção (F3.3)
    - hsv: imagem HSV opcional, usada apenas para o histograma. Se
      None, "hist_h" no retorno será None.
    - thresholds: SeverityThresholds customizado. Usa o padrão se None.

    Retorna dict:
    {
        "leaf_px": int,
        "lesion_px": int,
        "pct_affected": float,
        "severity": str,
        "hist_h": np.ndarray | None,
    }
    Ou None se leaf_mask/lesion_mask forem inválidas ou tiverem
    formatos incompatíveis entre si.
    """
    if not _validate_mask(leaf_mask) or not _validate_mask(lesion_mask):
        return None

    if leaf_mask.shape != lesion_mask.shape:
        return None

    leaf_px = count_mask_pixels(leaf_mask)
    lesion_px = count_mask_pixels(lesion_mask)

    pct_affected = calculate_affected_percentage(leaf_px, lesion_px)
    severity = classify_severity(pct_affected, thresholds=thresholds)

    hist_h: Optional[np.ndarray] = None
    if hsv is not None:
        hist_h = calculate_h_histogram(hsv, mask=leaf_mask)

    return {
        "leaf_px": leaf_px,
        "lesion_px": lesion_px,
        "pct_affected": pct_affected,
        "severity": severity,
        "hist_h": hist_h,
    }


class ResultAnalyzer:
    """Objeto responsável pela etapa 4 do pipeline."""

    def __init__(self, thresholds: Optional[SeverityThresholds] = None) -> None:
        self.thresholds = thresholds or SeverityThresholds()

    def run(
        self,
        leaf_mask: Optional[np.ndarray],
        lesion_mask: Optional[np.ndarray],
        hsv: Optional[np.ndarray] = None,
    ) -> Optional[Dict]:
        """Executa a análise quantitativa completa (F4.1-F4.4)."""
        return analyze(leaf_mask, lesion_mask, hsv=hsv, thresholds=self.thresholds)

    def panel(
        self,
        image_rgb: Optional[np.ndarray],
        leaf_mask: Optional[np.ndarray],
        lesion_mask: Optional[np.ndarray],
        hist_h: Optional[np.ndarray],
        metrics: Optional[Dict] = None,
    ) -> Optional[Figure]:
        """Cria o painel de visualização final (F4.5)."""
        return create_result_panel(image_rgb, leaf_mask, lesion_mask, hist_h, metrics)
