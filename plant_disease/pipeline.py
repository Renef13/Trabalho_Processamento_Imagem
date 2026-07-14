"""Pipeline de ponta a ponta — F4.6.

Integra as quatro etapas do projeto em uma única função de alto nível:

    process_image(path) -> dict

Fluxo (conforme especificado no plano):
1. Carregar imagem            — utils.io.load_image
2. Pré-processar              — stage1_preproc.preprocess
3. Converter para HSV         — utils.color.bgr_to_hsv
4. Segmentar folha            — stage2_leaf_seg.segment_leaf + extract_largest_contour
5. Detectar lesões            — stage3_lesion.detect_lesions
6. Analisar                   — stage4_analysis.analyze
7. Montar painel (opcional)   — stage4_analysis.create_result_panel
8. Retornar resultados completos

Decisão de design (documentada para a próxima fase, F5 — Validação):
`process_image` NUNCA retorna None. Mesmo quando a imagem não pode ser
carregada ou a folha não é detectada, a função retorna um dict com TODAS
as chaves esperadas (severity="Erro" e uma chave extra "error" com o
motivo). Isso permite que o processamento em lote da F5
(`process_batch`, ainda não implementado) simplesmente itere sobre os
resultados sem precisar tratar `None` como caso especial — alinhado com
o critério de pronto da F5.1: "não interrompe o batch por erro em uma
única imagem".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from matplotlib.figure import Figure

from plant_disease.utils.io import load_image
from plant_disease.utils.color import bgr_to_hsv, bgr_to_rgb
from plant_disease.stage1_preproc import preprocess, PreprocessConfig
from plant_disease.stage2_leaf_seg import (
    segment_leaf,
    extract_largest_contour,
    LeafSegmentationConfig,
)
from plant_disease.stage3_lesion import detect_lesions, LesionDetectionConfig
from plant_disease.stage4_analysis import (
    analyze,
    create_result_panel,
    SeverityThresholds,
)

__all__ = ["PipelineConfig", "process_image", "Pipeline"]

# Chaves sempre presentes no dict de retorno de process_image (F4.6).
_RESULT_KEYS = (
    "image",
    "leaf_mask",
    "lesion_mask",
    "leaf_px",
    "lesion_px",
    "pct_affected",
    "severity",
    "contours",
    "panel",
    "error",
)


@dataclass
class PipelineConfig:
    """Agrupa a configuração de cada etapa do pipeline em um só lugar.

    Isso mantém as interfaces entre módulos estáveis (requisito da
    F4.6): mudar um parâmetro de uma etapa não exige alterar a
    assinatura de process_image, apenas os defaults desta config.
    """

    preprocess_config: PreprocessConfig = field(default_factory=PreprocessConfig)
    leaf_config: LeafSegmentationConfig = field(default_factory=LeafSegmentationConfig)
    lesion_config: LesionDetectionConfig = field(default_factory=LesionDetectionConfig)
    severity_thresholds: SeverityThresholds = field(default_factory=SeverityThresholds)

    # Gerar o painel (F4.5) custa tempo/memória (uma Figure matplotlib
    # por imagem). Manter True por padrão para uso interativo/notebook;
    # a F5 (batch com 50-300 imagens) deve passar gerar_painel=False.
    gerar_painel: bool = True


def _resultado_com_erro(path: str, motivo: str) -> Dict:
    """Monta o dict de retorno padrão para qualquer falha do pipeline.

    Mantém as mesmas chaves de um resultado bem-sucedido (critério de
    pronto da F4.6), apenas com valores neutros e a chave "error"
    preenchida para diagnóstico (útil na F5.2 — análise de falhas).
    """
    return {
        "image": str(path),
        "leaf_mask": None,
        "lesion_mask": None,
        "leaf_px": 0,
        "lesion_px": 0,
        "pct_affected": 0.0,
        "severity": "Erro",
        "contours": [],
        "panel": None,
        "error": motivo,
    }


def process_image(
    path: str,
    config: Optional[PipelineConfig] = None,
) -> Dict:
    """Executa o pipeline completo (F1 -> F4) para uma única imagem.

    Parâmetros
    - path: caminho da imagem de entrada.
    - config: PipelineConfig agrupando as configurações de cada etapa.
      Usa os padrões de cada módulo (já validados em F2.6/F2.7 e F3)
      quando None.

    Retorna sempre um dict com as chaves:
    {
        "image": str,
        "leaf_mask": np.ndarray | None,
        "lesion_mask": np.ndarray | None,
        "leaf_px": int,
        "lesion_px": int,
        "pct_affected": float,
        "severity": str,
        "contours": list,
        "panel": matplotlib.figure.Figure | None,
        "error": str | None,
    }

    "error" é None em caso de sucesso total. Nunca lança exceção nem
    retorna None — ver nota de design no topo do módulo.
    """
    cfg = config or PipelineConfig()

    # 1. Carregar imagem
    img_bgr = load_image(str(path))
    if img_bgr is None:
        return _resultado_com_erro(path, f"Nao foi possivel carregar a imagem: {path}")

    # 2. Pré-processar (blur + resize conforme PreprocessConfig)
    pcfg = cfg.preprocess_config
    img_pre = preprocess(
        img_bgr,
        aplicar_blur=pcfg.aplicar_blur,
        aplicar_resize=pcfg.aplicar_resize,
        ksize=pcfg.ksize,
        sigmaX=pcfg.sigmaX,
        size=pcfg.size,
        interpolation=pcfg.interpolation,
    )
    if img_pre is None:
        return _resultado_com_erro(path, "Falha no pre-processamento (F1)")

    # 3. Converter para HSV (espaço principal de análise)
    hsv = bgr_to_hsv(img_pre)
    if hsv is None:
        return _resultado_com_erro(path, "Falha na conversao BGR->HSV")

    # 4. Segmentar folha (HSV + morfologia) e refinar com maior contorno
    lcfg = cfg.leaf_config
    leaf_result = segment_leaf(
        hsv,
        h_min=lcfg.h_min,
        h_max=lcfg.h_max,
        s_min=lcfg.s_min,
        s_max=lcfg.s_max,
        v_min=lcfg.v_min,
        v_max=lcfg.v_max,
        aplicar_morph=lcfg.aplicar_morph,
        morph_kernel_size=lcfg.morph_kernel_size,
        morph_iterations=lcfg.morph_iterations,
    )
    if leaf_result is None:
        return _resultado_com_erro(path, "Falha na segmentacao da folha (F2)")

    raw_leaf_mask, _raw_area = leaf_result

    contour_result = extract_largest_contour(raw_leaf_mask)
    if contour_result is None:
        # Folha nao detectada (nenhum contorno verde encontrado) - nao e
        # um crash, e um resultado valido de "sem folha na imagem".
        return _resultado_com_erro(path, "Folha nao detectada (nenhum contorno)")

    leaf_mask, _contour_area = contour_result

    # 5. Detectar lesões (já restritas à folha por construção, F3.3)
    lesion_result = detect_lesions(hsv, leaf_mask, config=cfg.lesion_config)
    if lesion_result is None:
        return _resultado_com_erro(path, "Falha na deteccao de lesoes (F3)")

    lesion_mask, contours = lesion_result

    # 6. Analisar (contagem, percentual, severidade, histograma)
    metrics = analyze(
        leaf_mask, lesion_mask, hsv=hsv, thresholds=cfg.severity_thresholds
    )
    if metrics is None:
        return _resultado_com_erro(path, "Falha na analise quantitativa (F4)")

    # 7. Painel de visualização (opcional — custa uma Figure por imagem)
    panel: Optional[Figure] = None
    if cfg.gerar_painel:
        image_rgb = bgr_to_rgb(img_pre)
        panel = create_result_panel(
            image_rgb, leaf_mask, lesion_mask, metrics["hist_h"], metrics
        )

    # 8. Resultado completo
    return {
        "image": str(path),
        "leaf_mask": leaf_mask,
        "lesion_mask": lesion_mask,
        "leaf_px": metrics["leaf_px"],
        "lesion_px": metrics["lesion_px"],
        "pct_affected": metrics["pct_affected"],
        "severity": metrics["severity"],
        "contours": contours,
        "panel": panel,
        "error": None,
    }


class Pipeline:
    """Objeto responsável pelo pipeline completo, no mesmo padrão de
    ImagePreprocessor / LeafSegmenter / LesionDetector / ResultAnalyzer.

    Útil quando se quer reaproveitar a mesma configuração para várias
    imagens sem repetir os parâmetros a cada chamada (ex.: F5 - batch).
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    def process(self, path: str) -> Dict:
        """Executa process_image com a configuração deste objeto."""
        return process_image(path, config=self.config)
