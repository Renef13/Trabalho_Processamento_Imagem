"""Comparacao de correcoes HSV experimentais.

Este script nao altera o pipeline padrao. Ele executa fluxos alternativos
para comparar:

- baseline atual;
- histogram matching apenas em S/V;
- range HSV adicional para lesoes verde-oliva/acinzentadas;
- proxy de pixels nao verdes dentro da folha;
- combinacoes entre as abordagens.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plant_disease.config import RESULTS_DIR, SELECTED_DATA_DIR
from plant_disease.normalize import calculate_channel_cdf, match_sv_histogram
from plant_disease.pipeline import PipelineConfig, process_image
from plant_disease.stage1_preproc import preprocess
from plant_disease.stage2_leaf_seg import extract_largest_contour, segment_leaf
from plant_disease.stage3_lesion import (
    LesionDetectionConfig,
    find_lesion_contours,
    intersect_with_leaf,
    refine_lesion_mask,
    segment_lesions_hsv,
)
from plant_disease.stage4_analysis import analyze
from plant_disease.stage5_validation import (
    analyze_failures,
    collect_dataset_paths,
    is_healthy_category,
)
from plant_disease.utils.color import bgr_to_hsv
from plant_disease.utils.io import load_image

OUTPUT_DIR = RESULTS_DIR / "hsv_correcao"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    use_sv_matching: bool = False
    use_olive_range: bool = False
    use_nongreen_proxy: bool = False
    olive_lower: Tuple[int, int, int] = (30, 40, 20)
    olive_upper: Tuple[int, int, int] = (65, 255, 130)


EXPERIMENTS = (
    ExperimentConfig("baseline"),
    ExperimentConfig("sv_matching", use_sv_matching=True),
    ExperimentConfig("olive_range", use_olive_range=True),
    ExperimentConfig("nongreen_proxy", use_nongreen_proxy=True),
    ExperimentConfig(
        "sv_matching_plus_olive",
        use_sv_matching=True,
        use_olive_range=True,
    ),
    ExperimentConfig(
        "sv_matching_plus_nongreen",
        use_sv_matching=True,
        use_nongreen_proxy=True,
    ),
)


def _error_result(path: str, reason: str) -> Dict:
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
        "error": reason,
    }


def _load_hsv_and_leaf(path: str) -> Tuple[np.ndarray, np.ndarray] | None:
    image = load_image(path)
    image_pre = preprocess(image)
    hsv = bgr_to_hsv(image_pre)
    if hsv is None:
        return None

    leaf_result = segment_leaf(hsv)
    if leaf_result is None:
        return None

    contour_result = extract_largest_contour(leaf_result[0])
    if contour_result is None:
        return None

    return hsv, contour_result[0]


def build_reference_cdfs() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    s_values: List[np.ndarray] = []
    v_values: List[np.ndarray] = []

    for path, category in collect_dataset_paths(SELECTED_DATA_DIR):
        if category != "potato_early_blight":
            continue

        loaded = _load_hsv_and_leaf(path)
        if loaded is None:
            continue

        hsv, leaf_mask = loaded
        inside_leaf = leaf_mask > 0
        s_values.append(hsv[:, :, 1][inside_leaf])
        v_values.append(hsv[:, :, 2][inside_leaf])

    if not s_values or not v_values:
        return None

    s_cdf = calculate_channel_cdf(np.concatenate(s_values))
    v_cdf = calculate_channel_cdf(np.concatenate(v_values))
    if s_cdf is None or v_cdf is None:
        return None

    return s_cdf, v_cdf


def _extra_masks(hsv: np.ndarray, leaf_mask: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    extra = np.zeros(leaf_mask.shape, dtype=np.uint8)
    h, s, v = cv2.split(hsv)
    inside_leaf = leaf_mask > 0

    if config.use_olive_range:
        olive = cv2.inRange(
            hsv,
            np.array(config.olive_lower, dtype=np.uint8),
            np.array(config.olive_upper, dtype=np.uint8),
        )
        extra = cv2.bitwise_or(extra, olive)

    if config.use_nongreen_proxy:
        green_inside_leaf = (
            inside_leaf
            & (h >= 35)
            & (h <= 85)
            & (s >= 30)
            & (v >= 40)
        )
        proxy = np.zeros_like(leaf_mask)
        proxy[inside_leaf & (~green_inside_leaf)] = 255
        extra = cv2.bitwise_or(extra, proxy)

    combined = intersect_with_leaf(extra, leaf_mask)
    if combined is None:
        return np.zeros_like(leaf_mask)
    return combined


def process_image_experimental(
    path: str,
    config: ExperimentConfig,
    reference_cdfs: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Dict:
    if config.name == "baseline":
        return process_image(path, PipelineConfig(gerar_painel=False))

    loaded = _load_hsv_and_leaf(path)
    if loaded is None:
        return _error_result(path, "Falha na segmentacao da folha")

    hsv, leaf_mask = loaded
    hsv_for_lesion = hsv

    if config.use_sv_matching:
        if reference_cdfs is None:
            return _error_result(path, "CDF de referencia indisponivel")
        matched = match_sv_histogram(hsv, reference_cdfs=reference_cdfs)
        if matched is None:
            return _error_result(path, "Falha no histogram matching S/V")
        hsv_for_lesion = matched

    lesion_cfg = LesionDetectionConfig()
    raw_lesion = segment_lesions_hsv(hsv_for_lesion, lesion_cfg)
    if raw_lesion is None:
        return _error_result(path, "Falha na segmentacao HSV de lesao")

    extra = _extra_masks(hsv_for_lesion, leaf_mask, config)
    raw_lesion = cv2.bitwise_or(raw_lesion, extra)

    inside_leaf = intersect_with_leaf(raw_lesion, leaf_mask)
    if inside_leaf is None:
        return _error_result(path, "Falha na intersecao lesao-folha")

    refined = refine_lesion_mask(
        inside_leaf,
        kernel_size=lesion_cfg.morph_kernel_size,
        iterations=lesion_cfg.morph_iterations,
    )
    if refined is None:
        return _error_result(path, "Falha no refinamento de lesao")

    contours = find_lesion_contours(refined, min_area=lesion_cfg.min_contour_area)
    if contours is None:
        return _error_result(path, "Falha nos contornos de lesao")

    # Mantem a mascara coerente com o filtro de area minima.
    filtered_mask = np.zeros_like(refined)
    if contours:
        cv2.drawContours(filtered_mask, contours, -1, 255, -1)

    metrics = analyze(leaf_mask, filtered_mask, hsv=hsv_for_lesion)
    if metrics is None:
        return _error_result(path, "Falha na analise quantitativa")

    return {
        "image": str(path),
        "leaf_mask": leaf_mask,
        "lesion_mask": filtered_mask,
        "leaf_px": metrics["leaf_px"],
        "lesion_px": metrics["lesion_px"],
        "pct_affected": metrics["pct_affected"],
        "severity": metrics["severity"],
        "contours": contours,
        "panel": None,
        "error": None,
    }


def run_experiment(config: ExperimentConfig, reference_cdfs: Optional[Tuple[np.ndarray, np.ndarray]]) -> List[Dict]:
    results: List[Dict] = []
    for path, category in collect_dataset_paths(SELECTED_DATA_DIR):
        result = process_image_experimental(path, config, reference_cdfs)
        result["category"] = category
        result["is_healthy_expected"] = is_healthy_category(category)
        results.append(result)
    return results


def summarize_experiment(config_name: str, results: List[Dict]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for category in sorted({r["category"] for r in results}):
        category_results = [
            r for r in results if r["category"] == category and not r.get("error")
        ]
        if not category_results:
            rows.append(
                {
                    "experiment": config_name,
                    "category": category,
                    "n": 0,
                    "errors": sum(1 for r in results if r["category"] == category and r.get("error")),
                    "false_positive": 0,
                    "false_negative": 0,
                    "fp_rate": 0.0,
                    "fn_rate": 0.0,
                    "pct_mean": np.nan,
                    "leaf_px_mean": np.nan,
                }
            )
            continue

        healthy = is_healthy_category(category)
        false_positive = sum(r["severity"] != "Saudavel" for r in category_results) if healthy else 0
        false_negative = sum(r["severity"] == "Saudavel" for r in category_results) if not healthy else 0

        rows.append(
            {
                "experiment": config_name,
                "category": category,
                "n": len(category_results),
                "errors": sum(1 for r in results if r["category"] == category and r.get("error")),
                "false_positive": false_positive,
                "false_negative": false_negative,
                "fp_rate": false_positive / len(category_results) if healthy else 0.0,
                "fn_rate": false_negative / len(category_results) if not healthy else 0.0,
                "pct_mean": float(np.mean([r["pct_affected"] for r in category_results])),
                "leaf_px_mean": float(np.mean([r["leaf_px"] for r in category_results])),
            }
        )

    return pd.DataFrame(rows)


def save_comparison_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    disease = summary[summary["fn_rate"] > 0].copy()
    disease["label"] = disease["experiment"] + "\n" + disease["category"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for experiment in summary["experiment"].unique():
        sub = summary[(summary["experiment"] == experiment) & (summary["fn_rate"] > 0)]
        ax.plot(sub["category"], sub["fn_rate"], marker="o", label=experiment)

    ax.set_title("Taxa de falso negativo por categoria doente")
    ax.set_xlabel("Categoria")
    ax.set_ylabel("FN rate")
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "fn_rate_por_categoria.png", dpi=150)
    plt.close(fig)

    healthy = summary[summary["category"].str.contains("healthy")].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    for experiment in summary["experiment"].unique():
        sub = healthy[healthy["experiment"] == experiment]
        ax.plot(sub["category"], sub["fp_rate"], marker="o", label=experiment)

    ax.set_title("Taxa de falso positivo nas categorias saudaveis")
    ax.set_xlabel("Categoria")
    ax.set_ylabel("FP rate")
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "fp_rate_saudaveis.png", dpi=150)
    plt.close(fig)


def write_conclusion(overall: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    baseline = overall[overall["experiment"] == "baseline"].iloc[0]
    sv = overall[overall["experiment"] == "sv_matching"].iloc[0]
    olive = overall[overall["experiment"] == "olive_range"].iloc[0]
    nongreen = overall[overall["experiment"] == "nongreen_proxy"].iloc[0]
    sv_olive = overall[overall["experiment"] == "sv_matching_plus_olive"].iloc[0]
    sv_nongreen = overall[overall["experiment"] == "sv_matching_plus_nongreen"].iloc[0]

    lines = [
        "# Conclusao da comparacao de correcoes HSV",
        "",
        f"Baseline: FP={int(baseline.false_positives)}, FN={int(baseline.false_negatives)}, "
        f"FP rate={baseline.fp_rate:.3f}, FN rate={baseline.fn_rate:.3f}.",
        f"Matching S/V: FP={int(sv.false_positives)}, FN={int(sv.false_negatives)}, "
        f"FP rate={sv.fp_rate:.3f}, FN rate={sv.fn_rate:.3f}.",
        f"Range oliva H30-65, S>=40, V<=130: FP={int(olive.false_positives)}, "
        f"FN={int(olive.false_negatives)}, FP rate={olive.fp_rate:.3f}, "
        f"FN rate={olive.fn_rate:.3f}.",
        f"Proxy nao verde dentro da folha: FP={int(nongreen.false_positives)}, "
        f"FN={int(nongreen.false_negatives)}, FP rate={nongreen.fp_rate:.3f}, "
        f"FN rate={nongreen.fn_rate:.3f}.",
        f"Matching S/V + range oliva: FP={int(sv_olive.false_positives)}, "
        f"FN={int(sv_olive.false_negatives)}, FP rate={sv_olive.fp_rate:.3f}, "
        f"FN rate={sv_olive.fn_rate:.3f}.",
        f"Matching S/V + proxy nao verde: FP={int(sv_nongreen.false_positives)}, "
        f"FN={int(sv_nongreen.false_negatives)}, FP rate={sv_nongreen.fp_rate:.3f}, "
        f"FN rate={sv_nongreen.fn_rate:.3f}.",
        "",
        "O matching S/V isolado melhorou pouco os falsos negativos, porque preserva "
        "H e a principal falha esta em H fora dos ranges atuais. O range oliva "
        "atacou a causa em H e reduziu fortemente os falsos negativos, mas "
        "classificou quase todas as folhas saudaveis como doentes. A proxy nao "
        "verde e mais conservadora, mas o ganho em FN foi pequeno. As combinacoes "
        "nao resolveram o trade-off: a combinacao com oliva manteve FP muito alto, "
        "e a combinacao com proxy nao verde continuou com FN alto.",
        "",
        "Conclusao pratica: nao ha uma correcao HSV global segura para aplicar ao "
        "pipeline padrao sem aumentar muito os falsos positivos. A melhor proxima "
        "linha seria uma regra adaptativa por categoria/especie ou um discriminador "
        "adicional de textura/forma para separar verde-oliva saudavel de necrose.",
    ]
    (output_dir / "conclusao.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_cdfs = build_reference_cdfs()

    all_summary: List[pd.DataFrame] = []
    overall_rows: List[Dict[str, object]] = []

    for config in EXPERIMENTS:
        results = run_experiment(config, reference_cdfs)
        failures = analyze_failures(results)
        all_summary.append(summarize_experiment(config.name, results))
        overall_rows.append(
            {
                "experiment": config.name,
                "total": failures["total"],
                "errors": failures["n_erros"],
                "evaluated": failures["n_avaliados"],
                "false_positives": failures["n_false_positives"],
                "false_negatives": failures["n_false_negatives"],
                "fp_rate": failures["fp_rate"],
                "fn_rate": failures["fn_rate"],
            }
        )

    summary = pd.concat(all_summary, ignore_index=True)
    overall = pd.DataFrame(overall_rows)

    summary.to_csv(OUTPUT_DIR / "comparison_by_category.csv", index=False)
    overall.to_csv(OUTPUT_DIR / "comparison_overall.csv", index=False)
    save_comparison_plot(summary, OUTPUT_DIR)
    write_conclusion(overall, summary, OUTPUT_DIR)

    print(overall.round(4).to_string(index=False))
    print(summary.round(4).to_string(index=False))
    print(f"Resultados salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
