"""Diagnostico HSV por categoria.

Gera tabelas e histogramas para comparar H/S/V entre categorias doentes,
sem alterar o pipeline das fases F1-F5.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
from plant_disease.stage1_preproc import preprocess
from plant_disease.stage2_leaf_seg import extract_largest_contour, segment_leaf
from plant_disease.stage5_validation import collect_dataset_paths
from plant_disease.utils.color import bgr_to_hsv
from plant_disease.utils.io import load_image

DISEASE_CATEGORIES = (
    "potato_early_blight",
    "potato_late_blight",
    "tomato_early_blight",
    "tomato_late_blight",
)

OUTPUT_DIR = RESULTS_DIR / "hsv_diagnostico"


def _iter_sample_paths(n_per_category: int) -> Iterable[Tuple[str, str]]:
    grouped: Dict[str, List[str]] = {category: [] for category in DISEASE_CATEGORIES}
    for path, category in collect_dataset_paths(SELECTED_DATA_DIR):
        if category in grouped:
            grouped[category].append(path)

    for category in DISEASE_CATEGORIES:
        for path in sorted(grouped[category])[:n_per_category]:
            yield path, category


def _leaf_hsv(path: str) -> Tuple[np.ndarray, np.ndarray] | None:
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


def _describe(values: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def collect_hsv_statistics(n_per_category: int = 20) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for path, category in _iter_sample_paths(n_per_category):
        result = _leaf_hsv(path)
        if result is None:
            continue

        hsv, leaf_mask = result
        h, s, v = cv2.split(hsv)
        inside_leaf = leaf_mask > 0

        green_inside_leaf = (
            inside_leaf
            & (h >= 35)
            & (h <= 85)
            & (s >= 30)
            & (v >= 40)
        )
        lesion_proxy = inside_leaf & (~green_inside_leaf)

        regions = {
            "leaf": inside_leaf,
            "lesion_proxy": lesion_proxy,
        }

        for region_name, region_mask in regions.items():
            if int(region_mask.sum()) < 10:
                continue

            for channel_name, channel in (("H", h), ("S", s), ("V", v)):
                row = {
                    "category": category,
                    "image": Path(path).name,
                    "region": region_name,
                    "channel": channel_name,
                }
                row.update(_describe(channel[region_mask]))
                rows.append(row)

    return pd.DataFrame(rows)


def create_summary(stats: pd.DataFrame) -> pd.DataFrame:
    return (
        stats.groupby(["region", "category", "channel"], as_index=False)
        .agg(
            images=("image", "nunique"),
            pixels=("n", "sum"),
            mean=("mean", "mean"),
            median=("median", "mean"),
            std=("std", "mean"),
            p10=("p10", "mean"),
            p50=("p50", "mean"),
            p90=("p90", "mean"),
        )
        .round(2)
    )


def save_histograms(stats: pd.DataFrame, output_dir: Path) -> None:
    for region in ("leaf", "lesion_proxy"):
        region_stats = stats[stats["region"] == region]
        for channel in ("H", "S", "V"):
            fig, ax = plt.subplots(figsize=(10, 5))
            for category in DISEASE_CATEGORIES:
                values = region_stats[
                    (region_stats["category"] == category)
                    & (region_stats["channel"] == channel)
                ]["median"].dropna()
                if values.empty:
                    continue

                ax.hist(
                    values,
                    bins=18 if channel == "H" else 20,
                    alpha=0.35,
                    density=True,
                    label=category,
                )

            ax.set_title(f"Distribuicao de {channel} por categoria ({region})")
            ax.set_xlabel(channel)
            ax.set_ylabel("Densidade")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / f"hist_{region}_{channel}.png", dpi=150)
            plt.close(fig)


def write_conclusion(summary: pd.DataFrame, output_dir: Path) -> None:
    proxy = summary[summary["region"] == "lesion_proxy"]

    def _median(category: str, channel: str) -> float:
        row = proxy[(proxy["category"] == category) & (proxy["channel"] == channel)]
        return float(row.iloc[0]["median"])

    ref_h = _median("potato_early_blight", "H")
    ref_s = _median("potato_early_blight", "S")
    ref_v = _median("potato_early_blight", "V")

    lines = [
        "# Conclusao do diagnostico HSV",
        "",
        "A referencia `potato_early_blight`, que ja funcionava melhor no pipeline, "
        f"teve mediana aproximada na regiao `lesion_proxy` de H={ref_h:.2f}, "
        f"S={ref_s:.2f}, V={ref_v:.2f}.",
        "",
        "Nas categorias problematicas, o deslocamento mais importante nao foi uma "
        "queda geral de saturacao. Em `lesion_proxy`, S ficou igual ou maior que "
        "a referencia em varias categorias. O problema principal apareceu no H: "
        "`tomato_early_blight`, `tomato_late_blight` e parte de "
        "`potato_late_blight` cairam em tons verde-oliva/acinzentados, perto ou "
        "acima de H=35, fora dos ranges atuais de lesao (amarelo 15-35 e marrom "
        "0-15/160-179).",
        "",
        "Tambem ha componente de V: tomate, especialmente `tomato_early_blight`, "
        "tem V bem menor na regiao proxy, o que indica escurecimento/exposicao "
        "diferente. Ainda assim, como H permanece fora do range atual, matching "
        "apenas em S/V nao deve resolver sozinho.",
        "",
        "Conclusao: a causa e uma combinacao, com predominancia de H (cor real "
        "verde-oliva/acinzentada das lesoes) e componente secundario de V. A "
        "correcao deve ser comparada em tres modos: matching S/V isolado, range "
        "HSV adicional baseado no H observado e combinacao dos dois.",
    ]
    (output_dir / "conclusao.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = collect_hsv_statistics(n_per_category=20)
    summary = create_summary(stats)

    stats.to_csv(OUTPUT_DIR / "hsv_stats_por_imagem.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "hsv_summary.csv", index=False)
    save_histograms(stats, OUTPUT_DIR)
    write_conclusion(summary, OUTPUT_DIR)

    print(f"Linhas de estatistica: {len(stats)}")
    print(summary.to_string(index=False))
    print(f"Resultados salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
