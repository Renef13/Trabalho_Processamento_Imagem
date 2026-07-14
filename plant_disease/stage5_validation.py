"""Rotinas de validação em lote (Etapa 5).

Objetivo (conforme plano): executar o pipeline (F4.6) em lote, analisar
falsos positivos/negativos e consolidar os resultados em CSV.

Implementações:

- F5.1: process_batch(input_dir_or_paths) -> list[dict]
    Roda process_image() em várias imagens sem interromper o lote por
    erro em uma única imagem (aproveita que process_image nunca
    retorna None — apenas filtramos por "error").
- F5.2: analyze_failures(results) -> dict
    Compara a classe real (inferida do nome da pasta, ex.:
    "tomato_healthy" / "potato_late_blight") contra a severidade
    prevista, e separa falsos positivos (saudável marcada como doente)
    e falsos negativos (doente marcada como "Saudavel").
- F5.4: export_results_csv(results, output_path) -> bool
    Escreve results/csv_results.csv com as colunas
    image, class, leaf_px, lesion_px, pct, severity.

Mantém o mesmo padrão de validação/robustez das etapas 1-4: nenhuma
função lança exceção para cima; falhas são reportadas nos dicts de
retorno, nunca travam o processo de lote (critério de pronto da F5.1).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from plant_disease.config import RESULTS_DIR
from plant_disease.pipeline import PipelineConfig, process_image

__all__ = [
    "BatchConfig",
    "collect_dataset_paths",
    "is_healthy_category",
    "process_batch",
    "analyze_failures",
    "export_results_csv",
]

_DEFAULT_EXTENSIONS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)


@dataclass
class BatchConfig:
    """Configuração da execução em lote (F5.1).

    - pipeline_config: reaproveita a mesma PipelineConfig de F4.6, mas
      com `gerar_painel=False` por padrão — gerar uma Figure do
      matplotlib por imagem é caro e desnecessário para 50-300 imagens
      (o painel é gerado sob demanda apenas para os casos de falha,
      em F5.2).
    - extensions: extensões de imagem aceitas ao varrer diretórios.
    """

    pipeline_config: PipelineConfig = field(
        default_factory=lambda: PipelineConfig(gerar_painel=False)
    )
    extensions: Tuple[str, ...] = _DEFAULT_EXTENSIONS


def is_healthy_category(category: Optional[str]) -> bool:
    """Infere se uma categoria/pasta representa folha saudável.

    Convenção do dataset (ver 04_pipeline.ipynb): pastas nomeadas como
    "<especie>_healthy" para saudáveis e "<especie>_<doenca>" (ex.:
    "tomato_early_blight") para doentes.
    """
    if not category:
        return False
    return "healthy" in category.lower()


def collect_dataset_paths(
    input_dir: Union[str, Path],
    extensions: Sequence[str] = _DEFAULT_EXTENSIONS,
) -> List[Tuple[str, str]]:
    """Varre um diretório de dataset e retorna (path, category).

    `category` é o nome da subpasta imediata que contém a imagem (ex.:
    "tomato_healthy"), seguindo a mesma convenção usada em
    04_pipeline.ipynb. Funciona tanto para um diretório com subpastas
    por categoria quanto para imagens soltas na raiz (category=""
    nesse caso).

    Retorna lista ordenada de tuplas (path, category); lista vazia se
    o diretório não existir ou não houver imagens compatíveis.
    """
    root = Path(input_dir)
    resultados: List[Tuple[str, str]] = []

    if not root.exists() or not root.is_dir():
        return resultados

    exts = tuple(e.lower() for e in extensions)

    for arquivo in sorted(root.rglob("*")):
        if not arquivo.is_file():
            continue
        if arquivo.suffix.lower() not in exts:
            continue

        # Categoria = subpasta imediata dentro de input_dir, se houver;
        # caso a imagem esteja direto na raiz, usa "" (sem categoria).
        relative = arquivo.relative_to(root)
        category = relative.parts[0] if len(relative.parts) > 1 else ""

        resultados.append((str(arquivo), category))

    return resultados


def process_batch(
    input_dir_or_paths: Union[str, Path, Sequence[Union[str, Tuple[str, str]]]],
    config: Optional[BatchConfig] = None,
) -> List[Dict]:
    """Executa process_image() em lote (F5.1).

    Parâmetros
    - input_dir_or_paths: um dos seguintes:
        * caminho de diretório (str/Path) — varrido recursivamente com
          `collect_dataset_paths` (categoria = subpasta imediata);
        * lista de paths (str) — categoria inferida do nome da pasta
          pai de cada arquivo;
        * lista de tuplas (path, category) — categoria usada como
          fornecida (útil para reaproveitar `demo_paths` do notebook
          04_pipeline.ipynb).
    - config: BatchConfig. Usa o padrão (`gerar_painel=False`) se None.

    Retorna uma lista de dicts, um por imagem, com todas as chaves de
    `process_image` MAIS duas chaves extras:
    - "category": a categoria/pasta de origem (string, pode ser "").
    - "is_healthy_expected": bool, rótulo esperado inferido de
      `category` via `is_healthy_category` (usado em F5.2).

    Critério de pronto da F5.1: nunca interrompe o lote por erro em
    uma única imagem — cada chamada de `process_image` já é
    "à prova de exceção" (nunca retorna None, nunca lança), então o
    laço simplesmente acumula os resultados, erro ou não.
    """
    cfg = config or BatchConfig()

    itens: List[Tuple[str, str]]

    if isinstance(input_dir_or_paths, (str, Path)) and Path(input_dir_or_paths).is_dir():
        itens = collect_dataset_paths(input_dir_or_paths, extensions=cfg.extensions)
    else:
        itens = []
        for item in input_dir_or_paths:  # type: ignore[union-attr]
            if isinstance(item, tuple):
                path, category = item
            else:
                path = item
                category = Path(path).parent.name
            itens.append((str(path), category))

    resultados: List[Dict] = []

    for path, category in itens:
        try:
            resultado = process_image(path, config=cfg.pipeline_config)
        except Exception as exc:  # nunca deve acontecer (process_image
            # já captura tudo internamente), mas o lote não pode parar
            # de jeito nenhum por causa de uma imagem problemática.
            resultado = {
                "image": str(path),
                "leaf_mask": None,
                "lesion_mask": None,
                "leaf_px": 0,
                "lesion_px": 0,
                "pct_affected": 0.0,
                "severity": "Erro",
                "contours": [],
                "panel": None,
                "error": f"Excecao nao tratada no lote: {exc}",
            }

        resultado["category"] = category
        resultado["is_healthy_expected"] = is_healthy_category(category)

        resultados.append(resultado)

    return resultados


def analyze_failures(results: List[Dict]) -> Dict:
    """Analisa falsos positivos e falsos negativos do lote (F5.2).

    Definições (com base na categoria/pasta de origem como "verdade"):
    - Falso positivo: imagem de categoria saudável (`is_healthy_expected
      == True`) cuja `severity` prevista NÃO é "Saudavel" (o pipeline
      "viu" doença onde não há).
    - Falso negativo: imagem de categoria doente (`is_healthy_expected
      == False`) cuja `severity` prevista É "Saudavel" (o pipeline
      "não viu" a doença).
    Imagens com `error` != None são contadas à parte (não entram em
    FP/FN, pois não há severidade confiável para comparar).

    Parâmetros
    - results: saída de `process_batch`.

    Retorna dict:
    {
        "total": int,
        "n_erros": int,
        "n_avaliados": int,               # total - n_erros
        "false_positives": list[dict],    # {"image","category","pct_affected","severity"}
        "false_negatives": list[dict],
        "n_false_positives": int,
        "n_false_negatives": int,
        "fp_rate": float,   # sobre saudaveis avaliadas
        "fn_rate": float,   # sobre doentes avaliadas
    }
    """
    total = len(results)
    erros = [r for r in results if r.get("error")]
    avaliados = [r for r in results if not r.get("error")]

    saudaveis_esperadas = [r for r in avaliados if r.get("is_healthy_expected")]
    doentes_esperadas = [r for r in avaliados if not r.get("is_healthy_expected")]

    false_positives = [
        {
            "image": Path(r["image"]).name,
            "category": r.get("category", ""),
            "pct_affected": r["pct_affected"],
            "severity": r["severity"],
        }
        for r in saudaveis_esperadas
        if r["severity"] != "Saudavel"
    ]

    false_negatives = [
        {
            "image": Path(r["image"]).name,
            "category": r.get("category", ""),
            "pct_affected": r["pct_affected"],
            "severity": r["severity"],
        }
        for r in doentes_esperadas
        if r["severity"] == "Saudavel"
    ]

    fp_rate = (
        len(false_positives) / len(saudaveis_esperadas)
        if saudaveis_esperadas
        else 0.0
    )
    fn_rate = (
        len(false_negatives) / len(doentes_esperadas)
        if doentes_esperadas
        else 0.0
    )

    return {
        "total": total,
        "n_erros": len(erros),
        "n_avaliados": len(avaliados),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "n_false_positives": len(false_positives),
        "n_false_negatives": len(false_negatives),
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
    }


def export_results_csv(
    results: List[Dict],
    output_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Exporta a tabela consolidada de resultados (F5.4).

    Escreve `results/csv_results.csv` (ou `output_path`, se fornecido)
    com exatamente as colunas pedidas no plano:

        image, class, leaf_px, lesion_px, pct, severity

    - "image": nome do arquivo (sem o caminho completo).
    - "class": a categoria/pasta de origem (ex.: "tomato_healthy").
    - "pct": percentual de área afetada, arredondado a 2 casas.
    - Imagens com erro ainda geram uma linha (leaf_px/lesion_px/pct
      vazios, severity="Erro") — nenhuma linha é omitida, conforme o
      critério de pronto "sem linhas vazias ou incompletas" (a linha
      existe e é consistente, só os valores numéricos ficam em branco
      quando não há medição válida).

    Parâmetros
    - results: saída de `process_batch`.
    - output_path: caminho de destino. Usa `RESULTS_DIR/csv_results.csv`
      se None.

    Retorna True em caso de sucesso, False em caso de falha de I/O.
    """
    destino = Path(output_path) if output_path is not None else RESULTS_DIR / "csv_results.csv"

    try:
        os.makedirs(destino.parent, exist_ok=True)

        with open(destino, "w", newline="", encoding="utf-8") as arquivo_csv:
            writer = csv.writer(arquivo_csv)
            writer.writerow(["image", "class", "leaf_px", "lesion_px", "pct", "severity"])

            for r in results:
                nome_imagem = Path(r["image"]).name
                categoria = r.get("category", "")

                if r.get("error"):
                    writer.writerow([nome_imagem, categoria, "", "", "", r["severity"]])
                    continue

                writer.writerow(
                    [
                        nome_imagem,
                        categoria,
                        r["leaf_px"],
                        r["lesion_px"],
                        round(r["pct_affected"], 2),
                        r["severity"],
                    ]
                )

        return True

    except Exception:
        return False
