# Detecção de Doenças em Folhas — Plano Compacto (UFMA / PI 2026)

**Objetivo:** pipeline Python+OpenCV (sem deep learning) que recebe imagem RGB de folha (PlantVillage, 256×256) e retorna % área afetada + severidade (Saudável<5% / Leve 5-20% / Moderada 20-50% / Grave>50%) + painel 4 subplots. Meta: ≥80% coerência com rótulos em teste visual.

**Escopo:** threshold HSV manual + morfologia p/ segmentar folha; detecção de lesões por range HSV; histograma canal H; validação em 150-300 imgs (tomate+batata, 3+ classes).
**Fora de escopo:** CNNs/transfer learning, identificação de espécie de doença, fundo não-uniforme, otimização automática de parâmetros.

**Prazo:** 05/mai a 06/jul/2026 (9 semanas, ~57h esperadas, ~65h com buffer).

---

## Arquitetura de Pastas
```
plant_disease/
├── data/selected/{tomato_healthy, tomato_blight, potato_early_blight}/
├── utils/
│   ├── io.py      → load_image(), load_batch(), save_result()
│   └── color.py   → to_rgb(), to_hsv(), to_gray()
├── stage1_preproc.py   → preprocess(img) -> (rgb, hsv, blurred)
├── stage2_leaf_seg.py  → segment_leaf(hsv) -> (mask, area_px)
├── stage3_lesion.py    → detect_lesions(hsv, leaf_mask) -> (mask, contours)
├── stage4_analysis.py  → analyze(leaf_mask, lesion_mask) -> (pct, severity, hist)
├── pipeline.py         → process_image(path) -> dict completo
├── notebooks/01_preproc, 02_leaf_seg, 03_lesion, 04_pipeline (.ipynb)
├── results/csv_results.csv  → image, leaf_px, lesion_px, pct, severity
└── requirements.txt
```
Convenções: imagens/máscaras em `np.uint8` (BGR/RGB/HSV; máscaras 0 ou 255); cada stage testável isolado; notebooks só importam os .py; commits frequentes.

---

## Fases e Subtarefas (F0–F6) — 44 itens
Formato: `ID — descrição [complexidade, horas esperadas]`

### F0 — Setup (Semana 1, até 11/05) 🚩BP1
- F0.1 — Criar/configurar venv Python [Baixa, 0.6h]
- F0.2 — Instalar dependências: opencv-python, numpy, matplotlib, jupyter [Baixa, 0.3h]
- F0.3 — Download e organização do dataset PlantVillage (tomate+batata) [Baixa, 2.0h]
- F0.4 — Selecionar subconjunto de imagens (50-100 img/classe, 3+ classes) [Baixa, 1.0h]
- F0.5 — Ler artigo Mohanty et al. 2016 (PlantVillage) [Baixa, 2.1h]
- F0.6 — Ler artigo Singh & Misra 2021 (pipeline HSV+GLCM) [Baixa, 2.1h]
- F0.7 — Definir arquitetura: pastas, módulos, interfaces (depende de F0.5+F0.6) [Média, 2.0h]
- F0.8 — Criar esqueleto do repositório e notebooks base (depende F0.7) [Baixa, 1.0h]
Saída: ambiente pronto + dataset local + doc de arquitetura.

### F1 — Pré-processamento (Semana 2, até 18/05) 🚩BP2
- F1.1 — Carregar imagens `cv2.imread()` single+batch (depende F0.3/F0.4/F0.8) [Baixa, 0.7h]
- F1.2 — Conversão BGR→RGB p/ visualização matplotlib (depende F1.1, paralelo) [Baixa, 0.3h]
- F1.3 — Conversão BGR→HSV, espaço principal (depende F1.1, paralelo) [Baixa, 0.3h]
- F1.4 — Suavização `cv2.GaussianBlur` kernel 5×5 (depende F1.1, paralelo) [Baixa, 0.5h]
- F1.5 — Redimensionamento p/ 256×256 `cv2.resize` (depende F1.1, paralelo) [Baixa, 0.3h]
- F1.6 — Notebook de pré-processamento com viz e comentários (depende F1.1-F1.5) [Baixa, 1.5h]
Critério pronto: HSV com H∈[0,179], S/V∈[0,255]; cores corretas no matplotlib; blur suaviza sem perder forma da folha.

### F2 — Segmentação da Folha (Semanas 3-4, até 01/06) 🚩BP3 [CRÍTICO]
- F2.1 — Segmentação HSV dos pixels verdes `cv2.inRange` (depende F1.3/F1.6) [Média, 2.0h]
- F2.2 — Limiarização de Otsu como alternativo (depende F1.1/F1.6, paralelo a F2.1) [Média, 1.0h]
- F2.3 — Morfologia: abertura e fechamento da máscara (depende F2.1) [Média, 1.6h]
- F2.4 — Extração do maior contorno `findContours`+sort área (depende F2.3) [Média, 1.6h]
- F2.5 — Aplicação de máscara `cv2.bitwise_and` (depende F2.4) [Baixa, 0.3h]
- F2.6 — ★ Ajuste fino dos ranges HSV via inspeção visual (depende F2.1-F2.5) [Alta, 3.2h — reservar 5h]
- F2.7 — Validação visual com 20+ imagens saudáveis+doentes (depende F2.6) [Média, 2.1h]
Critério pronto: máscara binária ≥80% acerto visual, sem buracos, sem ruído de fundo, parâmetros documentados.

### F3 — Detecção de Lesões (Semana 5, até 08/06) [CRÍTICO]
- F3.1 — ★ Definir ranges HSV p/ lesões: amarelo (H 15-35), marrom (H 0-15 e 160-179) (depende F2.7) [Alta, 2.6h]
- F3.2 — Segmentação de regiões doentes, multi-range `inRange` (depende F3.1) [Alta, 2.0h]
- F3.3 — Interseção lesão × máscara folha `bitwise_and` (depende F3.2/F2.5) [Média, 0.7h]
- F3.4 — Refinamento morfológico da máscara de lesão (depende F3.3) [Média, 1.5h]
- F3.5 — Detecção de contornos individuais de lesão (depende F3.4) [Média, 1.5h]
- F3.6 — Visualização: contornos sobre imagem original (depende F3.5) [Baixa, 0.7h]
Critério pronto: lesão sempre dentro da folha; folha saudável → lesão <1%; contornos corretos em doentes.

### F4 — Análise Quantitativa (Semana 6, até 15/06) 🚩BP4 [CRÍTICO]
- F4.1 — Contagem de pixels `np.sum` da máscara (depende F2.4/F3.4) [Baixa, 0.3h]
- F4.2 — Cálculo da % afetada = lesão/folha×100 (depende F4.1) [Baixa, 0.3h]
- F4.3 — Classificador de severidade por limiar fixo, 4 níveis (depende F4.2) [Baixa, 0.7h]
- F4.4 — Histograma canal H `calcHist`+normalização (depende F3.4/F2.5) [Média, 1.5h]
- F4.5 — Painel de visualização final, 4 subplots matplotlib (depende F4.4/F4.3) [Baixa, 1.6h]
- F4.6 — ★ Integrar os 4 estágios em pipeline end-to-end `process_image` (depende F4.1-F4.5) [Alta, 3.2h]
Critério pronto: saudável classifica <5%; histograma mostra pico verde (saudável) vs amarelo/marrom (doente); dict de retorno sem erros, testado em 10+ imgs.

### F5 — Validação (Semana 7, até 22/06) 🚩BP5 [CRÍTICO]
- F5.1 — Executar pipeline em batch com 50+ imagens (depende F4.6) [Média, 3.0h]
- F5.2 — ★ Analisar casos de falha: falsos positivos e negativos (depende F5.1) [Alta, 3.0h]
- F5.3 — Ajuste fino de parâmetros com base nos erros (depende F5.2, pode iterar com F5.1) [Alta, 3.7h]
- F5.4 — Gerar tabela de resultados consolidada (depende F5.1, paralelo a F5.2/F5.3) [Baixa, 1.5h]
- F5.5 — Documentar limitações e casos extremos (depende F5.2) [Baixa, 1.5h]

### F6 — Relatório e Entrega (Semanas 8-9, até 06/07) 🚩BP6 [CRÍTICO]
- F6.1 — Escrever seção de metodologia (depende F4.6, pode começar na S7) [Baixa, 3.0h]
- F6.2 — Inserir visualizações e tabela de resultados (depende F5.4) [Baixa, 1.5h]
- F6.3 — Escrever análise e discussão dos resultados (depende F5.2/F5.5) [Média, 3.0h]
- F6.4 — Limpeza final dos notebooks (depende F5.3) [Baixa, 1.5h]
- F6.5 — Revisão final do relatório e notebooks (depende F6.1-F6.4) [Baixa, 2.0h]
- F6.6 — Entrega final — 06/07/2026 (depende F6.5) [Baixa, 0.6h]

---

## Caminho Crítico
F0.5/F0.6 → F0.7 → F2.1 → **F2.6★** → **F3.1★** → F3.2 → **F4.6★** → F5.1/**F5.2★** → F6.3 → F6.5 → Entrega.
★ = maior risco de atraso.

Paralelizáveis: F1.2-F1.5 (após F1.1); F2.2 (Otsu) com F2.1; F5.4 com F5.2/F5.3; F6.1 pode começar na S7 junto com F5.x.

---

## Riscos Principais
| Risco | Mitigação |
|---|---|
| Ranges HSV não generalizam entre doenças | Testar classes distintas desde S4; union de ranges; documentar por doença |
| Falsos positivos (pixels saudáveis amarelos/marrons) | Sempre intersectar com máscara de folha; refinar lower bound H; threshold mínimo de contorno |
| Tuning HSV (F2.6) estoura tempo | Reservar 5h; se travar após 3h, avançar aproximado e refinar na S7 |
| Bugs de integração (F4.6) | Interfaces definidas em F0.7; testar módulos isolados antes |
| Dataset lento/indisponível | Kaggle mirror: kaggle.com/datasets/emmarex/plantdisease |

## Erros comuns a evitar
- OpenCV carrega em BGR — sempre converter antes de exibir/analisar em RGB.
- Nunca aplicar GaussianBlur depois de converter para HSV (aplicar em BGR antes).
- Nunca calcular área de lesão sem intersectar com máscara de folha.
- Canal H é circular (marrom perto de 0 e 179) — usar 2 `inRange` + `bitwise_or`.
- Kernels morfológicos grandes distorcem contorno — começar 5×5 ou 7×7.
- Não hardcodar paths — usar `os.path.join()`.

## Critérios finais de conclusão
`process_image()` roda sem erro em qualquer imagem do dataset; painel 4 subplots por imagem; CSV com 50+ resultados; notebooks limpos/executáveis; relatório entregue até 06/07/2026.
