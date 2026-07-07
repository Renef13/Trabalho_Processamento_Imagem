# Plano de Implementacao das Fases Restantes - Plant Disease

Projeto: **Deteccao e Quantificacao de Doencas em Folhas de Plantas**  
Contexto atual: projeto na **Fase F3 - Deteccao de Lesoes**  
Fases restantes: **F3, F4, F5 e F6**  
Prazo final: **06/07/2026**

---

## Estado Atual

As fases **F0 - Setup**, **F1 - Pre-processamento** e **F2 - Segmentacao da Folha** ja foram implementadas. A F2 possui validacao em `results/f2_validation_results.csv`, com segmentacao de folha aprovada visualmente nas imagens avaliadas.

A proxima etapa obrigatoria e concluir a **F3 - Deteccao de Lesoes**, pois todas as fases seguintes dependem da geracao correta da mascara de lesao (`lesion_mask`).

---

# F3 - Deteccao de Lesoes

Objetivo: identificar regioes doentes dentro da folha usando tecnicas classicas de processamento de imagem em HSV.

Modulo principal: `stage3_lesion.py`  
Notebook de apoio: `notebooks/03_lesion.ipynb`

Funcao central esperada:

```python
detect_lesions(hsv, leaf_mask) -> tuple[np.ndarray, list]
```

Entrada:

- `hsv`: imagem em HSV, `np.ndarray`, `uint8`, 3 canais.
- `leaf_mask`: mascara binaria da folha, `np.ndarray`, `uint8`, valores 0 ou 255.

Saida:

- `lesion_mask`: mascara binaria contendo apenas pixels classificados como lesao.
- `contours`: lista de contornos das lesoes detectadas.

Criterio geral de pronto da F3:

- A mascara de lesao deve estar sempre contida dentro da mascara da folha.
- Folhas saudaveis devem gerar area de lesao proxima de zero.
- Folhas doentes devem mostrar regioes lesionadas visualmente coerentes.
- O notebook `03_lesion.ipynb` deve demonstrar a etapa com imagens saudaveis e doentes.

---

## F3.1 - Definir ranges HSV para lesoes: amarelos (H 15-35) e marrons (H 0-15, 160-179)

Caminho critico: **sim**  
Depende de: **F2.7**

O que fazer:

Definir os intervalos HSV usados para detectar possiveis lesoes. Como o canal H do OpenCV e circular, os tons marrons/vermelhos proximos de 0 e 179 precisam ser tratados com dois ranges separados.

Sugestao inicial:

```python
YELLOW_LOWER = (15, 40, 40)
YELLOW_UPPER = (35, 255, 255)

BROWN_LOW_LOWER = (0, 40, 20)
BROWN_LOW_UPPER = (15, 255, 220)

BROWN_HIGH_LOWER = (160, 40, 20)
BROWN_HIGH_UPPER = (179, 255, 220)
```

Funcoes/modulos envolvidos:

- `stage3_lesion.py`
- possivel dataclass `LesionDetectionConfig`

Criterio de pronto:

- Ranges definidos em configuracao reutilizavel.
- Ranges documentados no codigo.
- Testados visualmente em pelo menos algumas imagens saudaveis e doentes.

Riscos:

- Os ranges HSV podem nao generalizar entre tomate e batata.
- Tons amarelados naturais da folha podem gerar falso positivo.

Mitigacao:

- Testar os ranges em classes diferentes.
- Sempre intersectar a lesao com `leaf_mask`.
- Documentar limitacoes e ajustar os thresholds apos validacao.

Erros comuns a evitar:

- Usar apenas um range para marrom/vermelho e esquecer que o canal H e circular.
- Detectar lesao fora da folha.
- Ajustar ranges olhando apenas uma classe.

---

## F3.2 - Implementar segmentacao de regioes doentes (cv2.inRange multi-range)

Caminho critico: **sim**  
Depende de: **F3.1**

O que fazer:

Implementar a geracao da mascara inicial de lesoes usando multiplos `cv2.inRange`, combinando amarelos e marrons com `cv2.bitwise_or`.

Funcao sugerida:

```python
segment_lesions_hsv(hsv, config=None) -> np.ndarray
```

Entrada:

- `hsv`: imagem HSV.

Saida:

- `raw_lesion_mask`: mascara binaria inicial das regioes suspeitas.

Criterio de pronto:

- A funcao retorna mascara `uint8`, 2D, com valores 0 ou 255.
- Lesoes visiveis em folhas doentes aparecem na mascara.
- Imagens saudaveis nao devem gerar mascara excessivamente preenchida.

Erros comuns a evitar:

- Misturar imagem BGR com HSV.
- Usar `matplotlib` para visualizar HSV diretamente sem conversao adequada.
- Nao validar tipo e formato da imagem.

---

## F3.3 - Implementar intersecao lesao x mascara folha (cv2.bitwise_and)

Caminho critico: **sim**  
Depende de: **F3.2** e **F2.5**

O que fazer:

Garantir que a mascara de lesao contenha apenas pixels dentro da folha.

Funcao sugerida:

```python
intersect_with_leaf(lesion_mask, leaf_mask) -> np.ndarray
```

Entrada:

- `lesion_mask`: mascara inicial de lesao.
- `leaf_mask`: mascara final da folha.

Saida:

- `lesion_inside_leaf_mask`: mascara de lesao restrita a area da folha.

Criterio de pronto:

- Nenhum pixel de lesao pode existir fora da folha.
- `cv2.countNonZero(cv2.bitwise_and(lesion_mask, cv2.bitwise_not(leaf_mask))) == 0`.

Risco:

- Calcular severidade futura com pixels fora da folha.

Mitigacao:

- Fazer a intersecao obrigatoriamente antes de qualquer calculo de area.

Erro comum a evitar:

- Nunca calcular percentual de lesao usando a mascara bruta antes da intersecao.

---

## F3.4 - Implementar refinamento morfologico da mascara de lesao

Caminho critico: nao  
Depende de: **F3.3**

O que fazer:

Aplicar operacoes morfologicas para remover ruidos pequenos e consolidar regioes lesionadas.

Funcao sugerida:

```python
refine_lesion_mask(mask, kernel_size=(3, 3), iterations=1) -> np.ndarray
```

Operacoes recomendadas:

- `cv2.MORPH_OPEN` para remover ruido.
- `cv2.MORPH_CLOSE` para fechar pequenos buracos.

Criterio de pronto:

- Pequenos ruidos isolados removidos.
- Lesoes reais preservadas.
- Mascara continua binaria.

Riscos:

- Kernel grande demais pode apagar lesoes pequenas.
- Kernel grande demais pode juntar regioes separadas indevidamente.

Mitigacao:

- Comecar com kernel 3x3.
- Testar 3x3, 5x5 e documentar escolha.

---

## F3.5 - Implementar deteccao de contornos individuais de lesao

Caminho critico: nao  
Depende de: **F3.4**

O que fazer:

Detectar contornos das regioes lesionadas para permitir visualizacao e contagem de componentes.

Funcao sugerida:

```python
find_lesion_contours(lesion_mask, min_area=10) -> list
```

Entrada:

- `lesion_mask`: mascara refinada.

Saida:

- Lista de contornos filtrados por area minima.

Criterio de pronto:

- Contornos pequenos demais sao descartados.
- Contornos retornados correspondem as manchas de lesao.
- A funcao nao quebra quando nao ha lesoes.

---

## F3.6 - Implementar visualizacao: contornos sobre imagem original

Caminho critico: nao  
Depende de: **F3.5**

O que fazer:

Criar funcao para desenhar os contornos sobre a imagem original.

Funcao sugerida:

```python
draw_lesion_contours(image_bgr, contours) -> np.ndarray
```

Entrada:

- `image_bgr`: imagem original ou pre-processada.
- `contours`: contornos das lesoes.

Saida:

- Imagem BGR com contornos desenhados.

Criterio de pronto:

- Contornos aparecem sobre as regioes doentes.
- Imagem saudavel nao exibe contornos relevantes.
- Visualizacao incluida em `03_lesion.ipynb`.

---

# F4 - Analise Quantitativa

Objetivo: calcular area afetada, severidade e visualizacoes finais.

Modulo principal: `stage4_analysis.py`  
Integracao: `pipeline.py`

Funcao central esperada:

```python
analyze(leaf_mask, lesion_mask, hsv=None) -> dict
```

Saida esperada:

```python
{
    "leaf_px": int,
    "lesion_px": int,
    "pct_affected": float,
    "severity": str,
    "hist_h": np.ndarray | None
}
```

---

## F4.1 - Implementar contagem de pixels (np.sum da mascara binaria)

Caminho critico: nao  
Depende de: **F2.4** e **F3.4**

Funcao sugerida:

```python
count_mask_pixels(mask) -> int
```

Criterio de pronto:

- Conta corretamente pixels brancos da mascara.
- Funciona para `uint8` com valores 0/255.
- Retorna 0 para mascara vazia.

---

## F4.2 - Implementar calculo de porcentagem afetada (lesao / folha x 100)

Caminho critico: nao  
Depende de: **F4.1**

Funcao sugerida:

```python
calculate_affected_percentage(leaf_px, lesion_px) -> float
```

Criterio de pronto:

- Retorna `(lesion_px / leaf_px) * 100`.
- Evita divisao por zero.
- Resultado limitado a faixa coerente de 0 a 100.

Erro comum a evitar:

- Usar area total da imagem em vez da area da folha.

---

## F4.3 - Implementar classificador de severidade por limiar fixo (4 niveis)

Caminho critico: nao  
Depende de: **F4.2**

Funcao sugerida:

```python
classify_severity(pct_affected) -> str
```

Regras:

- `< 5%`: `Saudavel`
- `5% a <20%`: `Leve`
- `20% a <50%`: `Moderada`
- `>=50%`: `Grave`

Criterio de pronto:

- Folhas saudaveis classificam como `Saudavel`.
- Classificacao e deterministica.
- Limiares documentados.

---

## F4.4 - Implementar histograma do canal H (cv2.calcHist + normalizacao)

Caminho critico: nao  
Depende de: **F3.4** e **F2.5**

Funcao sugerida:

```python
calculate_h_histogram(hsv, mask=None) -> np.ndarray
```

Criterio de pronto:

- Histograma calculado sobre a regiao da folha ou lesao.
- Histograma normalizado.
- Usado para analise visual das cores.

---

## F4.5 - Implementar painel de visualizacao final (4 subplots com matplotlib)

Caminho critico: nao  
Depende de: **F4.4** e **F4.3**

Funcao sugerida:

```python
create_result_panel(image_rgb, leaf_mask, lesion_mask, hist_h, metrics) -> matplotlib.figure.Figure
```

Painel esperado:

1. Imagem original.
2. Mascara da folha.
3. Mascara/contornos de lesao.
4. Histograma H ou resumo visual com severidade.

Criterio de pronto:

- Painel gerado sem erros.
- Titulos exibem percentual afetado e severidade.
- Usavel em notebook e pipeline.

---

## F4.6 - Integrar os 4 estagios em pipeline end-to-end (funcao process_image)

Caminho critico: **sim**  
Depende de: **F4.1-F4.5**

Modulo: `pipeline.py`

Funcao esperada:

```python
process_image(path) -> dict
```

Fluxo:

1. Carregar imagem com `load_image`.
2. Pre-processar com `preprocess`.
3. Converter para HSV com `bgr_to_hsv`.
4. Segmentar folha com `segment_leaf`.
5. Refinar folha com `extract_largest_contour`.
6. Detectar lesoes com `detect_lesions`.
7. Analisar com `analyze`.
8. Retornar resultados completos.

Saida esperada:

```python
{
    "image": str,
    "leaf_mask": np.ndarray,
    "lesion_mask": np.ndarray,
    "leaf_px": int,
    "lesion_px": int,
    "pct_affected": float,
    "severity": str,
    "contours": list,
    "panel": Figure | None
}
```

Criterio de pronto:

- `process_image(path)` executa sem erro para pelo menos 10 imagens.
- Retorna todas as chaves esperadas.
- Mascara de lesao sempre contida na folha.
- Interface entre modulos esta estavel.

Risco:

- Bugs de integracao por formatos diferentes entre funcoes.

Mitigacao:

- Validar shapes e dtypes em cada etapa.
- Testar modulos isoladamente antes do pipeline completo.

---

# F5 - Validacao

Objetivo: executar o pipeline em lote, analisar falhas e consolidar resultados.

---

## F5.1 - Executar pipeline em batch com 50+ imagens

Caminho critico: **sim**  
Depende de: **F4.6**

Funcao sugerida:

```python
process_batch(input_dir_or_paths) -> list[dict]
```

Criterio de pronto:

- Pipeline roda em pelo menos 50 imagens.
- Inclui saudaveis e doentes.
- Nao interrompe o batch por erro em uma unica imagem.

---

## F5.2 - Analisar casos de falha: falsos positivos e negativos

Caminho critico: **sim**  
Depende de: **F5.1**

O que fazer:

Revisar visualmente casos problematicos.

Criterios:

- Identificar saudaveis classificadas como doentes.
- Identificar doentes classificadas como saudaveis.
- Registrar padroes de erro.

Risco:

- Falsos positivos por tons amarelos/marrons naturais.

Mitigacao:

- Ajustar thresholds HSV.
- Usar area minima de contorno.
- Documentar limitacoes.

---

## F5.3 - Ajuste fino de parametros baseado nos erros observados

Caminho critico: nao  
Depende de: **F5.2**

O que fazer:

Ajustar ranges HSV, kernel morfologico e area minima de contorno.

Criterio de pronto:

- Parametros finais documentados.
- Melhora observavel nos casos de falha.
- Nao piora significativamente os casos saudaveis.

---

## F5.4 - Gerar tabela de resultados consolidada (por imagem)

Caminho critico: nao  
Depende de: **F5.1**

Arquivo esperado:

```text
results/csv_results.csv
```

Colunas esperadas:

```text
image, class, leaf_px, lesion_px, pct, severity
```

Criterio de pronto:

- CSV preenchido com 50+ imagens.
- Valores coerentes.
- Sem linhas vazias ou incompletas.

---

## F5.5 - Documentar limitacoes e casos extremos observados

Caminho critico: nao  
Depende de: **F5.2**

O que documentar:

- Falhas por variacao de iluminacao.
- Lesoes muito pequenas.
- Tons saudaveis confundidos com doenca.
- Dependencia de fundo uniforme.
- Limitacoes de HSV manual.

Criterio de pronto:

- Limitacoes prontas para entrar no relatorio final.
- Casos extremos associados a exemplos visuais quando possivel.

---

# F6 - Relatorio e Entrega

Objetivo: consolidar metodologia, resultados, discussao e preparar entrega final.

---

## F6.1 - Escrever secao de metodologia do relatorio

Caminho critico: nao  
Depende de: **F4.6**

Conteudo:

- Dataset usado.
- Pre-processamento.
- Segmentacao da folha.
- Deteccao de lesoes.
- Calculo de severidade.
- Validacao.

Criterio de pronto:

- Metodologia permite reproduzir o pipeline.
- Funcoes principais citadas.

---

## F6.2 - Inserir visualizacoes e tabela de resultados no relatorio

Caminho critico: nao  
Depende de: **F5.4**

Conteudo:

- Exemplos de imagens processadas.
- Mascaras de folha.
- Mascaras de lesao.
- Painel final.
- Tabela consolidada.

Criterio de pronto:

- Figuras legiveis.
- Tabela de resultados incluida.
- Exemplos cobrem saudaveis e doentes.

---

## F6.3 - Escrever analise e discussao dos resultados

Caminho critico: **sim**  
Depende de: **F5.2** e **F5.5**

Conteudo:

- Interpretacao dos resultados.
- Analise de falsos positivos e negativos.
- Pontos fortes do metodo.
- Limitacoes.
- Possiveis melhorias.

Criterio de pronto:

- Discussao baseada nos resultados reais.
- Limitacoes explicitamente reconhecidas.
- Relacao clara entre falhas e decisoes tecnicas.

---

## F6.4 - Limpeza final dos notebooks (comentarios, organizacao)

Caminho critico: nao  
Depende de: **F5.3**

Notebooks:

- `01_preproc.ipynb`
- `02_leaf_seg.ipynb`
- `03_lesion.ipynb`
- `04_pipeline.ipynb`

Criterio de pronto:

- Notebooks executaveis do inicio ao fim.
- Celulas desnecessarias removidas.
- Comentarios explicam cada etapa.
- Sem caminhos absolutos hardcoded.

---

## F6.5 - Revisao final do relatorio e notebooks

Caminho critico: **sim**  
Depende de: **F6.1-F6.4**

Criterio de pronto:

- Relatorio revisado.
- Notebooks revisados.
- CSV presente.
- Pipeline executavel.
- Nenhuma etapa critica pendente.

---

## F6.6 - Entrega final (06/07/2026)

Caminho critico: **sim**  
Depende de: **F6.5**

Criterio de pronto:

- Codigo final organizado.
- Notebooks finais disponiveis.
- Relatorio final pronto.
- Resultados em `results/csv_results.csv`.
- Entrega feita dentro do prazo.

---

# Caminho Critico Consolidado

As tarefas que nao podem atrasar sao:

```text
F3.1 -> F3.2 -> F4.6 -> F5.1 -> F5.2 -> F6.3 -> F6.5 -> F6.6
```

Tambem sao altamente sensiveis:

- F3.3, porque garante que a lesao esteja dentro da folha.
- F5.3, porque ajustes tardios podem afetar os resultados finais.

---

# Proxima Acao Recomendada

Comecar imediatamente por `stage3_lesion.py`, implementando nesta ordem:

1. `LesionDetectionConfig`
2. `segment_lesions_hsv(hsv, config)`
3. `intersect_with_leaf(lesion_mask, leaf_mask)`
4. `refine_lesion_mask(mask)`
5. `find_lesion_contours(mask)`
6. `detect_lesions(hsv, leaf_mask)`
7. `draw_lesion_contours(image_bgr, contours)`

Depois validar tudo em `03_lesion.ipynb` antes de avancar para F4.

A regra principal e: **nao iniciar F4 sem uma mascara de lesao funcional e visualmente validada**.
