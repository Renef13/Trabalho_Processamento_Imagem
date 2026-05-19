# 🌿 Plant Disease Detection — Classical Image Processing

Projeto acadêmico desenvolvido para a disciplina de **Processamento de Imagem** da
**Universidade Federal do Maranhão (UFMA)**, curso de Ciência da Computação — 2026.

---

## 📖 Sobre o Repositório

Este repositório contém o código-fonte, notebooks e resultados de um sistema de
**detecção e quantificação automática de doenças em folhas de plantas**, construído
inteiramente com técnicas clássicas de processamento de imagem.

O projeto nasceu de um problema real e relevante para o agronegócio: o diagnóstico
tradicional de doenças em plantas depende de inspeção visual humana, um processo
lento, subjetivo e sujeito a erros. Como alternativa, este sistema analisa imagens de
folhas e entrega, de forma automática e reprodutível, uma estimativa objetiva da
severidade da infecção.

**Por que técnicas clássicas?**
A abordagem escolhida usa exclusivamente operações de PI implementadas com OpenCV,
sem redes neurais, sem grandes volumes de dados rotulados. Isso torna o
sistema:
- **Interpretável:** cada decisão tem justificativa explícita no espaço de pixels
- **Acessível:** roda em qualquer CPU comum, inclusive no Google Colab gratuito
- **Transparente:** ideal para fins acadêmicos, onde entender *como* funciona
  importa tanto quanto *se* funciona

---

## 🎯 O que este sistema faz

Dada uma imagem de folha de planta, o pipeline executa
quatro estágios sequenciais e retorna:

1. **Máscara da folha** — isola a região foliar do fundo da imagem
2. **Máscara de lesões** — identifica os pixels correspondentes a tecido doente
3. **Porcentagem de área afetada** — métrica objetiva calculada em pixels
4. **Nível de severidade** — classificação automática em quatro categorias
5. **Painel visual** — imagem original + máscaras + histograma de cor, lado a lado

| Nível      | Área Afetada | Interpretação                                      |
|------------|--------------|----------------------------------------------------|
| Saudável   | < 5%         | Sem lesões significativas                          |
| Leve       | 5% – 20%     | Manchas isoladas, início de infecção               |
| Moderada   | 20% – 50%    | Infecção em progressão, intervenção recomendada    |
| Grave      | > 50%        | Grande parte da folha comprometida                 |

---

## 🧪 Dataset

As imagens utilizadas são do **PlantVillage** (Mohanty et al., 2016), um benchmark
público com mais de 54 mil imagens de folhas rotuladas por espécie e doença.

Para este projeto, foram selecionadas imagens de **tomate** e **batata**, cobrindo
classes saudáveis e pelo menos duas classes de doenças por espécie — totalizando
aproximadamente 150–300 imagens de teste.

> Dataset disponível em:
> [github.com/spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
> — mirror no Kaggle: [kaggle.com/datasets/emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

## 🧠 Tecnologias

| Ferramenta       | Função                              |
|------------------|-------------------------------------|
| Python 3.10+     | Linguagem principal                 |
| OpenCV 4.x       | Todas as operações de PI            |
| NumPy            | Operações sobre arrays de pixels    |
| Matplotlib       | Visualizações e histogramas         |
| Jupyter Notebook | Exploração interativa por estágio   |

O ambiente pode ser configurado localmente ou via
**Google Colab** (gratuito), que já possui todas as dependências instaladas.

---

## 📂 Estrutura do Projeto

```text
plant_disease/
├── data/
│   └── selected/               # Subconjunto curado do PlantVillage
│       ├── tomato_healthy/
│       ├── tomato_blight/
│       └── potato_early_blight/
│
├── utils/
│   ├── io.py                   # Carregamento e salvamento de imagens
│   └── color.py                # Conversões de espaço de cor
│
├── stage1_preproc.py           # Pré-processamento (BGR→HSV, blur, resize)
├── stage2_leaf_seg.py          # Segmentação da folha (HSV + morfologia)
├── stage3_lesion.py            # Detecção de lesões (multi-range HSV)
├── stage4_analysis.py          # Análise quantitativa e classificação
├── pipeline.py                 # Pipeline end-to-end: process_image(path)
│
├── notebooks/
│   ├── 01_preproc.ipynb        # Exploração do pré-processamento
│   ├── 02_leaf_seg.ipynb       # Tuning visual da segmentação da folha
│   ├── 03_lesion.ipynb         # Tuning das máscaras de doença
│   └── 04_pipeline.ipynb       # Execução em batch + tabela de resultados
│
├── results/
│   └── csv_results.csv         # image | leaf_px | lesion_px | pct | severity
│
├── requirements.txt
└── README.md
```

---

## 🚀 Como executar

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/plant-disease-detection.git
cd plant-disease-detection

# 2. Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute o pipeline em uma imagem
python pipeline.py --image data/selected/tomato_blight/img001.jpg
```

Ou explore os notebooks na pasta `notebooks/` para execução interativa estágio a estágio.

---

## 📚 Referências

- MOHANTY, S. P.; HUGHES, D. P.; SALATHE, M. *Using Deep Learning for Image-Based
  Plant Disease Detection.* Frontiers in Plant Science, 2016.
- SINGH, V.; MISRA, A. K. *Plant Disease Detection Using Image Processing and Machine
  Learning.* arXiv:2106.10698, 2021.

---

*UFMA — Ciência da Computação — Processamento de Imagem — 2026*
