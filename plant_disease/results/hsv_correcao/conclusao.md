# Conclusao da comparacao de correcoes HSV

Baseline: FP=1, FN=130, FP rate=0.010, FN rate=0.653.
Matching S/V: FP=2, FN=126, FP rate=0.020, FN rate=0.633.
Range oliva H30-65, S>=40, V<=130: FP=99, FN=3, FP rate=0.980, FN rate=0.015.
Proxy nao verde dentro da folha: FP=7, FN=112, FP rate=0.069, FN rate=0.563.
Matching S/V + range oliva: FP=96, FN=8, FP rate=0.950, FN rate=0.040.
Matching S/V + proxy nao verde: FP=2, FN=120, FP rate=0.020, FN rate=0.603.

O matching S/V isolado melhorou pouco os falsos negativos, porque preserva H e a principal falha esta em H fora dos ranges atuais. O range oliva atacou a causa em H e reduziu fortemente os falsos negativos, mas classificou quase todas as folhas saudaveis como doentes. A proxy nao verde e mais conservadora, mas o ganho em FN foi pequeno. As combinacoes nao resolveram o trade-off: a combinacao com oliva manteve FP muito alto, e a combinacao com proxy nao verde continuou com FN alto.

Conclusao pratica: nao ha uma correcao HSV global segura para aplicar ao pipeline padrao sem aumentar muito os falsos positivos. A melhor proxima linha seria uma regra adaptativa por categoria/especie ou um discriminador adicional de textura/forma para separar verde-oliva saudavel de necrose.