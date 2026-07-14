# Conclusao do diagnostico HSV

A referencia `potato_early_blight`, que ja funcionava melhor no pipeline, teve mediana aproximada na regiao `lesion_proxy` de H=29.11, S=78.58, V=125.34.

Nas categorias problematicas, o deslocamento mais importante nao foi uma queda geral de saturacao. Em `lesion_proxy`, S ficou igual ou maior que a referencia em varias categorias. O problema principal apareceu no H: `tomato_early_blight`, `tomato_late_blight` e parte de `potato_late_blight` cairam em tons verde-oliva/acinzentados, perto ou acima de H=35, fora dos ranges atuais de lesao (amarelo 15-35 e marrom 0-15/160-179).

Tambem ha componente de V: tomate, especialmente `tomato_early_blight`, tem V bem menor na regiao proxy, o que indica escurecimento/exposicao diferente. Ainda assim, como H permanece fora do range atual, matching apenas em S/V nao deve resolver sozinho.

Conclusao: a causa e uma combinacao, com predominancia de H (cor real verde-oliva/acinzentada das lesoes) e componente secundario de V. A correcao deve ser comparada em tres modos: matching S/V isolado, range HSV adicional baseado no H observado e combinacao dos dois.