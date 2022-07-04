
# `Estudo de caso: Geração de áudio musical através de GAN, LSTM e Transformer`
# `Case study: Generation of musical audio through GAN,   LSTM and Transformer` 

## Apresentação:

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376L - Deep Learning aplicado a Síntese de Sinais*, oferecida no primeiro semestre de 2022, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).</p>

## Equipe:

> |Nome  | RA | Especialização|
> |--|--|--|
> | Gabriel Santos Martins Dias  | 172441  | Eng. Eletricista |
> | Gleyson Roberto do Nascimento  | 043801  | Eng. Eletricista |
> | Patrick Carvalho Tavares R. Ferreira  | 175480  | Eng. Eletricista |

## Resumo (Abstract)

>Este projeto visa realizar a síntese de áudio voltado para música. Para isso, iremos reproduzir os resultados encontrados no artigo GANSynth, além de realizar um estudo de caso acerca de diferentes possibilidades de arquiteturas para geração de áudio, como LSTM e Transformer.  
>Utilizaremos métricas de avaliação que possam analisar a fidelidade dos dados reais e sintetizados, considerando Music Information Retrieval e também analisando a resposta de forma tabular.
>Durante esta etapa, obtivemos êxito na síntese através da LSTM, o Transformer ainda está em treinamento, mas já foi obtido um bom resultado no último checkpoint de treinamento, por fim, a GANSynth também está em treinamento e não obteve ainda resultados significativos no checkpoint de treinamento.

## Vídeo de apresentação da E3:
>| Vídeo de Apresentação E3  | PDF Apresentação E3 | 
> |--|--|
> | [![video](https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/capa.jpg)](https://www.youtube.com/watch?v=axYGN9mMJGE) | [![pdf](https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/capa.jpg)](https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/Entrega%20E3.pdf)  |

## Descrição do Problema/Motivação

>Grandes avanços no estado da arte de síntese de áudio foram iniciados quase exclusivamente por modelos autorregressivos, como WaveNet. No entanto, essa rede perde em termos de coerência global do áudio gerado, além de ter baixa taxa de amostragem, devido ao processo iterativo utilizado.
>
>Mesmo redes neurais com forte coerência local, como as redes convolucionais, têm dificuldade em realizar a modelagem de áudio, já que as múltiplas frequências que compõem as amostras não coincidem com o stride utilizado nestas camadas, gerando batimento que aumenta o erro de reprodução em fase, conforme estendemos a geração.
>
<p align="center">
	<img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/F01.png" height="280">
</p>
<p align="center">
Fonte: Referência Bibliográfica [1].
</p>

>Este é um desafio para uma rede de síntese, pois ela deve aprender todas as combinações apropriadas de frequência/fase e ativá-las na combinação certa para produzir uma forma de onda coerente. 
>
>Além disso, para um contexto específico como a música, com regras e teoria bastante consolidadas, a síntese resultante estaria de acordo com o que é estabelecido com música?

## Objetivo

>De forma geral, o projeto visa fazer um estudo de caso do comportamento da geração de áudio musical por três tipos de arquiteturas distintas: GAN (GANSynth), LSTM e Transformer, de forma a compreender melhor as vantagens, desvantagens, limitações de cada arquitetura e, principalmente, verificar se a resposta do áudio gerado pode de fato ser qualificado enquanto música através da análise do seu Music Information Retrieval.
>
>De forma específica, temos o seguinte objetivo:
>
>**GAN:**  com um tensor de rótulos e um tensor de ruído, gerar áudio musical (wav) com diferentes timbres;
>
>**LSTM:** com um arquivo de entrada (midi), gerar uma composição musical polifônica (midi) dentro das características musicais do arquivo de entrada;
>
>**Transformer:** com um arquivo de entrada (midi), gerar uma composição musical com melodia e harmonia (midi) dentro das características musicais do arquivo de entrada.


## Metodologia Proposta

>**Metodologia para a GAN:**
>>
>>Neste caso, a rede será construída layer by layer seguindo a ref. [1], de forma que, diferentes arquiteturas podem ocupar o lugar do gerador e do discriminador, mas o modelo geral é apresentado na figura a seguir. Este tipo de estrutura é classificado como uma CGAN (as GANs cuja classe de saída é controlável).

>> <p align="center">
>> <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/FIG02.png" height="500">
>> </p>
>> <p align="center">
>> Fonte: Referência Bibliográfica [1].
>> </p>

>>Temos como entrada um tensor de ruído cuja rede geradora mapeia um hiperespaço de representação de áudios em tempo de treinamento, além de classes de timbres e notas em one-hot encoding, as quais definem o tipo de saída desejada.
>>
>>Um áudio de mesma categoria (timbre e nota) é inserido para o discriminador posteriormente, o qual serve de comparação para a rede discriminadora. Seu objetivo é classificar o áudio real como verídico, enquanto o áudio sintético deve ser dado como falso. 
>>
>>Aprender a reconhecer um áudio como falso é uma tarefa consideravelmente mais simples do que aprender a gerar um áudio de boa qualidade. Essa discrepância de dificuldades entre gerador e discriminador é motivo pelo qual muitos projetistas optam por uma estratégia baseada em deixar o discriminador aprender ao longo dos primeiros mini-batches e congelar seus pesos em seguida, permitindo que o gerador aprenda o mapeamento.
>>
>>Fazemos isso através de dois escalonadores de Learning Rate, um para o gerador, e outro para o discriminador.
>>
>>O discriminador utiliza o cosine annealing como escalonador, de forma que seu step de gradiente varie entre valores adequados ao treinamento e valores tão pequenos que praticamente o impeçam de variar, de forma a aguardar o desenvolvimento do gerador neste meio tempo.
>>
>>O escalonador do gerador, por outro lado, é baseado em MultiStep, começando com um valor elevado, que favorece a fuga de mínimos locais ruins, e decaindo para valores menores, que proporcionam ajuste fino de parâmetros.
>>
>>Para treino, teste e validação será utilizado o [Nsynth Dataset](https://app.activeloop.ai/activeloop/nsynth-train) que contém 300.000 notas musicais de 1.000 instrumentos diferentes alinhados e gravados isoladamente. Cada amostra tem quatro segundos de duração e é amostrada em 16kHz, dando 64.000 dimensões.

>**Metodologia para o LSTM:**
>>
>>Para este caso, a rede da ref. [2] será utilizada de forma pré-treinada sem que haja a necessidade da construção layer by layer, uma vez que a experiência do grupo na tratativa com a GANSynth demonstrou ser bastante desafiadora. Assim, considerando que para esta arquitetura buscaremos uma geração do tipo composição musical, será utilizado o [Pop MIDI Dataset](https://drive.google.com/uc?id=1WcOkwcFfIoELXE4ueyVS2QlCuywAd5K6), composto por 1000 arquivos midi com músicas polifônicas com diferentes gêneros musicais, para treinamento com um batch size 64, learning rate 0,001, layers (128,128,128), 20000 passos e presença de dropout, uma vez que o uso direto do LSTM não trouxe resultados positivos.
><p align="center">
><img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/lstm.png" height="600">
></p>
><p align="center">
>Fonte: Referência Bibliográfica [2].
></p>

>*Metodologia para o Transformer:*
>>
>>Para este caso, a rede da ref. [3] também será utilizada de forma pré-treinada, contudo, diferente do LSTM, será utilizada uma fração do [Maestro Dataset](https://magenta.tensorflow.org/datasets/maestro#dataset), originalmente composto por mais 30000 arquivos de áudios musicais em piano para treinamento local e com algoritmo de atenção, para que a resposta possa ser mais otimizada.
><p align="center">
><img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/transformer.png" height="400">
></p>
><p align="center">
>Fonte: Referência Bibliográfica [3].
></p>

>**Metodologia de avaliação:**
>>
>> Para a avaliação comparativa entre os dados reais e sintetizados contará com três abordagens: quantitativa, qualitativa e MIR, desta forma, teremos:
>>
>>* **Análise Quantitativa:** considerando-se que a entrada e a saída do áudio serão dados do tipo *numpy array*, através da ferramenta Table Evaluate, será analisado o KS Teste, a fim de determinar a distância entre o que foi gerado e o real resultando no *score* entre eles, além disso, também será realizada uma pesquisa sensorial com os alunos da disciplina IA376L para buscar quantificar a percepção acerca dos áudios gerados;
>>
>> * **Análise Qualitativa:** gráfico log-log da média dos dados reais x média dos dados sintetizados (para verificar tendência de 45º); do gráfico log-log do desvio-padrão dos dados reais x desvio-padrão dos dados sintetizados (para verificar tendência de 45°); gráfico da soma cumulativa dos dados reais e sintetizados (para verificar diferença de distribuição), histograma dos dados reais e sintetizados (para verificar diferença de distribuição);
>>
>> * **Music Information Retrieval:** conforme a ref. [4], através das análises espectrais existentes na Biblioteca Librosa, fazer o comparativo entre as *features* espectrais do áudio real e do áudio sintetizado. Serão analisadas as seguintes *features* espectrais: 
>>
>>> a) STFT:  a transformada de Fourier de tempo curto é uma transformada integral derivada da transformada de Fourier, sendo uma forma de representação tempo-frequência;
>>>
>>>b) Espectrograma: pode ser definido como um gráfico que mostra a intensidade por meio do escurecimento ou coloração do traçado, as faixas de frequência no eixo vertical e o tempo no eixo horizontal. Sua representação mostra estrias horizontais, denominadas harmônicos;
>>> 
>>>c) Mel espectrograma: é um espectrograma onde as frequências são convertidas para a escala mel, isto é, uma escala onde a unidade de altura é tal que distâncias iguais em altura soam igualmente distantes para o ouvinte;
>>> 
>>> d) Cromagrama: é o gráfico do vetor de características tipicamente de 12 elementos que indica quanta energia de cada classe de altura, {C, C#, D, D#, E, ..., B}, é presente no sinal;
>>> 
>>> e) MFCC: Os coeficientes cepstral de frequência Mel (MFCCs) são coeficientes que coletivamente compõem uma representação do espectro de potência de curto prazo de um som, com base em uma transformação de cosseno linear de um espectro de potência de log em uma escala de frequência mel não linear;
>>> 
>>> f) RMS: potência real (Root Mean Squared) do sinal;
>>> 
>>> g) Espectral *bandwith*: é a diferença entre as frequências superior e inferior em uma banda contínua de frequências;
>>> 
>>> h) Espectral *rolloff*: Pode ser definido como a ação de um tipo específico de filtro que é projetado para reduzir as frequências fora de uma faixa específica, de forma que o ponto de rolagem espectral é a fração de compartimentos no espectro de potência em que 85% da potência está em frequências mais baixas;
>>> 
>>> i) Espectral *flatness*: é uma medida usada no processamento de sinal digital para caracterizar um espectro de áudio e é normalmente medida em decibéis, fornecendo uma maneira de quantificar o quanto um som se assemelha a um tom puro, em vez de ser semelhante a ruído; 
>>> 
>>> j) *Zero-crossing rate*: é a medida da taxa na qual o sinal está passando pela linha zero, mais formalmente, o sinal está mudando de positivo para negativo ou vice-versa.
>>>
>> Ainda seria possível a análise subjetiva através de pesquisa entre os alunos da disciplina, contudo, considerando o conjunto universo de alunos, não haveria um acréscimo significativo na estatística comparativa entre os dados reais e sintetizados.



## Resultados e Discussão dos Resultados

>**GANSynth:** Embora muitas tentativas de adaptação da arquitetura, mudanças de hiperparâmetros e tempo de treinamento tenham sido realizadas, para a GANSynth não existem resultados compatíveis com a proposta estabelecida na ref. Bibliográfica [1], uma vez que a GAN não convergiu mesmo com todos os esforços, de forma que os áudios resultantes podem ser classificados apenas como ruído com a frequência fundamental em 8kHz, como se segue:
>> | | |
>> |--|--|
>> | [audio](https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/h04.wav)  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/audiogan.png" height="200">  |
>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/stfgan.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/cromagan.png" height="200"> |
>> 
>**LSTM:** Para o LSTM houve convergência, de forma que temos os seguintes resultados do treino:
>><p align="center">
>><img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/pretreinado.png" height="700">
>></p>

>>**MIR:** Além disso, temos o seguinte cenário de MIR para o áudio de entrada e o áudio de saída:

>>> |Áudio Real  | Áudio Sintetizado| 
>>> |--|--|
>>> | [audio](https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/input.wav)  | [audio](https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/output1.wav)  |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/barre.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/barsi.png" height="200"> |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/midmap1.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/midmap2.png" height="200"> |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/audiore.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/audiosin.png" height="200">  |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/stftre.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/stftsin.png" height="200"> |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/especre.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/especsi.png" height="200"> |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/melre.png" height="200"> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/melsin.png" height="200">  |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/cromare.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/cromasi.png" height="200">  |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/mfccre.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/mfccsin.png" height="200"> |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/band.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/rool.png" height="200"> |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/flat.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/zero.png" height="200">  |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/rms.png" height="200">  |   |

>>**Resposta Tabular:** Já para a resposta tabular, temos:
>>><p align="center">
>>><img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/tab.png" height="350">
>>><img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/tab2.png" height="350">
>>><img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/tab3.png" height="350">
>>></p>
>>>
>>>**KS Test = 0,7780642732875434**
>>>
>>**Pesquisa sensorial:** Também foi realizada uma pesquisa sensorial com os alunos da disciplina IA376L, de forma que 13 alunos responderam a pesquisa e o resultado para a composição LSTM foi o seguinte:
>>><p align="center">
>>><img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/pesqlstm.png" height="600">
>>></p>
>>>
>>Assim, para o grupo, a resposta do LSTM foi bastante proveitosa em termos de Music Information Retrieval, sendo apenas considerável a mudança do compasso 3/4 da música original para 4/4 e o aumento significativo da energia no áudio de saída, com magnitude aproximadamente 9x maior que o áudio original, além disso, o resultado esperado respeita o campo harmônico original e o bit (168), resultando em um áudio plausível como sequência melódica.
>>
>>Já como dado tabular, temos um valor de KS Test de 0,7780642732875434 indicando que há proximidade, mas não houve cópia exata dos dados de entrada, os ângulos de desvio-padrão e médias log-log também estão fora do ângulo de 45° e a soma cumulativa também indica alguma semelhança, mas sem cópia. Além disso, pela pesquisa sensorial o que fica evidente é que a maioria das pessoas considerou o áudio como uma música sem ruído, com audibilidade agradável e com uma composição coerente, confundível com um áudio real, resultando numa nota média 7,7/10.


>**Transformer:** Para o Transformer houve convergência, de forma que temos os seguintes resultados:

>>**MIR:** Além disso, temos o seguinte cenário de MIR para o áudio de entrada e o áudio de saída:

>>> |Áudio Real  | Áudio Sintetizado| 
>>> |--|--|
>>> | [audio](https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/input.wav)  | [audio](https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/output2.wav)  |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/barre.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/barsi2.png" height="200"> |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/midmap1.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/midmap3.png" height="200"> |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/audiore.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/audiot.png" height="200">  |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/stftre.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/stftt.png" height="200"> |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/especre.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/espect.png" height="200"> |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/melre.png" height="200"> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/melspect.png" height="200">  |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/cromare.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/cromat.png" height="200">  |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/mfccre.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/mfcct.png" height="200"> |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/bandt.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/rollt.png" height="200"> |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/flatt.png" height="200">  | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/zerot.png" height="200">  |
>>> | <img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/rmst.png" height="200">  |   |

>>**Resposta Tabular:** Já para a resposta tabular, temos:
>>><p align="center">
>>><img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/a01.png" height="350">
>>><img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/a02.png" height="350">
>>><img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/a03.png" height="350">
>>></p>
>>>
>>>**KS Test = 0,7722103025240228**
>>>
>>**Pesquisa sensorial:** Também foi realizada uma pesquisa sensorial com os alunos da disciplina IA376L, de forma que 13 alunos responderam a pesquisa e o resultado para a composição LSTM foi o seguinte:
>>><p align="center">
>>><img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/pesqtrans.png" height="600">
>>></p>
>>>
>>Assim, para o grupo, a resposta do Transformer também foi bastante proveitosa em termos de Music Information Retrieval, de forma que a energia no áudio de saída manteve magnitude semelhante ao áudio original, o resultado esperado respeita o campo harmônico original e o compasso 3/4, tendo uma leve variação do bit original (168) para o bit sintetizado (171), resultando em um áudio plausível como sequência melódica.
>>
>>Já como dado tabular, temos um valor de KS Test de 0,7722103025240228 indicando que há proximidade, mas não houve cópia exata dos dados de entrada, os ângulos de desvio-padrão e médias log-log também estão fora do ângulo de 45° e a soma cumulativa também indica alguma semelhança, mas sem cópia. Além disso, pela pesquisa sensorial o que fica evidente é que, diferente da composição por LSTM, houve uma maior distribuição de opiniões sobre a qualidade do áudio, desta vez a maioria das pessoas considerou o áudio como uma música sem ruído, confundível com um áudio real, mas com audibilidade pouco agradável e uma composição estranha, resultando numa nota média 6,3/10.

## Conclusão

>* Recriar a GANSynth layer by layer, embora bastante didático, demonstrou ser desafiador e nos deparamos com entraves não necessariamente citados no artigo de referência, de forma que o grupo tomou a decisão de tentar também outras arquiteturas e metodologias. Todavia tentamos várias alternativas com relação aos hiperparâmetros, tempo de treinamento, alterações básicas de arquitetura, mas infelizmente esta montagem não resultou em áudio enquanto música, somente áudio com fundamental em 8kHz, de forma que podemos caracterizar como ruído;
>
>* A mudança de paradigma para note sequence, isto é, token de música, demonstra ser um fator bastante significativo com relação ao Music Information Retrieval, já que esta estrutura mantém as propriedades do arquivo MIDI e isto se reflete na qualidade do áudio. Para maior ilustração, cabe as figuras abaixo, indicando o formato do arquivo mid e o arquivo de note sequence;
>Arquivo MIDI:
><p align="center">
><img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/arquivomid.png" height="400">
></p>	
><p align="center">
>Fonte: Referência Bibliográfica [4].
></p>
>Note sequence:
><p align="center">
><img src="https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E3/reports/figures/notesequence.png" height="50">
></p>
>
><p align="center">
>Fonte: Referência Bibliográfica [4].
></p>
>
>* Mesmo com a perplexidade próxima a 2, o grupo considera que os resultados da LSTM foram bastante representativos, acontecendo o mesmo com o Transformer, de forma que ambos os áudios estavam em um range plausível para a entrada;
>* Embora em termos estruturais o resultado do Transformer seja mais fiel ao padrão de entrada, isto é manteve o compasso 3/4, a magnitude da energia e o padrão de composição, o LSTM teve maior aceitação perante o público conforme a pesquisa sensorial, de forma que fica evidente que embora tecnicamente o Transformer tenha um desempenho superior, lhe falta criatividade de composição e definição de áudio. 
>
>* Procuramos compreender resultados parciais da entrega anterior em busca de respostas para tentar corrigir as mudanças de compasso, bit, magnitude que apareceram no LSTM, contudo, nossos esforços pessoais não resultaram em melhora e, também, a equipe do Magenta, responsável direto pelos modelos pré-treinados não deu retorno com relação aos questionamentos, além disso, após o trabalho realizado, ocorreram atualizações no Google Colab e no Tensorflow, de forma que isso prejudicou bastante o desempenho dos modelos pré-treinados, havendo a necessidade de downgrade para versões anteriores.

## Referências Bibliográficas
>[1] Engel, J.; Agrawal, K. K.; Chen, S.; Gulrajani, I.; Donahue, C.; Roberts, A.;”**Gansynth: Adversarial Neural Audio Synthesis**”; ICLR; 2019;
>
>[2]  Conner, M.; Gral, L.; Adams, K.; Hunger, D.; Strelow, R.; Neuwirth, A.; "**Music Generation Using an LSTM**"; MICS; 2022;
>
>[3] Huang, C.; Vaswani, A.; Uszkoreit, J.; Shazeer, N.; Simon, I.; Hawthorne, C.; Dai, A.; Hoffman, M., Dinculescu, M.; Eck, D.; "**Music Transformer**"; 2018;
>
>[4] Muller, M.; ”**Information Retrieval for Music and Motion**”; 1ª Edição; Editora Springer; 2010.
