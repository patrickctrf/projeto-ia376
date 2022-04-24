# Deep Learning Aplicado a Síntese de Sinais – Primeira Entrega

# `<Síntese de timbres de instrumentos musicais através de Redes Adversárias Generativas>`
# `<Synthesis of musical instrument timbres through Generative Adversary Networks>`

## Apresentação:

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação *IA376L - Deep Learning aplicado a Síntese de Sinais*, 
oferecida no primeiro semestre de 2022, na Unicamp, sob supervisão da Profa. Dra. Paula Dornhofer Paro Costa, do Departamento de Engenharia de Computação e Automação (DCA) da Faculdade de Engenharia Elétrica e de Computação (FEEC).

## Equipe:

> |Nome  | RA | Especialização|
> |--|--|--|
> | Gabriel Santos Martins Dias  | 172441  | Eng. Eletricista |
> | Gleyson Roberto do Nascimento  | 043801  | Eng. Eletricista |
> | Patrick Carvalho Tavares R. Ferreira  | 175480  | Eng. Eletricista |


## Descrição Resumida do Projeto:

> Neste projeto, temos por objetivo a síntese generativa dos timbres  de instrumentos musicais como flauta, violão, piano, órgão, metais (sopro), cordas e demais exemplares que constam no Nsynth Dataset através do uso de Redes Adversárias Generativas.
 
### Vídeo de apresentação da proposta do projeto:
[![Projeto IA376](https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E1/M%C3%ADdia/v%C3%ADdeo.JPG)](https://youtu.be/LqRADM9dA_o)


## Metodologia Proposta:

### Base de Dados:

> Neste primeiro momento, para o projeto serão utilizadas as seguintes bases de dados:
> |Base de Dados  | Fonte | Descrição|
> |--|--|--|
> | [Nysinth – Treino](https://app.activeloop.ai/activeloop/nsynth-train) | *Active Loop*  | Áudios de instrumentos musicais, no formato midi, captados com boa definição e que possuem rotulagem quanto ao tipo de instrumento, *pitch* e *sample rate* (dados para treino)|
> | [Nysinth – Teste](https://app.activeloop.ai/activeloop/nsynth-test) | *Active Loop*  | Áudios de instrumentos musicais, no formato midi, captados com boa definição e que possuem rotulagem quanto ao tipo de instrumento, *pitch* e *sample rate* (dados para teste)|
> | [Nysinth – Validação](https://app.activeloop.ai/activeloop/nsynth-val) | *Active Loop*  | Áudios de instrumentos musicais, no formato midi,  captados com boa definição e que possuem rotulagem quanto ao tipo de instrumento, *pitch* e *sample rate* (dados para validação)|

### Arquitetura da Rede Neural:

> Para a síntese dos timbres dos instrumentos que fazem parte do Nsynth Dataset, uma abordagem interessante é a GAN conhecida por GANSynth, cuja arquitetura é representada por:
>
![arquitetura](https://github.com/patrickctrf/projeto-ia376/blob/e1gr/E1/M%C3%ADdia/arquitetura_gansynth.jpg)
>
> Assim, o modelo amostra um vetor aleatório z de uma gaussiana esférica e o executa através de uma pilha de convoluções transpostas para aumentar a amostragem e gerar dados de saída x = G(z), que são alimentados em uma rede discriminadora de convoluções de redução de amostragem (cuja arquitetura espelha a do gerador) para estimar uma medida de divergência entre as distribuições real e gerada. 

### Artigos de Referência:

> Em [1] temos a apresentação da GANSynth e sua arquitetura, de forma que representa a principal referência para este projeto;
> 
> Em [2] temos a apresentação da abordagem da síntese de imagens que inspirou o artigo da GANSynth, de forma que ela apresenta maiores detalhes com relação ao processo de síntese da GAN;
> 
> Em [3] temos as definições de *Music Information Retrieval* (MIR) e, consequentemente, das *features* a serem observadas para a comparação entre os dados reais e os dados sintetizados;  

### Ferramentas a serem utilizadas:

> Neste início de projeto, elencamos as seguintes ferramentas:
> |Ferramenta | Descrição|
> |--|--|
> | [Google Colab](https://colab.research.google.com/) | Ferramenta para elaboração dos notebooks e códigos em linguagem Python 3.8 |
> | [Active Loop - HUB](https://github.com/activeloopai/Hub) | Ferramenta para manipular o repositório com o Nsynth Dataset |
> | [Tensorflow](https://www.tensorflow.org/?hl=pt-br) | Ferramenta principal de manipulação de tensores e construção da GAN |
> | [Pytorch](https://pytorch.org/) | Ferramenta secundária de manipulação de tensores e rede neural |
> | [Magenta](https://magenta.tensorflow.org/) | Ferramenta para manipulação de áudio em inteligência artificial |
> | [Librosa](https://librosa.org/doc/latest/index.html) | Ferramenta de análise de *features* de áudio |
> | [SDV – *Table Evaluate*](https://sdv.dev/SDV/user_guides/evaluation/single_table_metrics.html) | Ferramenta para avaliação de dados tabulares |
> | [Seaborn](https://seaborn.pydata.org/) | Ferramenta para *Data Visualization* |

### Resultados Esperados:

> A síntese de áudio ainda está em franca evolução, todavia, pelos artigos envolvendo o uso da GANSynth, o que esperamos é que a diferença entre os dados sintetizados e os reais não seja tão grande tanto nas análises quantitativas quanto qualitativas e, principalmente, que respeite a ordem de grandeza das *features* espectrais de MIR. 

### Proposta de avaliação:

> Nossa proposta para a avaliação comparativa entre os dados reais e sintezados contará com três abordagens: quantitativa, qualitativa e MIR, desta forma, teremos:
>
> * **Análise Quantitativa:** considerando-se que a entrada e a saída do áudio serão dados do tipo *numpy array*, através da ferramenta Table Evaluate, será analisado o KS Teste, a fim de determinar a distância entre o que foi gerado e o real resultando no *score* entre eles;
>
> * **Análise Qualitativa:** gráfico log-log da média dos dados reais x média dos dados sintetizados (para verificar tendência de 45º); do gráfico log-log do desvio-padrão dos dados reais x desvio-padrão dos dados sintetizados (para verificar tendência de 45°); gráfico da soma cumulativa dos dados reais e sintetizados (para verificar diferença de distribuição), histograma dos dados reais e sintetizados (para verificar diferença de distribuição);
>
> * **Music Information Retrieval:** através das análises espectrais existentes na Biblioteca Librosa, fazer o comparativo entre as *features* espectrais do áudio real e do áudio sintetizado. Serão analisadas as seguintes *features* espectrais: 
> 
>> a) STFT:  a transformada de Fourier de tempo curto é uma transformada integral derivada da transformada de Fourier, sendo uma forma de representação tempo-frequência;
>> 
>> b) Espectrograma: pode ser definido como um gráfico que mostra a intensidade por meio do escurecimento ou coloração do traçado, as faixas de frequência no eixo vertical e o tempo no eixo horizontal. Sua representação mostra estrias horizontais, denominadas harmônicos;
>> 
>> c) Mel espectrograma: é um espectrograma onde as frequências são convertidas para a escala mel, isto é, uma escala onde a unidade de altura é tal que distâncias iguais em altura soam igualmente distantes para o ouvinte;
>> 
>> d) Cromagrama: é o gráfico do vetor de características tipicamente de 12 elementos que indica quanta energia de cada classe de altura, {C, C#, D, D#, E, ..., B}, é presente no sinal;
>> 
>> e) MFCC: Os coeficientes cepstral de frequência Mel (MFCCs) são coeficientes que coletivamente compõem uma representação do espectro de potência de curto prazo de um som, com base em uma transformação de cosseno linear de um espectro de potência de log em uma escala de frequência mel não linear;
>> 
>> f) RMS: potência real (Root Mean Squared) do sinal;
>> 
>> g) Espectral *bandwith*: é a diferença entre as frequências superior e inferior em uma banda contínua de frequências;
>> 
>> h) Espectral *rolloff*: Pode ser definido como a ação de um tipo específico de filtro que é projetado para reduzir as frequências fora de uma faixa específica, de forma que o ponto de rolagem espectral é a fração de compartimentos no espectro de potência em que 85% da potência está em frequências mais baixas;
>> 
>> i) Espectral *flatness*: é uma medida usada no processamento de sinal digital para caracterizar um espectro de áudio e é normalmente medida em decibéis, fornecendo uma maneira de quantificar o quanto um som se assemelha a um tom puro, em vez de ser semelhante a ruído; 
>> 
>> j) *Zero-crossing rate*: é a medida da taxa na qual o sinal está passando pela linha zero, mais formalmente, o sinal está mudando de positivo para negativo ou vice-versa.
>
> Ainda seria possível a análise subjetiva através de pesquisa entre os alunos da disciplina, contudo, considerando o conjunto universo de alunos, não haveria um acréscimo significativo na estatística comparativa entre os dados reais e sintetizados.


## Cronograma:
> Proposta de cronograma:
>
> |Atividade  | Descrição | Tempo estimado|
> |--|--|--|
> | Entendimento do Problema | Análise do problema e busca de artigos para encontrarmos uma boa metodologia a para a síntese de timbre instrumental  | 16/03 a 06/04 (3 semanas)|
> | Entendimento dos Dados  | Busca e avaliação dos dados necessários para o projeto   | 06/04 a 20/04 (2 semanas)|
> | Entrega E1  | Discussão, formalização e elaboração do *commit* da E1 no Github do projeto | 20/04 a 27/04 (1 semana) |
> | Arquitetura da GANSynth  | Montagem e testes da arquitetura do GANSynth no Google Colab | 27/04 a 11/05 (2 semanas) |
> | Entrega E2 - Checkpoint  | Discussão, formalização e elaboração do *commit* da E2 no Github do projeto | 11/05 a 18/05 (1 semana) |
> | Análises dos Dados Sintetizados | Elaboração do código para a síntese, avaliação dos dados sintetizados e análises dos resultados | 11/05 a 08/06 (4 semanas) |
> | Finalização | Ajustes finais para a entrega do projeto e análise crítica dos resultados e suas contribuições | 08/06 a 15/06 (1 semana) |
> | Entrega E3 – Código Final  | Discussão, formalização e elaboração do *commit* da E3 no Github do projeto | 15/06 a 22/06 (1 semana) |
> | Entrega E4 – Apresentação do Projeto  | Discussão, formalização e elaboração do *commit* da E4 no Github do projeto | 22/06 a 06/07 (2 semanas) |




## Referências Bibliográficas:
> [1] Engel, J.; Agrawal, K. K.; Chen, S.; Gulrajani, I.; Donahue, C.; Roberts, A.;**” Gansynth: Adversarial Neural Audio Synthesis”**; ICLR; 2019;
>
> [2] Karras, T.; Laine, S.; Aila, T.; **”A style-based generator architecture for generative adversarial networks”**; CoRR; abs/1812.04948; 2018b;
> 
> [3] Muller, M.; **”Information Retrieval for Music and Motion”**;  1ª Edição;  Editora Springer;  2010.

