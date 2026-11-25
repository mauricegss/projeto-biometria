// Compilar a partir da raiz do repositório (o artigo importa figuras de results/):
//   typst compile --root . artigo/main.typ artigo/artigo.pdf
// Modelo IEEE de conferência (charged-ieee), o mesmo usado pela Dupla 8.

#import "@preview/charged-ieee:0.1.4": ieee

#set text(lang: "pt", region: "br")

#show: ieee.with(
  title: [Verificação de Locutor com ECAPA-TDNN Pré-Treinada: Avaliação no VCTK, no LibriSpeech e Comparação com uma ResNet-34],
  authors: (
    (name: "Luiza Baldoqui", organization: [Universidade Tecnológica Federal do Paraná], location: [Ponta Grossa, Brasil]),
    (name: "Maurice Santos", organization: [Universidade Tecnológica Federal do Paraná], location: [Ponta Grossa, Brasil]),
  ),
  abstract: [
    Este trabalho avalia uma ECAPA-TDNN pré-treinada na VoxCeleb, sem _fine-tuning_, para verificação de locutor 1:1. Os _embeddings_ de 192 dimensões são comparados por similaridade de cosseno, e o limiar de decisão é congelado na validação do VCTK, a base principal. No teste do VCTK, o sistema obteve EER de 1,36% e AUC de 0,9991. Aplicado sem ajuste ao LibriSpeech, obteve EER de 0,64% nos nossos pares e de 0,68% nos pares da outra dupla da modalidade, com FMR próxima de 1% no limiar congelado. O desempenho cai pouco com ruído branco (EER de 8,5% a 0 dB), mas muito com ruído de várias vozes (39,1% a 0 dB). Na comparação com a ResNet-34 da Dupla 8, o modelo menor teve EER mais baixo nas duas bases, enquanto a ECAPA-TDNN manteve a FMR mais perto do ponto de operação escolhido. A análise de erros mostra que os falsos _matches_ ocorrem quase sempre entre locutores do mesmo sexo e sotaque, e que os falsos _non-matches_ do LibriSpeech vêm de um leitor que muda a voz ao interpretar personagens, e não de erro de rótulo.
  ],
  index-terms: ("biometria", "verificação de locutor", "ECAPA-TDNN", [_embeddings_], "avaliação entre bases"),
  bibliography: bibliography("refs.bib"),
  paper-size: "a4",
)
#set figure(placement: auto)
#show figure.where(kind: table): set figure(supplement: [Tabela])
// The template hard-codes "TABLE" in captions; use the Portuguese label instead.
#show "TABLE": "TABELA"

= Introdução

A verificação de locutor responde a uma pergunta simples: duas gravações de voz pertencem à mesma pessoa? É uma comparação 1:1 e não se confunde com o reconhecimento de fala, que procura saber o que foi dito. A voz pode ser captada sem contato e com microfones comuns, o que a torna útil em atendimento telefônico e em dispositivos pessoais. Em troca, o sinal muda com o microfone, o ambiente, o ruído, a duração da fala e o próprio estado ou estilo de quem fala.

Os sistemas atuais usam redes profundas para transformar cada gravação em um vetor de dimensão fixa, o _embedding_, que é comparado a outro por uma medida de similaridade. Uma decisão de aceitação ou rejeição depende então de um limiar (_threshold_). O EER, métrica mais citada na área, não depende de um limiar fixado antes; um sistema real, porém, opera com um limiar escolhido de antemão, e o comportamento dele fora da condição em que foi calibrado é tão importante quanto o EER.

O objetivo deste trabalho é avaliar uma ECAPA-TDNN @desplanques2020ecapa pré-treinada na VoxCeleb @nagrani2017voxceleb @chung2018voxceleb2 em duas bases de fala em inglês, VCTK @yamagishi2019vctk e LibriSpeech @panayotov2015librispeech, com protocolo reprodutível e limiar congelado na validação. As contribuições são:

- a avaliação do mesmo modelo, sem ajuste, na base principal e em uma segunda base, com análise de robustez a ruído e estudo de ablação do pré-processamento;
- a comparação com a ResNet-34 da Dupla 8 nos mesmos pares de avaliação, discutindo desempenho, ponto de operação e custo computacional;
- uma análise de erros que associa os falsos _matches_ a sexo e sotaque e explica os falsos _non-matches_ do LibriSpeech pela interpretação de personagens em um audiolivro.

= Trabalhos relacionados

Os _i-vectors_ @dehak2011ivector dominaram a verificação de locutor por cerca de uma década, representando cada gravação em um espaço de baixa dimensão que modela ao mesmo tempo a variabilidade do locutor e a do canal. Os _x-vectors_ @snyder2018xvector substituíram essa modelagem por uma rede de atraso temporal (TDNN) treinada para classificar locutores, da qual se extrai um _embedding_ após uma camada de agregação estatística.

A ECAPA-TDNN @desplanques2020ecapa estende os _x-vectors_ com blocos Res2Net @gao2021res2net, que capturam padrões em várias escalas, recalibração de canais por _squeeze-and-excitation_ @hu2018senet, agregação das saídas de várias camadas e _pooling_ estatístico com atenção @okabe2018attentive. Treinada com perda de margem angular aditiva, a mesma ideia da ArcFace @deng2019arcface, ela obteve EER de 0,87% na lista de teste da VoxCeleb1 @desplanques2020ecapa. As redes residuais @he2016resnet, que tratam o espectrograma como imagem, são a principal alternativa; a ResNet-34 do WeSpeaker @wang2023wespeaker, usada pela Dupla 8, é um exemplo. Os valores publicados servem de referência, mas dependem da base, do protocolo e do pós-processamento de _scores_, e não são diretamente comparáveis aos deste trabalho.

= Bases de dados

*VCTK 0.92* @yamagishi2019vctk: 110 locutores nativos de inglês com sotaques variados, cada um lendo cerca de 400 frases em estúdio, gravadas em dois microfones (`mic1` e `mic2`). Usamos a versão `wav48_silence_trimmed`, em que 109 locutores têm áudio, num total de 88.328 arquivos contando os dois microfones. As frases são curtas (mediana de 3,1 s).

*LibriSpeech `train-clean-100`* @panayotov2015librispeech: cerca de 100 horas de audiolivros de domínio público lidos por 251 voluntários (125 mulheres e 126 homens), com mediana de 14 s por trecho. A gravação é feita em casa, com microfones variados, e cada locutor lê um ou mais capítulos.

As duas bases têm licença CC BY 4.0, não possuem protocolo oficial de verificação e não são redistribuídas no repositório; o README explica como obtê-las. A @tab:bases resume as características e o uso de cada uma neste trabalho.

#figure(
  table(
    columns: 3,
    align: (left, left, left),
    stroke: none,
    table.hline(),
    table.header([*Característica*], [*VCTK*], [*LibriSpeech*]),
    table.hline(stroke: 0.5pt),
    [Versão], [0.92 (_trimmed_)], [`train-clean-100`],
    [Locutores com áudio], [109], [251 (125 F, 126 M)],
    [Arquivos], [88.328 (2 microfones)], [28.539],
    [Duração mediana], [3,1 s], [14,0 s],
    [Aquisição], [Estúdio, leitura de frases], [Casa, audiolivros],
    [Taxa original], [48 kHz], [16 kHz],
    [Papel], [Principal (limiar e A, D, E)], [Segunda base (B e C)],
    table.hline(),
  ),
  caption: [Bases usadas. Durações medianas e contagem por sexo conforme o levantamento da Dupla 8 nas mesmas bases.],
) <tab:bases>

= Metodologia

== Pré-processamento

Cada arquivo é convertido para mono a 16 kHz. Um detector de atividade de voz (VAD do WebRTC, agressividade 2) remove trechos sem fala, e arquivos com menos de 1 s de fala restante são contados como falha de aquisição (FTA). Em seguida calcula-se o espectrograma log-Mel com 80 bandas (janela de 25 ms, passo de 10 ms, FFT de 512 pontos, 20 a 7.600 Hz), normalizado por subtração da média de cada banda ao longo da gravação (CMN). Essa normalização segue a da receita original do SpeechBrain; uma versão anterior que também dividia pelo desvio-padrão fazia os _embeddings_ colapsarem (cosseno de cerca de 0,98 entre locutores diferentes) e foi corrigida.

A @fig:pipeline resume o caminho de uma amostra até a decisão, e a @fig:exemplo mostra uma amostra real em cada etapa do pré-processamento. No exemplo, o VAD remove a pausa longa no meio do trecho e reduz 11,3 s de áudio a 8,7 s de fala.

#figure(
  {
    set text(size: 7pt)
    let etapa(t) = box(stroke: 0.5pt, inset: 3pt, radius: 2pt, t)
    let seta = h(2pt) + sym.arrow.r + h(2pt)
    etapa[áudio] + seta + etapa[mono 16 kHz] + seta + etapa[VAD] + seta + etapa[FTA \< 1 s?] + seta + etapa[log-Mel 80] + seta + etapa[CMN]
    linebreak()
    v(4pt)
    seta + etapa[ECAPA-TDNN] + seta + etapa[_embedding_ 192-d, L2] + seta + etapa[cosseno] + seta + etapa[_score_ ≥ 0,353?] + seta + etapa[aceita / rejeita]
  },
  caption: [Pipeline de verificação, da amostra bruta à decisão.],
) <fig:pipeline>

#figure(
  image("../results/paper/pipeline_example.png"),
  caption: [Uma amostra do LibriSpeech (6209-34599-0000) em cada etapa do pré-processamento.],
) <fig:exemplo>

== Modelo e _embedding_

Usamos a ECAPA-TDNN da biblioteca SpeechBrain @ravanelli2021speechbrain com os pesos `speechbrain/spkrec-ecapa-voxceleb`, treinados com AAM-Softmax nos conjuntos de treino da VoxCeleb1 e da VoxCeleb2. A rede recebe o log-Mel de 80 bandas e tem uma camada convolucional inicial de 1.024 canais, três blocos SE-Res2Net com dilatações 2, 3 e 4, a concatenação das saídas desses blocos (3.072 canais), _pooling_ estatístico com atenção e uma camada linear final que produz o _embedding_ de 192 dimensões. O modelo tem 20.767.552 parâmetros (79,2 MB), todos carregados dos pesos publicados.

O _embedding_ é normalizado em norma L2 antes da comparação. Nenhuma camada foi ajustada: todos os resultados são de _zero-shot_. A VoxCeleb é formada por vídeos de celebridades, e não há sobreposição conhecida de locutores com as bases avaliadas. O código também tem um caminho de _fine-tuning_ com AAM-Softmax (margem 0,2, escala 30, três _seeds_), testado em dados sintéticos, mas ele não foi executado nas bases reais (ver Limitações).

== Comparação e decisão

O _score_ de um par é a similaridade de cosseno entre os dois _embeddings_. O limiar é o ponto de EER na validação do VCTK (0,3532) e fica congelado em todos os testes, inclusive no LibriSpeech e sob ruído. O par é aceito quando o _score_ é maior ou igual ao limiar.

= Protocolo experimental <sec:protocolo>

*Divisão por identidade.* No VCTK, os 109 locutores foram divididos em treino, validação e teste (76, 16 e 17 locutores) com _seed_ 42, sem nenhum locutor em mais de uma partição. Como não houve _fine-tuning_, o treino não foi usado; a validação serve só para o limiar.

*Pares.* Na validação do VCTK há 160 pares genuínos e 5.000 impostores; no teste, 170 e 5.000. Pares com algum arquivo em FTA são excluídos e contados à parte, o que deixou 166 genuínos e 4.799 impostores no teste. No LibriSpeech há dois conjuntos: os nossos (2.510 genuínos e 4.998 impostores dos 251 locutores, usados no experimento entre bases) e os da Dupla 8 (3.800 genuínos e 38.000 impostores de 38 locutores de teste, usados na comparação entre CNNs). Os pares da Dupla 8 foram importados de um _commit_ fixo do repositório dela e conferidos por SHA-256.

*Experimentos.* (A) desempenho no VCTK; (B) matriz de 2 CNNs × 2 bases com a Dupla 8, em que cada base usa os pares da dupla que a tem como principal; (C) avaliação no LibriSpeech com o modelo e o limiar congelados; (D) robustez com ruído branco e ruído de várias vozes (_babble_, mistura de 3 a 6 falas do VCTK) a 20, 10, 5 e 0 dB de SNR; (E) ablação do VAD e da CMN, com o limiar de cada variante recalculado na validação.

*Métricas.* EER, FMR e FNMR no limiar, AUC da curva ROC, TAR com FAR de 1%, _d-prime_ e FTA, seguindo a terminologia da ISO/IEC 19795-1 @iso19795.

*Ambiente.* Os experimentos A, C, D e E rodaram em CPU AMD (Zen 3) com Windows e Python 3.13; o experimento B, em CPU Intel i5-10400F com Linux e Python 3.12. Para garantir que a troca de máquina não altera os resultados, 200 pares do experimento C foram reprocessados na segunda máquina: a maior diferença de _score_ foi de 0,0014 e a decisão foi a mesma em todos. O código, os protocolos, os _scores_ e as métricas estão no repositório #link("https://github.com/mauricegss/projeto-biometria")[`github.com/mauricegss/projeto-biometria`].

= Resultados

A @tab:principal resume os experimentos A, B e C. No VCTK, o EER foi de 1,36%, e o limiar da validação deixou FMR e FNMR próximos entre si (1,08% e 1,81%), sinal de que a validação representou bem o teste. No LibriSpeech, sem nenhum ajuste, o EER caiu para 0,64% nos nossos pares e 0,68% nos pares da Dupla 8. Com o limiar do VCTK, a FMR ficou em 1,84% e 1,11%, e a FNMR abaixo de 0,5%. As distribuições de _score_ e a curva ROC do VCTK estão na @fig:dist.

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: none,
    table.hline(),
    table.header([*Métrica*], [*VCTK*], [*Libri (nossos)*], [*Libri (D8)*]),
    table.hline(stroke: 0.5pt),
    [Experimento], [A], [C], [B],
    [EER (%)], [1,36], [0,64], [0,68],
    [FMR no limiar (%)], [1,08], [1,84], [1,11],
    [FNMR no limiar (%)], [1,81], [0,44], [0,47],
    [AUC], [0,9991], [0,9991], [0,9995],
    [TAR \@ FAR=1% (%)], [96,99], [99,44], [99,47],
    [_d-prime_], [4,78], [6,73], [6,75],
    [Pares gen./imp.], [166/4.799], [2.510/4.998], [3.800/38.000],
    [FTA (arquivos)], [322/13.917], [1/11.689], [0/4.444],
    table.hline(),
  ),
  caption: [Resultados da ECAPA-TDNN com limiar congelado em 0,3532 (validação do VCTK).],
) <tab:principal>

#figure(
  grid(columns: 2, gutter: 4pt,
    image("../results/vctk/distributions.png"),
    image("../results/vctk/roc.png"),
  ),
  caption: [VCTK, teste: distribuição dos _scores_ genuínos e impostores com o limiar (esq.) e curva ROC (dir.).],
) <fig:dist>

A @tab:padrao traz a tabela padronizada pedida no enunciado, para a base principal.

#figure(
  table(
    columns: 2,
    align: (left, left),
    stroke: none,
    table.hline(),
    table.header([*Informação*], [*Resultado*]),
    table.hline(stroke: 0.5pt),
    [Modalidade], [Voz (verificação de locutor 1:1)],
    [Dataset], [VCTK 0.92 (teste, 17 locutores)],
    [CNN], [ECAPA-TDNN (C = 1.024)],
    [Pré-treinamento], [VoxCeleb1 + VoxCeleb2, AAM-Softmax],
    [Nº de parâmetros], [20.767.552],
    [Dimensão do _embedding_], [192],
    [_Matching_], [Cosseno entre _embeddings_ L2],
    [Limiar], [0,3532 (EER da validação)],
    [EER], [1,36%],
    [AUC], [0,9991],
    [FMR], [1,08%],
    [FNMR], [1,81%],
    [TAR \@ FAR=1%], [96,99%],
    [FTA], [322/13.917 arquivos (2,3%)],
    [Tempo de inferência], [107,5 ± 54,1 ms por arquivo],
    [Tamanho do modelo], [79,2 MB],
    [Hardware], [CPU AMD Zen 3, sem GPU],
    table.hline(),
  ),
  caption: [Tabela padronizada de resultados (enunciado, §26).],
) <tab:padrao>

As distribuições também mostram por que o LibriSpeech foi mais fácil (@fig:distlibri). Nele, o _score_ genuíno médio foi 0,777 e o impostor 0,103 (_d-prime_ de 6,75); no VCTK, 0,592 e 0,115 (_d-prime_ de 4,78). Os impostores ficam no mesmo lugar nas duas bases, e o que muda é a cauda inferior dos genuínos: com frases curtas, mais pares do mesmo locutor caem perto do limiar.

#figure(
  grid(columns: 2, gutter: 4pt,
    image("../results/dupla8/librispeech/distributions.png"),
    image("../results/dupla8/librispeech/roc.png"),
  ),
  caption: [LibriSpeech, pares da Dupla 8 (Exp. B): distribuição dos _scores_ com o limiar do VCTK (esq.) e curva ROC (dir.).],
) <fig:distlibri>

*Robustez (D).* O ruído branco eleva o EER de 1,36% para 3,92% a 20 dB e para 8,54% a 0 dB. O _babble_ é bem mais prejudicial: 4,75% a 20 dB, 15,90% a 10 dB e 39,06% a 0 dB (@fig:rob). Com o limiar congelado, quase todo o aumento vem da FNMR, que chega a 87% com _babble_ a 0 dB, enquanto a FMR com _babble_ fica abaixo de 3%: o ruído afasta as amostras do mesmo locutor mais do que aproxima as de locutores diferentes.

#figure(
  image("../results/paper/robustness.png", width: 85%),
  caption: [EER no VCTK por tipo de ruído e SNR, com limiar congelado.],
) <fig:rob>

#figure(
  table(
    columns: 5,
    align: (left, right, right, right, right),
    stroke: none,
    table.hline(),
    table.header([*Condição*], [*EER*], [*FMR*], [*FNMR*], [*TAR\@1%*]),
    table.hline(stroke: 0.5pt),
    [limpo], [1,36], [1,08], [1,81], [96,99],
    [branco 20 dB], [3,92], [1,03], [7,19], [92,81],
    [branco 10 dB], [6,17], [1,45], [8,03], [90,12],
    [branco 5 dB], [5,87], [3,53], [8,82], [81,18],
    [branco 0 dB], [8,54], [6,04], [12,94], [62,35],
    [_babble_ 20 dB], [4,75], [1,12], [9,52], [89,88],
    [_babble_ 10 dB], [15,90], [1,86], [38,82], [54,12],
    [_babble_ 5 dB], [27,52], [2,26], [65,29], [26,47],
    [_babble_ 0 dB], [39,06], [2,90], [87,06], [6,47],
    table.hline(),
  ),
  caption: [Robustez no VCTK (%), limiar congelado em 0,3532.],
) <tab:rob>

#figure(
  table(
    columns: 5,
    align: (left, right, right, right, right),
    stroke: none,
    table.hline(),
    table.header([*Variante*], [*EER*], [*FMR*], [*FNMR*], [*AUC*]),
    table.hline(stroke: 0.5pt),
    [completo (VAD + CMN)], [1,36], [1,08], [1,81], [0,9991],
    [sem VAD], [1,10], [1,02], [1,18], [0,9994],
    [sem CMN], [1,81], [2,48], [1,20], [0,9984],
    [sem VAD e sem CMN], [2,36], [2,56], [1,76], [0,9973],
    table.hline(),
  ),
  caption: [Ablação do pré-processamento no VCTK (%). O limiar de cada variante vem da validação dela.],
) <tab:abl>

*Ablação (E).* Retirar a CMN eleva o EER de 1,36% para 1,81%, e retirar CMN e VAD juntos, para 2,36%. Retirar só o VAD reduziu o EER para 1,10%. As frases do VCTK já vêm sem os silêncios das bordas, e o VAD acaba cortando fala útil de frases curtas; além disso, sem VAD nenhum arquivo cai abaixo de 1 s, então essa variante é avaliada em todos os pares, inclusive os 205 que o sistema completo descarta.

*Comparação entre CNNs (B).* A @tab:matriz traz a matriz de EER das duas CNNs nos mesmos pares. A ResNet-34 da Dupla 8, pré-treinada na VoxCeleb2 com ArcFace e com 6,6 milhões de parâmetros, teve EER menor nas duas bases, e o _fine-tuning_ que a Dupla 8 fez no LibriSpeech o reduziu ainda mais. Com os limiares de cada dupla, porém, a ECAPA-TDNN manteve a FMR perto de 1% nas duas bases, enquanto a da ResNet-34 pré-treinada subiu para 3,02% no VCTK, e a da ajustada para 4,39%.

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    stroke: none,
    table.hline(),
    table.header([*Modelo*], [*VCTK (D7)*], [*Libri (D8)*]),
    table.hline(stroke: 0.5pt),
    [ECAPA-TDNN (nossa)], [1,36 / 1,08], [0,68 / 1,11],
    [ResNet-34 pré-treinada], [0,72 / 3,02], [0,39 / 1,26],
    [ResNet-34 ajustada (3 _seeds_)], [0,61 / 4,39], [0,20 / 0,66],
    table.hline(),
  ),
  caption: [EER (%) / FMR no limiar (%) das duas CNNs nos mesmos pares. Valores da ResNet-34 tirados do repositório da Dupla 8; o limiar dela vem da validação do LibriSpeech.],
) <tab:matriz>

*Eficiência.* A ECAPA-TDNN tem 20,8 milhões de parâmetros e 79,2 MB, contra 6,6 milhões e 25,4 MB da ResNet-34. Em CPU, a extração levou 107,5 ms por arquivo no VCTK e 348 ms no LibriSpeech, cujos trechos são cerca de quatro vezes mais longos; na segunda máquina, o LibriSpeech levou 709 ms. A Dupla 8 mediu cerca de 31 ms por arquivo em GPU, então os tempos não são comparáveis entre si, mas o número de parâmetros é.

= Análise e discussão

== Análise de erros

No limiar congelado houve 52 falsos _matches_ (FM) e 3 falsos _non-matches_ (FNM) no VCTK, e 423 FM e 18 FNM nos pares da Dupla 8 no LibriSpeech. A @fig:erros mostra os três FM e os três FNM de maior erro no LibriSpeech, e a @tab:erros, os do VCTK.

#figure(
  image("../results/paper/errors_librispeech.png", width: 80%),
  caption: [Log-Mel dos três falsos _matches_ (FM) e dos três falsos _non-matches_ (FNM) de maior erro no LibriSpeech, com _score_ e limiar de 0,353.],
  placement: top,
  scope: "parent",
) <fig:erros>

#figure(
  table(
    columns: 4,
    align: (left, left, left, right),
    stroke: none,
    table.hline(),
    table.header([*Caso*], [*Amostra 1*], [*Amostra 2*], [*Score*]),
    table.hline(stroke: 0.5pt),
    [FM1], [p318 F, EUA, mic2], [p306 F, EUA, mic1], [0,519],
    [FM2], [p228 F, Ingl., mic2], [p244 F, Ingl., mic1], [0,489],
    [FM3], [p240 F, Ingl., mic2], [p228 F, Ingl., mic2], [0,489],
    [FNM1], [p363 M, Can., mic1], [p363 M, Can., mic2], [0,307],
    [FNM2], [p363 M, Can., mic2], [p363 M, Can., mic2], [0,327],
    [FNM3], [p240 F, Ingl., mic1], [p240 F, Ingl., mic2], [0,341],
    table.hline(),
  ),
  caption: [Erros do VCTK: locutor, sexo, sotaque e microfone. Limiar de 0,353.],
  placement: top,
) <tab:erros>

Para cada caso, a hipótese técnica é:

- *FM1, VCTK (p318 × p306, 0,519):* duas mulheres americanas em microfones diferentes; voz e sotaque parecidos dominam o _embedding_, e o canal diferente não ajuda a separá-las.
- *FM2, VCTK (p228 × p244, 0,489):* duas jovens inglesas; esse par de locutoras sozinho gera 17 FM, o que indica vozes de fato próximas no espaço de _embeddings_, e não um erro pontual.
- *FM3, VCTK (p240 × p228, 0,489):* de novo duas jovens do sul da Inglaterra, no mesmo microfone; sem diferença de canal, sobra só a semelhança de voz.
- *FM1–FM3, LibriSpeech (0,523 a 0,550):* pares do mesmo sexo, com trechos longos e leitura neutra. A duração não explica o erro; a hipótese é a semelhança de timbre somada ao estilo uniforme de leitura de audiolivro. No FM1, uma das gravações tem uma faixa constante de ruído em alta frequência (@fig:erros), sinal de canal pior.
- *FNM1, VCTK (p363, mic1 × mic2, 0,307):* mesmo locutor em microfones diferentes; a mudança de canal reduz a similaridade de um locutor que já varia mais que a média.
- *FNM2, VCTK (p363, mic2 × mic2, 0,327):* mesmo locutor e mesmo microfone; o erro aponta para variação na própria voz de p363 entre as frases.
- *FNM3, VCTK (p240, mic1 × mic2, 0,341):* _score_ muito perto do limiar; frases curtas e troca de microfone bastam para cruzá-lo.
- *FNM1–FNM3, LibriSpeech (0,148 a 0,159):* todos do leitor 6209 contra trechos do capítulo 34601; a hipótese, discutida abaixo, é a interpretação de personagens. O FNM3 tem ainda um trecho de só 2,1 s.

*Falsos _matches_.* Os seis casos são entre pessoas do mesmo sexo, e no VCTK também do mesmo sotaque. O padrão vale para todos os erros: 98% dos FM do VCTK são entre pessoas do mesmo sexo e 71% entre pessoas do mesmo sotaque, contra 53% e 19% entre todos os pares impostores. No LibriSpeech, a FMR entre pessoas do mesmo sexo foi de 2,1%, contra 0,16% entre sexos diferentes. Os erros também se concentram em poucos pares de locutores: só p228 e p244, duas jovens inglesas, respondem por 17 dos 52 FM do VCTK. A hipótese é que o _embedding_ separa bem características globais, como sexo e sotaque, mas tem menos margem para distinguir vozes parecidas dentro de um mesmo grupo. Nos FM do LibriSpeech, os trechos são longos (6 a 16 s) e de leitura neutra, então a duração não explica o erro; a semelhança de estilo de leitura de audiolivro deve contribuir.

*Falsos _non-matches_ no VCTK.* Os três têm _score_ logo abaixo do limiar (0,31 a 0,34). Dois envolvem microfones diferentes, mas, no conjunto, o _score_ genuíno médio quase não muda entre pares do mesmo microfone e de microfones diferentes (0,595 e 0,587), então o canal explica pouco. O locutor p363 aparece em dois casos, o que sugere maior variação dentro da própria voz dele.

*Falsos _non-matches_ no LibriSpeech.* Dos 18 FNM, 13 são do locutor 6209 e todos envolvem o capítulo 34601 de _O Homem que Ri_. A @fig:6209 mostra o _score_ médio de cada trecho desse capítulo contra os outros dois capítulos do mesmo leitor. Os trechos abaixo do limiar (29, 31, 32, 33 e 38) são o monólogo do personagem Ursus; o trecho 30, entre eles, é narração e tem _score_ normal; o trecho 7 é uma fala da criança e fica no limite; e o 9 tem só 2 s. A frequência fundamental desses trechos ficou na mesma faixa da narração (104 a 134 Hz), o que não indica outro leitor. A explicação mais provável é que o leitor muda a voz ao interpretar personagens, e o modelo trata a voz interpretada como outra pessoa. Isso é variação dentro do mesmo locutor, e não erro de rótulo da base: o modelo erra porque a voz de fato mudou.

#figure(
  image("../results/paper/speaker_6209.png"),
  caption: [Locutor 6209: _score_ médio de cada trecho do capítulo 34601 contra os capítulos 34599 e 34600. Em vermelho, abaixo do limiar.],
) <fig:6209>

== Entre bases, robustez e comparação

O desempenho no LibriSpeech foi melhor que no VCTK, embora o limiar tenha vindo do VCTK. A duração é a explicação mais provável: os trechos do LibriSpeech têm cerca de 14 s, e as frases do VCTK, cerca de 3 s, e _embeddings_ de falas curtas são mais instáveis. Os 322 arquivos do VCTK descartados por terem menos de 1 s de fala reforçam esse ponto. A FMR de 1,84% nos nossos pares do LibriSpeech, acima da do VCTK, mostra que o limiar não se transfere perfeitamente entre bases, mas a diferença foi pequena.

O _babble_ é o pior caso porque o ruído também é voz humana: ele carrega características de outros locutores para dentro do _embedding_, enquanto o ruído branco só encobre parte do espectro. Para uso em ambientes com várias pessoas falando, o sistema precisaria de supressão de ruído ou de treino com esse tipo de degradação.

Na comparação com a Dupla 8, um modelo três vezes maior não trouxe EER menor, o que responde à pergunta do enunciado sobre custo e benefício: aqui o custo extra da ECAPA-TDNN não se paga em EER. A vantagem dela apareceu na estabilidade do ponto de operação, com FMR perto de 1% nas duas bases. A comparação tem duas ressalvas: os limiares vêm de bases diferentes (VCTK para nós, LibriSpeech para a Dupla 8), e no VCTK a ECAPA-TDNN foi avaliada sem os 205 pares descartados por FTA, enquanto a ResNet-34 avaliou todos.

== Falhas de aquisição

A regra de FTA (menos de 1 s de fala após o VAD) descartou 2,3% dos arquivos do VCTK e quase nenhum do LibriSpeech. Descartar uma amostra curta é uma escolha de projeto legítima, porque um _embedding_ de menos de 1 s é pouco confiável, mas ela tem custo: o usuário que fala pouco é rejeitado antes da comparação, e esse custo não aparece na FNMR. Por isso a FTA é reportada à parte (@tab:principal) e precisa ser considerada junto do EER ao comparar com sistemas que não descartam nada, como a ResNet-34 da Dupla 8. A variante sem VAD da ablação dá uma ideia do efeito: avaliando todos os pares, o EER no VCTK foi de 1,10%.

== Limitações

Não fizemos _fine-tuning_, então não há média e desvio-padrão entre _seeds_; todos os números vêm de uma única execução determinística do modelo pré-treinado. O teste do VCTK tem só 17 locutores e 166 pares genuínos, o que deixa o EER sensível a poucos erros: cada FNM vale 0,6 ponto percentual de FNMR. O _babble_ foi gerado com falas do próprio VCTK, o que pode tornar o ruído mais parecido com os locutores avaliados do que seria em uso real.

= Conclusão

A ECAPA-TDNN pré-treinada, sem nenhum ajuste, obteve EER de 1,36% no VCTK e de 0,64% e 0,68% no LibriSpeech, com o limiar calibrado no VCTK mantendo FMR perto de 1% nas duas bases. O sistema resiste bem a ruído branco e mal a ruído de várias vozes. Na comparação com a ResNet-34 da Dupla 8, o modelo maior não teve EER menor, mas manteve o ponto de operação mais estável entre bases. A análise de erros mostrou que os falsos _matches_ se concentram entre pessoas do mesmo sexo e sotaque, e que os falsos _non-matches_ do LibriSpeech vêm de um leitor que interpreta personagens. Como trabalhos futuros, ficam o _fine-tuning_ com várias _seeds_, verificando se ele melhora o EER sem desestabilizar a FMR fora da base de treino, como aconteceu com a ResNet-34, e o treino com ruído de várias vozes.
