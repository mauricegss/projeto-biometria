// Compilar a partir da raiz do repositório:
//   typst compile --root . apresentacao/main.typ apresentacao/apresentacao.pdf
// Ordem do enunciado (§27): problema → dataset → pipeline → CNN → embedding →
// matching → protocolo → resultados → cross-dataset → robustez → erros → conclusões.

#set page(paper: "presentation-16-9", margin: (x: 1.6cm, top: 1.4cm, bottom: 1.1cm))
#set text(font: "Libertinus Serif", size: 19pt, lang: "pt", region: "br")
#set par(justify: false, leading: 0.6em)
#show heading.where(level: 1): set text(size: 28pt, fill: rgb("#1f4e79"))
#show strong: set text(fill: rgb("#1f4e79"))

#let slide(title, body) = {
  pagebreak(weak: true)
  heading(level: 1, title)
  v(0.3em)
  body
  place(bottom + right, dy: 0.6cm, text(size: 11pt, fill: gray)[
    Dupla 7 · Biometria UTFPR-PG · #context [#counter(page).display() / #counter(page).final().first()]
  ])
}
#let cols(..args) = grid(columns: (1fr, 1fr), gutter: 1.2em, ..args)
#let fig(path, ..args) = image("../" + path, ..args)
#let destaque(body) = block(fill: rgb("#eaf1f8"), inset: 0.6em, radius: 4pt, width: 100%, body)

// 1
#align(center + horizon)[
  #text(size: 32pt, fill: rgb("#1f4e79"))[*Verificação de locutor com ECAPA-TDNN*]
  #v(0.2em)
  #text(size: 20pt)[VCTK, LibriSpeech e comparação com a ResNet-34 da Dupla 8]
  #v(1em)
  Luiza Baldoqui · Maurice Santos \
  #text(size: 15pt)[Biometria · UTFPR Ponta Grossa · 25/11/2026]
  #v(0.6em)
  #text(size: 13pt, fill: gray)[github.com/mauricegss/projeto-biometria]
]

#slide[O problema][
  - *Verificação 1:1:* duas gravações de voz são da mesma pessoa?
  - Não é reconhecimento de fala: importa *quem* falou, não *o que* foi dito.
  - Voz é captada sem contato, com microfone comum, mas muda com microfone, ruído, duração e o estado de quem fala.
  - *Nossa pergunta:* uma ECAPA-TDNN pré-treinada, *sem ajuste*, funciona bem em bases que ela nunca viu? E com um limiar fixado antes?
]

#slide[Bases de dados][
  #table(
    columns: (auto, 1fr, 1fr),
    stroke: none,
    table.hline(),
    table.header([], [*VCTK 0.92* (principal)], [*LibriSpeech* (segunda)]),
    table.hline(stroke: 0.5pt),
    [Locutores], [109 com áudio], [251 (125 F, 126 M)],
    [Gravação], [estúdio, 2 microfones], [audiolivros, em casa],
    [Duração mediana], [*3,1 s*], [*14,0 s*],
    [Uso], [limiar, A, D, E], [B, C],
    table.hline(),
  )
  #v(0.5em)
  - Nenhuma das duas tem protocolo oficial de verificação → protocolo próprio, com _seed_ 42.
  - Guardem a diferença de duração: ela explica muito dos resultados.
]

#slide[Pipeline][
  #align(center)[
    #set text(size: 15pt)
    #let e(t) = box(stroke: 0.7pt, inset: 5pt, radius: 3pt, t)
    #e[áudio] → #e[mono 16 kHz] → #e[VAD] → #e[FTA se < 1 s] → #e[log-Mel 80] → #e[CMN] \
    #v(0.3em)
    → #e[ECAPA-TDNN] → #e[_embedding_ 192-d] → #e[cosseno] → #e[≥ 0,353?] → #e[decisão]
  ]
  #v(0.2em)
  #align(center, fig("results/paper/pipeline_example.png", height: 52%))
]

#slide[CNN e _embedding_][
  #cols[
    - *ECAPA-TDNN* (SpeechBrain), C = 1.024
    - 3 blocos SE-Res2Net (dilatações 2, 3, 4)
    - agrega as saídas dos blocos + _pooling_ estatístico com atenção
    - saída: *_embedding_ de 192 dimensões*, normalizado L2
  ][
    - Pesos `spkrec-ecapa-voxceleb`: VoxCeleb1 + 2, AAM-Softmax
    - *20,8 milhões de parâmetros*, 79 MB
    - *Zero-shot*: nenhuma camada ajustada
    - _Embedding_, não classificador: serve para qualquer locutor novo
  ]
  #v(0.4em)
  #destaque[Bug que achamos: a CMN dividia pelo desvio-padrão e colapsava os _embeddings_ (cosseno ≈ 0,98 entre pessoas diferentes, EER ≈ 40%). Corrigido para só subtrair a média.]
]

#slide[_Matching_ e protocolo][
  - *_Score_* = cosseno entre os dois _embeddings_.
  - *Limiar* = ponto de EER na *validação* do VCTK (0,3532), *congelado* em todos os testes.
  - VCTK dividido por identidade: 76 / 16 / 17 locutores, sem repetição.
  - Teste VCTK: 170 genuínos + 5.000 impostores (166 + 4.799 após FTA).
  - *Exp. B:* cada base usa os pares da dupla que a tem como principal → no LibriSpeech, os 41.800 pares da Dupla 8, conferidos por hash.
  - Reprodutibilidade: 200 pares refeitos em outra máquina, diferença máxima de _score_ de 0,0014.
]

#slide[Resultados: base principal][
  #cols[
    #fig("results/vctk/distributions.png", width: 100%)
  ][
    #table(
      columns: 2, stroke: none,
      [*EER*], [*1,36%*],
      [FMR / FNMR], [1,08% / 1,81%],
      [AUC], [0,9991],
      [TAR \@ FAR 1%], [96,99%],
      [FTA], [322 arquivos (2,3%)],
    )
    #v(0.3em)
    #text(size: 16pt)[FMR e FNMR próximos: o limiar da validação representou bem o teste.]
  ]
]

#slide[Entre bases e 2 CNNs × 2 bases][
  #cols[
    *ECAPA-TDNN no LibriSpeech, sem ajuste*
    - nossos pares (C): EER *0,64%*
    - pares da Dupla 8 (B): EER *0,68%*
    - FMR no limiar do VCTK: 1,84% e 1,11%
    #v(0.3em)
    #text(size: 16pt)[Melhor que no VCTK: trechos de 14 s dão _embeddings_ mais estáveis que frases de 3 s.]
  ][
    #set text(size: 16pt)
    #table(
      columns: 3, stroke: none,
      table.hline(),
      table.header([EER / FMR (%)], [*VCTK*], [*Libri*]),
      table.hline(stroke: 0.5pt),
      [ECAPA (nossa)], [1,36 / 1,08], [0,68 / 1,11],
      [ResNet-34 pré], [0,72 / 3,02], [0,39 / 1,26],
      [ResNet-34 ajust.], [0,61 / 4,39], [0,20 / 0,66],
      table.hline(),
    )
    - ResNet-34: *3× menor* e EER menor.
    - ECAPA: FMR mais *estável* no limiar entre bases.
  ]
]

#slide[Robustez e ablação][
  #cols[
    #fig("results/paper/robustness.png", width: 100%)
    #text(size: 15pt)[Ruído branco 0 dB: EER 8,5%. _Babble_ 0 dB: *39,1%* (FNMR de 87%).]
  ][
    #set text(size: 16pt)
    #table(
      columns: 2, stroke: none,
      table.hline(),
      table.header([*Ablação*], [*EER*]),
      table.hline(stroke: 0.5pt),
      [completo], [1,36%],
      [sem VAD], [1,10%],
      [sem CMN], [1,81%],
      [sem VAD e CMN], [2,36%],
      table.hline(),
    )
    - CMN ajuda.
    - VAD atrapalha um pouco: as frases já vêm aparadas, e ele corta fala útil.
  ]
]

#slide[Erros: falsos _matches_][
  - VCTK: 52 FM. *98%* entre pessoas do mesmo sexo (vs. 53% de todos os impostores) e *71%* do mesmo sotaque (vs. 19%).
  - Um par de locutoras (p228 × p244, jovens inglesas) gera *17 dos 52* FM.
  - LibriSpeech: FMR de 2,1% entre mesmo sexo vs. 0,16% entre sexos diferentes.
  #destaque[*Hipótese:* o _embedding_ separa bem características globais (sexo, sotaque), mas tem pouca margem para vozes parecidas dentro do mesmo grupo.]
  - Falsos _non-matches_ do VCTK: só 3, todos logo abaixo do limiar; dois com troca de microfone.
]

#slide[Erros: o leitor que interpreta personagens][
  #fig("results/paper/speaker_6209.png", width: 92%)
  #set text(size: 16pt)
  - 13 dos 18 FNM do LibriSpeech: leitor 6209, capítulo 34601 de _O Homem que Ri_.
  - Abaixo do limiar: o *monólogo do personagem Ursus*. O trecho 30, no meio, é narração e passa.
  - A F0 fica na mesma faixa da narração → não é outro leitor. *O leitor muda a voz*, e o modelo vê outra pessoa.
]

#slide[Eficiência][
  #table(
    columns: 3, stroke: none,
    table.hline(),
    table.header([], [*ECAPA-TDNN*], [*ResNet-34 (D8)*]),
    table.hline(stroke: 0.5pt),
    [Parâmetros], [20,8 M], [6,6 M],
    [Tamanho], [79,2 MB], [25,4 MB],
    [_Embedding_], [192-d], [256-d],
    [Tempo], [107 ms (CPU, VCTK)], [≈ 31 ms (GPU T4)],
    table.hline(),
  )
  #v(0.5em)
  - Os tempos não são comparáveis (CPU × GPU), mas o tamanho é.
  - *Aqui, 3× mais parâmetros não compraram EER menor.*
]

#slide[Conclusões][
  - ECAPA-TDNN pré-treinada, sem ajuste: EER *1,36%* no VCTK e *0,64–0,68%* no LibriSpeech.
  - Limiar do VCTK manteve a FMR perto de 1% nas duas bases.
  - Aguenta ruído branco, sofre com várias vozes ao fundo.
  - Erros: vozes parecidas do mesmo sexo e sotaque, e um leitor que interpreta personagens.
  - *Limitações:* sem _fine-tuning_ nem várias _seeds_; teste do VCTK pequeno (17 locutores).
  - *Próximos passos:* _fine-tuning_ checando a FMR fora da base, treino com _babble_.
  #v(0.6em)
  #align(center, text(size: 22pt)[*Obrigada!* Perguntas?])
]
