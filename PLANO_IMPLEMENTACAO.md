# Plano de Implementação & Status do Projeto — Dupla 7

**Projeto:** Speaker Verification com ECAPA-TDNN (Modalidade Voz)
**Disciplina:** Biometria — UTFPR Campus Ponta Grossa
**Bases:** VCTK (principal) + LibriSpeech train-clean-100 (cross-dataset)
**Enunciado:** `Projeto_Final_Disciplina_Biometria.pdf` / `pdf_content.txt`
**Data de entrega:** 18 ou 25/11/2026 (artigo PDF + link do repo + apresentação 10 min)
**Documento gerado em:** 03/11/2026

> Este arquivo passa a ser o **plano de implementação oficial** da dupla.
> Atualizar as seções de status conforme os itens forem concluídos.

---

## 0. Status atualizado (03/11/2026 — fim do dia 1)

### Fase 0 — concluída ✅ (exceto downloads e contato com a Dupla 8)

- ✅ **C1–C10 corrigidos**; `model.py` reescrito sobre o backbone oficial do
  SpeechBrain (`speechbrain.lobes.models.ECAPA_TDNN`), carregando **231/231
  tensores** de `speechbrain/spkrec-ecapa-voxceleb` — medido: **20.767.552
  parâmetros / ~79 MB** (números antigos 14,8M/6,6M estavam errados)
- ✅ `AAMSoftmax.forward` retorna `(loss, logits)`; pares de **validação e
  teste** gerados (`val_*_pairs.csv`, `test_*_pairs.csv`); threshold
  congelado na validação (anti-leak, enunciado §13)
- ✅ **Novo bug C11 (ambiente) resolvido:** `torchaudio 2.11` delega `load`
  ao torchcodec (não instalado) → leitura via `soundfile` em `preprocess.py`
- ✅ **Smoke test ponta-a-ponta** em dados sintéticos (20 falantes, 120 wavs):
  `generate_protocols` → `extract_embeddings` (120/120, 51 ms/utt CPU) →
  `evaluate` (EER/ROC/metrics.json/error_analysis) → `train` (1 época,
  seed 42, checkpoint salvo) → extração com checkpoint — **tudo passou**
- ✅ Ambiente: `torchaudio 2.11.1`, `speechbrain 1.1.2`, `librosa 1.0`,
  `soundfile`, `webrtcvad-wheels` (o `webrtcvad` puro não compila no
  Windows/Py3.13)
- ✅ `README.md` reescrito (sem `src/`, flags reais, scripts reais, data de
  entrega, 20.767.552 params, nota `PYTHONUTF8` no Windows)
- ✅ `requirements.txt` (`webrtcvad-wheels`), `.gitignore` (data/** + gitkeep,
  venv), `LICENSE` (MIT), `config.yaml` (`n_parameters: 20767552`)
- ⏳ **Downloads em background:** VCTK zip + LibriSpeech tar.gz (verificar
  tamanho/extração)
- [ ] Contatar Dupla 8 (protocolos do Exp. B) — movido para a **Fase 5**

### Repo Git

- ✅ `git init` + commit `d1a7d22` em 03/11/2026
- ✅ GitHub público: <https://github.com/mauricegss/projeto-biometria>
  (branch `main`, remote `origin`)
- 🔲 Tag `final-v1.0` apenas no commit de entrega

---

## 0.1 Status — Fase 1 / Experimento A (12/11/2026)

- ✅ Downloads verificados (VCTK zip 133.135 entradas; LibriSpeech 29.966) e **extraídos**
  - `data/vctk/wav48_silence_trimmed/` (109 falantes, 88.328 utterances)
  - `data/librispeech/train-clean-100/` (251 falantes)
- ✅ Protocolos VCTK seed 42: 76/16/17 falantes; val 160+5000 pares; test 170+5000 pares
- ✅ Novo bug **C12 (crítico):** `_cmn_normalize` fazia **z-score** (média+desvio) e
  **colapsava os embeddings** do ECAPA pré-treinado (cosine ≈ 0,98 entre falantes
  distintos → EER ≈ 40%). O SpeechBrain usa `InputNormalization(std_norm=False)`
  = só subtração de média. Corrigido; embeddings reextraídos.
- ✅ Extração zero-shot com `--files_csv` (só os 13.917 arquivos dos pares):
  **13.595 OK / 322 FTA** (áudio < 1 s pós-VAD), 107,5 ms/utt em CPU
- ✅ **Exp. A concluído** (`results/vctk/`):

| Métrica | Valor (teste) |
|---|---|
| EER | **1,36%** |
| Threshold (congelado na val) | 0,353 (EER val = 0,353) |
| FMR / FNMR @ threshold | 1,08% / 1,81% |
| AUC-ROC | 0,9991 |
| TAR @ FAR=1% | 96,99% |
| d-prime | 4,78 |
| N pares (test) | 166 genuínos / 4799 impostores |

- ⏳ Pendências da Fase 1: análise de ≥3 FM + ≥3 FNM no artigo (dados já em
  `results/vctk/error_analysis.json`)

---

## 1. Onde está o plano de implementação?

**Não havia.** A análise do repositório mostrou que o planejamento estava disperso em três fontes:

| Fonte | Conteúdo |
|---|---|
| `Projeto_Final_Disciplina_Biometria.pdf` | Requisitos, pipeline obrigatório, experimentos A–E, métricas, entrega |
| `README.md` | Comandos do pipeline (parcialmente desatualizados em relação ao código) |
| Código (`*.py`, `config.yaml`) | Implementação efetiva |

Este documento centraliza tudo: **o que foi feito, o que está quebrado, o que falta e o plano até a entrega**.

---

## 2. O que foi FEITO (inventário do código)

### 2.1 Arquivos existentes (prontos / funcionais)

| Arquivo | Status | O que faz |
|---|---|---|
| `preprocess.py` | ✅ Funcional | Leitura WAV/FLAC, mono 16 kHz, VAD (webrtcvad + fallback por energia), Log-Mel 80 bins (25/10 ms, FFT 512), CMN por utterance, CLI, geração de PNG de espectrograma, log de FTA |
| `generate_protocols.py` | ✅ Funcional | Descobre falantes (VCTK/LibriSpeech), split **por identidade** 70/15/15 (seed 42), gera `train.csv`, `validation.csv`, `test.csv`, `genuine_pairs.csv`, `impostor_pairs.csv` |
| `extract_embeddings.py` | ⚠️ Quase | Extração em batch com padding, L2-norm, medição de tempo de inferência, hardware info, relatório `inference_report.json`, índice `embeddings_index.csv` |
| `model.py` | ⚠️ Quase | ECAPA-TDNN completo (SEBlock, Res2Block, MFA, ASP, AAM-Softmax), teste `__main__` com asserts. **Mas os pesos SpeechBrain não são transferidos** (ver §3) |
| `evaluate.py` | ⚠️ Quase | Scores de cosseno, EER com interpolação, FMR/FNMR, ROC/AUC, TAR@FAR=1%, d-prime, gráficos ROC + distribuição, `scores.csv`, `metrics.json`, análise de 3 FM + 3 FNM |
| `config.yaml` | ✅ | Configuração central (pré-processamento, modelo, treino, matching) |
| `README.md` | ⚠️ | Completo como documentação, **mas comandos desatualizados** (ver §3) |
| `requirements.txt` | ✅ | torch, torchaudio, speechbrain, librosa, webrtcvad, sklearn, seaborn etc. |

### 2.2 O que o pipeline já cobre dos requisitos do enunciado

- ✅ Pipeline: dado bruto → controle de qualidade (FTA) → pré-processamento → VAD (segmentação) → normalização (CMN) → CNN → embedding → matching (cosseno) → score → threshold → decisão
- ✅ Embedding biométrico de 192-d (não é classificador de identidade)
- ✅ Matching por similaridade de cosseno entre embeddings L2-normalizados (métrica padrão pedida)
- ✅ Separação treino/validação/teste **mutuamente exclusiva por identidade** (sem identity leakage)
- ✅ Pares genuínos e impostores com seed fixa (42)
- ✅ Métricas: EER, FMR, FNMR, ROC, AUC, distribuição de scores, TAR@FAR=1%
- ✅ Análise de erros (3 falsos matches + 3 falsos non-matches)
- ✅ Seeds de reprodutibilidade (42, 123, 2026)
- ✅ Documentação da CNN no README (arquitetura, origem dos pesos, dimensão, perda, parâmetros)

---

## 3. O que está QUEBRADO (bugs conhecidos — corrigir antes de qualquer execução)

### 🔴 Críticos (impedem a execução)

| # | Arquivo | Problema |
|---|---|---|
| C1 | `model.py:559-574` | `load_pretrained_ecapa()` carrega o SpeechBrain e **descarta os pesos** — o modelo fica com pesos aleatórios. A afirmação "pré-treinado em VoxCeleb2" do README é **falsa** na prática |
| C2 | `train.py:513-519` | `AudioPreprocessor(sample_rate=..., n_mels=...)` — o construtor aceita apenas `dict config` → `TypeError` |
| C3 | `train.py:564-568` | `ECAPATDNN(..., emb_dim=...)` — parâmetro correto é `embedding_dim` → `TypeError` |
| C4 | `train.py:573-578` | `AAMSoftmax(in_features=..., num_classes=...)` — parâmetros corretos são `embedding_dim`, `n_classes` → `TypeError` |
| C5 | `train.py:276` | `loss, logits = classifier(...)` — `AAMSoftmax.forward()` retorna **só a loss** → `ValueError` |
| C6 | `train.py:562` | `load_pretrained_ecapa(pretrained_path, num_classes=...)` — assinatura incompatível |
| C7 | `train.py:460` | `_score_pairs()` espera colunas `file1`/`file2`; o protocolo gera `amostra_1`/`amostra_2` → `KeyError` |
| C8 | `train.py:508` | `protocols_dir = data/vctk/protocols/vctk` — os protocolos ficam em `protocols/vctk/` → `FileNotFoundError` |
| C9 | `evaluate.py:620` | `load_embeddings()` lê coluna `arquivo`; `extract_embeddings.py` grava `file_path` → `KeyError` |
| C10 | `evaluate.py:624-625` | `embedding_path` já é relativo ao cwd; evaluate pré-anexa `embeddings_dir/dataset/` → caminho duplicado → `FileNotFoundError` |

### 🟠 Protocolares (violam o enunciado)

| # | Problema |
|---|---|
| P1 | `generate_protocols.py` gera pares **apenas do teste**; não há pares de validação. `evaluate.py:863-875` cai no fallback `protocols/vctk/genuine_pairs.csv` = **pares de teste** → threshold calculado no teste (**vazamento de informação**, item 13 do enunciado) |
| P2 | `evaluate.py` não implementa o Experimento C "modelo congelado sem novo fine-tuning" como modo explícito (`--cross-dataset` inexistente) |
| P3 | Threshold: `config.yaml` diz `threshold_from: validation`, mas sem pares de validação o fallback é o EER do teste |

### 🟡 Médios (inconsistências de documentação/configuração)

| # | Problema |
|---|---|
| M1 | README manda rodar `python src/*.py` — **não existe pasta `src/`** (arquivos estão na raiz) |
| M2 | README usa flags inexistentes: `--protocol`, `--output-dir`, `--cross-dataset`, `--noise-types`, `--snr-levels`, `--components`, `--freeze-encoder` etc. (além de `--embeddings-dir` com hífen, que o argparse não reconhece) |
| M3 | README referencia scripts **inexistentes**: `src/evaluate_robustness.py`, `src/ablation_study.py`, `src/finetune.py` |
| M4 | README: "~14,8M parâmetros" vs `model.py` docstring: "~6.6M" vs `config.yaml`: 14800000 — padronizar (o valor real com channels=512 é ~14,7M) |
| M5 | `config.yaml` (`robustness`) só tem 2 tipos de ruído e 3 SNRs — o enunciado pede **mínimo 4 níveis** (original, baixa, média, alta degradação) |
| M6 | README tem "Data de Entrega: A definir" e commit/tag de entrega por preencher |
| M7 | `train.py:344-348` tem bloco de código morto (`pass`) no `validate()` |
| M8 | Sem `LICENSE` no repo, embora o README cite um |

---

## 4. O que FALTA (mapa contra o enunciado)

### 4.1 Código / experimentos

| Item do enunciado | Status | Falta |
|---|---|---|
| **Exp. A** — baseline na base principal | ✅ | Concluído 12/11/2026 — EER 1,36% em `results/vctk/` |
| **Exp. B** — 2 CNNs × 2 datasets | 🔴 | **Depende da Dupla 8** — isolado na Fase 5; compartilhar os mesmos `genuine_pairs.csv`/`impostor_pairs.csv` |
| **Exp. C** — cross-dataset (VCTK→LibriSpeech, sem fine-tuning) | ✅ | Concluído 13/11 — threshold VCTK (0,353) → LibriSpeech; EER **0,64%** em `results/librispeech/` |
| **Exp. D** — robustez a ruído (mín. 4 níveis) | ✅ | Concluído 13/11 — `evaluate_robustness.py`; white+babble × 20/10/5/0 dB; `results/vctk/robustness/` |
| **Exp. E** — estudo de ablação (com/sem VAD, CMN) | ✅ | Concluído 13/11 — `ablation_study.py`; `results/vctk/ablation/` |
| Fine-tuning + 3 seeds (média ± desvio) | 🔶 | `train.py` funcionando (smoke OK) — falta rodar no VCTK real (Fase 3) |
| Análise de ≥3 FM + ≥3 FNM | 🔶 | Função existe (`analyze_errors`) — falta rodar e **discutir as causas** no artigo |
| FTA (Failure to Acquire) | 🔶 | Contagem existe no preprocess/extract — falta reportar na tabela final |
| Eficiência (parâmetros, tamanho, ms/utt, hardware) | 🔶 | Coletado em `inference_report.json` — falta preencher a tabela do README/artigo |

### 4.2 Resultados

- [x] `results/vctk/scores.csv` + `val_scores.csv` + `metrics.json` + `roc.png` + `distributions.png` (Exp. A, 12/11)
- [x] `results/librispeech/` (Exp. C — 13/11)
- [x] Resultados Exp. D (`results/vctk/robustness/`) e E (`results/vctk/ablation/`) — 13/11
- [x] Tabela preenchida no README (Exp. A + C + D + E)
- [ ] Arquivos finais para consolidado da disciplina (§25 do enunciado)

### 4.3 Dados

- [x] Download **VCTK** → `data/vctk/wav48_silence_trimmed/` — extraído em 12/11/2026
- [x] Download **LibriSpeech train-clean-100** → `data/librispeech/train-clean-100/` — extraído em 12/11/2026
- [x] Baixar VCTK e LibriSpeech → extraídos em 12/11/2026
- [x] Corpus de ruído para Exp. D — white (gerado) + babble (mistura de utterances VCTK), 13/11/2026
- [ ] Confirmar com a **Dupla 8** a troca dos protocolos de pares

### 4.4 Entrega formal

- [ ] **Artigo científico 6–8 páginas** (template SBC/IEEE) — versão sem Exp. B na Fase 4; final na Fase 5
- [ ] **Apresentação 10 min** + arguição 5 min — não iniciada
- [x] Repositório Git: `git init` + remote GitHub (`mauricegss/projeto-biometria`, público) — 03/11/2026
- [ ] Tag `final-v1.0` com hash no README — **Fase 5** (congelamento final com Exp. B)
- [x] `LICENSE` (MIT) — criado em 03/11/2026
- [x] Data de entrega no README (18 ou 25/11/2026) e eficiência do Exp. A preenchidas

---

## 5. Plano de ação até a entrega

**Hoje: 12/11/2026 · Entregas: 18/11 ou 25/11/2026**

> **Regra de dependência:** tudo que exige a **Dupla 8** (Exp. B e o
> congelamento final do documento com os resultados B) está isolado na
> **Fase 5**. As Fases 2–4 avançam 100% sem eles.

### Fase 0 — Fundação (03–05/11) — ✅ concluída
1. `git init` + commit + remote GitHub ✅
2. Corrigir bugs críticos **C1–C11** ✅
3. Baixar VCTK e LibriSpeech ✅ (extraídos em 12/11)
4. Corrigir README / LICENSE / requirements ✅

### Fase 1 — Experimento A: baseline VCTK (06–12/11) — ✅ concluída
1. Protocolos com pares de validação (anti-leak) ✅
2. Embeddings zero-shot (13.595) ✅
3. Avaliação threshold congelado na validação ✅ → EER **1,36%**
4. Bug C12 (CMN z-score colapsava embeddings) corrigido ✅

### Fase 2 — Experimentos C, D, E (independente da Dupla 8) — ✅ concluída (13/11)
1. **Exp. C:** protocolos LibriSpeech + extração (11.688 emb.) + flag
   `--threshold_from` no `evaluate.py` (modelo congelado, sem fine-tuning)
   ✅ → threshold VCTK 0,353 → LibriSpeech; **EER 0,64%**
2. **Exp. D:** `evaluate_robustness.py` — white + babble × 20/10/5/0 dB
   + original (9 condições); threshold congelado na val limpa ✅
   → EER 1,36% (limpo) → 8,54% (white@0) → 39,06% (babble@0)
3. **Exp. E:** `ablation_study.py` — full / no_vad / no_cmn / no_vad_no_cmn ✅
   → EER 1,36 / 1,10 / 1,81 / 2,36%

### Fase 3 — Fine-tuning e reprodutibilidade (independente) 🔲
1. Rodar `train.py` no VCTK real (encoder congelado, AAM-Softmax) 🔲
2. 3 seeds (42, 123, 2026) → média ± desvio-padrão de EER 🔲
3. Reavaliar Exp. A com o melhor checkpoint (zero-shot vs fine-tuned) 🔲
4. Consolidar eficiência: parâmetros, tamanho, ms/utt, hardware 🔲

### Fase 4 — Artigo e apresentação, versão sem Exp. B (independente) 🔲
1. Esboço do artigo (8 seções do enunciado §21) com resultados A, C, D, E 🔲
2. Figuras: pipeline, espectrograma, distribuição+threshold, ROC,
   robustez, ablação 🔲
3. Análise de ≥3 FM + ≥3 FNM com hipóteses técnicas 🔲
4. Slides da apresentação 🔲
5. Seção do Exp. B deixada como *“protocolos em consolidação com a
   Dupla 8”* até a Fase 5 🔲
   > Obs.: a **entrega final** do documento só é congelada na Fase 5
   > (precisa dos resultados B) — aqui sai a versão completa possível
   > sem a outra dupla.

### Fase 5 — Dupla 8: Exp. B + congelamento final (bloqueada) 🔲
1. Contatar Dupla 8 e trocar `genuine_pairs.csv` / `impostor_pairs.csv` 🔲
2. Rodar a matriz **2 CNN × 2 datasets** com os protocolos compartilhados 🔲
3. Inserir resultados do Exp. B no artigo e nos slides 🔲
4. Limitações e comparação com a Dupla 8 🔲
5. **Congelamento:** tag `final-v1.0` + hash no README + submeter
   `scores.csv`/`metrics.json` 🔲

---

## 6. Riscos

| Risco | Impacto | Mitigação |
|---|---|---|
| Dupla 8 atrasar/sumir | Alto — só trava a **Fase 5** (Exp. B + tag final) | Fases 2–4 independentes; se não responderem até 16/11, entregar sem Exp. B e justificar no artigo |
| Sem GPU / datasets grandes | Médio | CPU com `--files_csv` (só pares) já resolveu Exp. A |
| Prazo 18/11 | Alto | Priorizar Fases 2–4; Fase 5 só se houver resposta da Dupla 8 |

---

## 7. Checklist resumido da entrega

**Independente (Fases 2–4):**
- [x] Pipeline reproduzível ponta a ponta (1 comando por etapa, sem `src/`)
- [x] Exp. A (VCTK) com métricas completas — EER 1,36% (12/11/2026)
- [x] Exp. C (cross-dataset sem fine-tuning) — EER 0,64% (13/11/2026)
- [x] Exp. D (robustez, ≥4 níveis) — 9 condições (13/11/2026)
- [x] Exp. E (ablação) — 4 variantes (13/11/2026)
- [ ] Fine-tuning 3 seeds (média ± desvio)
- [ ] ≥3 FM + ≥3 FNM analisados
- [x] scores.csv + metrics.json no repositório (`results/vctk/`, `results/librispeech/`)
- [ ] Artigo PDF 6–8 páginas (versão sem Exp. B)
- [ ] Apresentação 10 min (versão sem Exp. B)

**Depende da Dupla 8 (Fase 5):**
- [ ] Exp. B (matriz 2 CNN × 2 datasets, protocolos compartilhados)
- [ ] Resultados B inseridos no artigo e slides
- [ ] Tag `final-v1.0` + hash no README (congelamento final)
