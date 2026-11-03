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
- ⏳ Contatar Dupla 8 (protocolos do Exp. B)

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
| **Exp. A** — baseline na base principal | 🔶 | Rodar pipeline zero-shot no VCTK e gerar `scores.csv` + `metrics.json` |
| **Exp. B** — 2 CNNs × 2 datasets | 🔴 | **Depende da Dupla 8**: compartilhar os mesmos `genuine_pairs.csv`/`impostor_pairs.csv` (o código já prevê isso em `generate_protocols.py`) |
| **Exp. C** — cross-dataset (VCTK→LibriSpeech, sem fine-tuning) | 🔴 | Modo `--cross-dataset` no `evaluate.py` + rodar no LibriSpeech |
| **Exp. D** — robustez a ruído (mín. 4 níveis) | 🔴 | Script `robustness.py` **não existe** (baixar ruído: NOISE-ESC-58 / MUSAN / gerador de ruído branco) |
| **Exp. E** — estudo de ablação (com/sem VAD, CMN, etc.) | 🔴 | Script de ablação **não existe** |
| Fine-tuning + 3 seeds (média ± desvio) | 🔴 | `train.py` existe mas está **quebrado** (bugs C2–C8) |
| Análise de ≥3 FM + ≥3 FNM | 🔶 | Função existe (`analyze_errors`) — falta rodar e **discutir as causas** no artigo |
| FTA (Failure to Acquire) | 🔶 | Contagem existe no preprocess/extract — falta reportar na tabela final |
| Eficiência (parâmetros, tamanho, ms/utt, hardware) | 🔶 | Coletado em `inference_report.json` — falta preencher a tabela do README/artigo |

### 4.2 Resultados (diretório `results/` está **vazio**)

- [ ] `results/{vctk,librispeech}/scores.csv`
- [ ] `results/{vctk,librispeech}/metrics.json`
- [ ] `roc.png`, `distributions.png`
- [ ] Tabela preenchida no README (EER, AUC, FMR, FNMR, TAR@FAR=1%, tempo, hardware)
- [ ] Arquivos finais para consolidado da disciplina (§25 do enunciado)

### 4.3 Dados

- [ ] Baixar **VCTK** → `data/vctk/wav48_silence_trimmed/` (https://datashare.ed.ac.uk/handle/10283/3443) — ⏳ zip baixando em `data/vctk/VCTK-Corpus-0.92.zip`
- [ ] Baixar **LibriSpeech train-clean-100** → `data/librispeech/train-clean-100/` (https://www.openslr.org/12) — ⏳ `data/librispeech/train-clean-100.tar.gz` baixando
- [ ] Baixar/gerar corpus de ruído para o Exp. D
- [ ] Confirmar com a **Dupla 8** a troca dos protocolos de pares

### 4.4 Entrega formal

- [ ] **Artigo científico 6–8 páginas** (template SBC/IEEE) — não iniciado
- [ ] **Apresentação 10 min** + arguição 5 min — não iniciada
- [ ] Repositório Git: `git init`, remote, e **tag `final-v1.0`** com hash no README — **hoje nem é um repo git**
- [x] `LICENSE` (MIT) — criado em 03/11/2026
- [ ] Preencher: hardware, data de entrega, versão, hash do commit (README já com data 18/25/11)

---

## 5. Plano de ação até a entrega

**Hoje: 03/11/2026 · Entregas: 18/11 ou 25/11/2026 → janela de 15–22 dias**

### Fase 0 — Fundação (dias 1–3: 03–05/11)
1. `git init` + primeiro commit + remote (GitHub) 🔄 (03/11: commit local + `gh repo create` em andamento)
2. Corrigir bugs críticos **C1–C10** (prioridade: `model.py`, `train.py`, `evaluate.py`) ✅ (+ C11 torchaudio/torchcodec)
3. Baixar VCTK e LibriSpeech ⏳ (downloads em background)
4. Corrigir README (caminhos `src/`, flags, scripts inexistentes) ✅
5. Contatar Dupla 8 para alinhar protocolos compartilhados (Exp. B) 🔲

### Fase 1 — Experimento A: baseline (dias 4–6: 06–08/11)
1. `generate_protocols.py` para VCTK **+ gerar pares de validação** (corrige P1) 🔲
2. `extract_embeddings.py` no VCTK (zero-shot) 🔲
3. `evaluate.py` com **threshold congelado na validação** 🔲
4. Preencher `results/vctk/` (scores.csv, metrics.json, roc.png, distributions.png) 🔲
5. Validação do C1: confirmar se o modelo é realmente pré-treinado (senão, declarar treino do zero) 🔲

### Fase 2 — Experimentos C, D, E (dias 7–11: 09–13/11)
1. **Exp. C:** protocolo LibriSpeech + modo cross-dataset (modelo congelado, sem fine-tuning) 🔲
2. **Exp. D:** criar `robustness.py` — ruído aditivo (white, babble, music) × ≥4 SNRs (ex.: 20/10/5/0 dB + original) 🔲
3. **Exp. E:** criar script de ablação (com/sem VAD, com/sem CMN, com/sem normalização de amplitude) 🔲
4. Consolidar protocolos com a Dupla 8 (Exp. B) 🔲

### Fase 3 — Fine-tuning e reprodutibilidade (dias 10–14: 12–16/11, paralelo)
1. Consertar `train.py` e rodar fine-tuning (encoder congelado, AAM-Softmax) 🔲
2. 3 seeds (42, 123, 2026) → média ± desvio-padrão de EER 🔲
3. Reavaliar Exp. A com o melhor checkpoint (comparar zero-shot vs fine-tuned) 🔲
4. Preencher eficiência: parâmetros, tamanho do modelo, ms/utt, hardware 🔲

### Fase 4 — Artigo e apresentação (dias 13–20: 15–22/11)
1. Esboço do artigo (8 seções exigidas no enunciado §21) 🔲
2. Figuras obrigatórias: pipeline real, espectrograma, distribuição de scores com threshold, ROC, robustez, ablação 🔲
3. Análise de erros com hipóteses técnicas (ruído, canal, duração, condição vocal) 🔲
4. Limitações e comparação com a Dupla 8 🔲
5. Slides (problema → dataset → pipeline → CNN → embedding → matching → protocolo → resultados → cross-dataset → robustez → erros → conclusões) 🔲
6. **Congelamento:** tag `final-v1.0` + hash no README + submeter scores.csv/metrics.json 🔲

---

## 6. Riscos

| Risco | Impacto | Mitigação |
|---|---|---|
| Pesos SpeechBrain não transferem para o modelo custom (C1) | Alto — inviabiliza o "zero-shot" | Mapear state_dict para as camadas equivalentes **ou** declarar no artigo que o treino é do zero (resultado negativo é válido, §29.18) |
| Sem GPU / datasets grandes demais | Médio | Usar subconjunto do VCTK (ex.: 40 falantes) já no protocolo, mantendo splits por identidade |
| Dupla 8 atrasar os protocolos | Alto — trava Exp. B | Gerar protocolos o quanto antes e enviar |
| Prazo 18/11 | Alto | Priorizar Fases 0–1 + artigo; fine-tuning e ablação podem ficar para a semana 2 se necessário |

---

## 7. Checklist resumido da entrega

- [ ] Pipeline reproduzível ponta a ponta (1 comando por etapa, sem `src/`)
- [ ] Exp. A (VCTK) com métricas completas
- [ ] Exp. B (matriz 2 CNN × 2 datasets, com Dupla 8)
- [ ] Exp. C (cross-dataset sem fine-tuning)
- [ ] Exp. D (robustez, ≥4 níveis)
- [ ] Exp. E (ablação)
- [ ] ≥3 FM + ≥3 FNM analisados
- [ ] scores.csv + metrics.json no repositório
- [ ] Artigo PDF 6–8 páginas
- [ ] Apresentação 10 min
- [ ] Tag `final-v1.0` + hash no README
