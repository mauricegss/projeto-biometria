# Speaker Verification com ECAPA-TDNN — Dupla 7

## Projeto Final — Biometria | UTFPR Campus Ponta Grossa

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Informações do Projeto

| Campo | Valor |
|---|---|
| **Modalidade Biométrica** | Voz (Speaker Verification) |
| **Dupla** | 7 |
| **Disciplina** | Biometria |
| **Instituição** | UTFPR — Campus Ponta Grossa |
| **Data de Entrega** | 18 ou 25/11/2026 |
| **Arquitetura** | ECAPA-TDNN |
| **Dataset Principal** | VCTK Corpus 0.92 |
| **Segunda Base** | LibriSpeech train-clean-100 |

---

## Arquitetura e Modelo

| Componente | Detalhe |
|---|---|
| **Arquitetura** | ECAPA-TDNN (Desplanques et al., Interspeech 2020) |
| **Implementação** | Backbone oficial do SpeechBrain (`ECAPA_TDNN`) |
| **Origem dos pesos** | `speechbrain/spkrec-ecapa-voxceleb` (VoxCeleb1+2), 231/231 tensores |
| **Camadas congeladas** | Encoder (opcional no fine-tuning) |
| **Função de perda** | AAM-Softmax (margin=0.2, scale=30) |
| **Dimensão do embedding** | 192 |
| **Número de parâmetros** | **20.767.552** (~79 MB) |
| **Matching** | Similaridade de cosseno entre embeddings L2-normalizados |

> **Nota:** o pesos é baixado do Hugging Face na primeira execução (requer internet).
> O número de parâmetros acima é o medido na prática (`python model.py`), e não o
> estimado de literatura.

---

## Pré-processamento

| Etapa | Configuração |
|---|---|
| **Leitura** | `soundfile` (WAV/FLAC — `torchaudio.load` exige torchcodec ≥ 2.9) |
| **Resample** | 16 kHz mono |
| **VAD WebRTC** | Agressividade 2 (fallback por energia) |
| **Log-Mel Spectrogram** | 80 mel-banks, win=25 ms, hop=10 ms, n_fft=512 |
| **Normalização** | Por utterance — CMN (subtração de média; sem divisão por desvio) |
| **Duração mínima** | 1,0 s após VAD (arquivos mais curtos são FTA) |

---

## Dados

### VCTK Corpus 0.92 (base principal)

> Download: <https://datashare.ed.ac.uk/handle/10283/3443>
> ZIP direto (~11 GB): `https://datashare.ed.ac.uk/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/download`

110 falantes nativos de inglês com diferentes sotaques. Extraia para:

```
data/vctk/
└── wav48_silence_trimmed/
    ├── p225/
    │   ├── p225_001_mic1.flac
    │   └── ...
    └── ...
```

### LibriSpeech train-clean-100 (segunda base)

> Download: <https://www.openslr.org/12> (~6,3 GB: `train-clean-100.tar.gz`)

~100 horas de fala de audiolivros. Extraia para:

```
data/librispeech/
└── train-clean-100/
    ├── 19/
    │   ├── 198/
    │   │   ├── 19-198-0000.flac
    │   │   └── ...
    │   └── ...
    └── ...
```

> Os downloads ficam em `data/` (ignorado pelo Git; apenas `.gitkeep` é versionado).

---

## Instalação

```bash
# 1. Clone o repositório
git clone <url-do-repositorio>
cd projeto-biometria

# 2. Ambiente virtual
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Dependências
pip install -r requirements.txt
```

Requisitos: **Python ≥ 3.10** (testado em 3.13), CPU funciona; GPU opcional (CUDA).

> **Windows (console):** use `PYTHONUTF8=1` antes de rodar os scripts —
> o console cp1252 quebra com símbolos como `→`/`✔` nas saídas:
> ```powershell
> $env:PYTHONUTF8 = "1"
> ```

> **Nota `webrtcvad`:** o pacote `webrtcvad` puro não compila no Windows com
> Python 3.13; usamos `webrtcvad-wheels` (mesma API, binário pré-compilado).

---

## Estrutura do repositório

```
.
├── model.py                # ECAPA-TDNN (SpeechBrain) + AAM-Softmax
├── preprocess.py           # resample, VAD, Log-Mel, CMN
├── generate_protocols.py   # pares genuínos/impostores (val + test)
├── extract_embeddings.py   # inferência em lote (+ índice CSV)
├── train.py                # fine-tuning multi-seed
├── evaluate.py             # EER, ROC, FMR/FNMR, threshold na validação
├── config.yaml             # configuração central
├── requirements.txt
├── PLANO_IMPLEMENTACAO.md  # status e plano até a entrega
└── data/                   # datasets (não versionados)
```

---

## Reprodução dos Experimentos

### 1. Gerar protocolos

```bash
python generate_protocols.py --dataset vctk \
    --data_dir data/vctk/wav48_silence_trimmed \
    --output_dir protocols/ --seed 42

python generate_protocols.py --dataset librispeech \
    --data_dir data/librispeech/train-clean-100 \
    --output_dir protocols/ --seed 42
```

Saídas em `protocols/<dataset>/`: `train_pairs.csv`, `val_genuine_pairs.csv`,
`val_impostor_pairs.csv`, `test_genuine_pairs.csv`, `test_impostor_pairs.csv`.

### 2. Extrair embeddings

```bash
# Pesos pré-treinados (zero-shot) — só os arquivos citados nos pares do protocolo
python extract_embeddings.py --dataset vctk \
    --data_dir data/vctk/wav48_silence_trimmed \
    --output_dir embeddings/ --config config.yaml \
    --files_csv protocols/vctk/val_genuine_pairs.csv \
                protocols/vctk/val_impostor_pairs.csv \
                protocols/vctk/test_genuine_pairs.csv \
                protocols/vctk/test_impostor_pairs.csv

# A partir de um checkpoint fine-tuned (opcional)
python extract_embeddings.py --dataset vctk \
    --data_dir data/vctk/wav48_silence_trimmed \
    --output_dir embeddings_ft/ \
    --checkpoint checkpoints/best_model_seed42.pt --config config.yaml
```

### 3. Avaliar (Experimento A — baseline VCTK)

```powershell
python evaluate.py --dataset vctk
```

### 3b. Experimentos C, D e E (Fase 2)

```powershell
# Exp. C — cross-dataset: threshold do VCTK aplicado ao LibriSpeech
python generate_protocols.py --dataset librispeech --data_dir data/librispeech/train-clean-100
python extract_embeddings.py --dataset librispeech --data_dir data/librispeech/train-clean-100 `
  --files_csv protocols/librispeech/test_genuine_pairs.csv protocols/librispeech/test_impostor_pairs.csv
python evaluate.py --dataset librispeech --threshold_from results/vctk/metrics.json

# Exp. D — robustez a ruído aditivo (≥4 níveis)
python evaluate_robustness.py --dataset vctk --noise_types white babble --snr_levels 20 10 5 0

# Exp. E — ablação do pré-processamento
python ablation_study.py --dataset vctk --variants full no_vad no_cmn no_vad_no_cmn --reuse_full
```

### 4. Fine-tuning (opcional — Fase 3)

```bash
python evaluate.py --dataset vctk \
    --embeddings_dir embeddings/ \
    --protocols_dir protocols/ \
    --results_dir results/ --config config.yaml
```

O threshold é **congelado na validação** (EER) e só então aplicado ao teste —
sem vazamento do conjunto de teste. Métricas em `results/<dataset>/metrics.json`,
gráficos ROC/distribuição em PNG.

### 4. Fine-tuning (opcional)

```bash
python train.py --dataset vctk \
    --data_dir data/vctk/wav48_silence_trimmed \
    --protocols_dir protocols/ \
    --output_dir checkpoints/ \
    --epochs 20 --seeds 42 123 2026
```

### 5. Auto-teste rápido do modelo

```bash
python model.py   # verifica shapes, normalização L2, lens e a loss
```

---

## Status dos Experimentos

| ID | Nome | Status |
|---|---|---|
| **A** | Baseline VCTK (zero-shot) | ✅ **Concluído** (12/11/2026) — `results/vctk/` |
| **B** | 2 CNNs × 2 datasets (com Dupla 8) | ⏳ Fase 5 — depende da Dupla 8 |
| **C** | Cross-dataset (VCTK → LibriSpeech) | ✅ **Concluído** (13/11/2026) — `results/librispeech/` |
| **D** | Robustez a ruído (SNR 0–20 dB) | ✅ **Concluído** (13/11/2026) — `results/vctk/robustness/` |
| **E** | Ablação do pipeline (VAD/CMN) | ✅ **Concluído** (13/11/2026) — `results/vctk/ablation/` |

### Resultados — Experimento A (VCTK, zero-shot, 12/11/2026)

Protocolo: 16 falantes val / 17 falantes test (split por identidade, seed 42).
Threshold **congelado na validação** (EER val = 0,353) e aplicado ao teste.
Arquivos em `results/vctk/` (`scores.csv`, `val_scores.csv`, `metrics.json`,
`roc.png`, `distributions.png`, `error_analysis.json`).

| Métrica | VCTK (Exp. A) | LibriSpeech (Exp. C) |
|---|---|---|
| **EER (%)** | **1,36** | **0,64** |
| **AUC-ROC** | **0,9991** | **0,9991** |
| **FMR @ threshold** | **1,08%** | 1,84% |
| **FNMR @ threshold** | **1,81%** | 0,44% |
| **TAR @ FAR=1%** | **96,99%** | 99,44% |
| **d-prime** | 4,78 | 6,73 |
| **Score genuíno médio** | 0,592 (n=166) | 0,783 (n=2510) |
| **Score impostor médio** | 0,115 (n=4799) | 0,106 (n=4998) |
| **Inferência (ms/utt, CPU)** | 107,5 ± 54,1 | 348,0 ± 30,7 |
| **FTA (pós-VAD < 1 s)** | 322 / 13.917 (2,3%) | 1 / 11.689 |
| **Parâmetros** | 20.767.552 | 20.767.552 |
| **Tamanho do modelo (MB)** | 79,2 | 79,2 |
| **Hardware** | CPU AMD64 (sem GPU) | CPU AMD64 |

> **Exp. C (cross-dataset):** threshold congelado no VCTK (0,3532) aplicado
> ao LibriSpeech **sem novo fine-tuning** no destino (enunciado §11).
> Embeddings extraídos com o mesmo modelo zero-shot; 7.508 pares avaliados
> (1 FTA). Resultado em `results/librispeech/metrics.json`.

### Resultados — Experimento D (robustez, 13/11/2026)

Ruído aditivo **white** e **babble** em SNR ∈ {20, 10, 5, 0} dB + original
(enunciado: ≥4 níveis). Threshold congelado na validação limpa (0,3532).
Babble = mistura de 3–6 utterances de falantes do próprio VCTK.
Script: `evaluate_robustness.py` · Saída: `results/vctk/robustness/`.

| Condição | EER% | AUC | FMR% | FNMR% | TAR@1% |
|---|---:|---:|---:|---:|---:|
| original | **1,36** | 0,9991 | 1,08 | 1,81 | 96,99 |
| white@20dB | 3,92 | 0,9944 | 1,03 | 7,19 | 92,81 |
| white@10dB | 6,17 | 0,9879 | 1,45 | 8,03 | 90,12 |
| white@5dB | 5,87 | 0,9831 | 3,53 | 8,82 | 81,18 |
| white@0dB | 8,54 | 0,9748 | 6,04 | 12,94 | 62,35 |
| babble@20dB | 4,75 | 0,9918 | 1,12 | 9,52 | 89,88 |
| babble@10dB | 15,90 | 0,9174 | 1,86 | 38,82 | 54,12 |
| babble@5dB | 27,52 | 0,8052 | 2,26 | 65,29 | 26,47 |
| babble@0dB | 39,06 | 0,6530 | 2,90 | 87,06 | 6,47 |

> **Leitura:** o modelo é razoavelmente robusto a ruído branco
> (EER ≤ 8,5% mesmo a 0 dB) mas degrada fortemente com babble
> (concorrentes fonéticos) — FNMR salta de 9,5% (20 dB) para 87% (0 dB).

### Resultados — Experimento E (ablação, 13/11/2026)

Variantes do pré-processamento; threshold congelado na validação **de cada
variante** (anti-leak). Script: `ablation_study.py` ·
Saída: `results/vctk/ablation/`.

| Variante | EER% | AUC | FMR% | FNMR% | TAR@1% |
|---|---:|---:|---:|---:|---:|
| full (VAD+CMN) | **1,36** | 0,9991 | 1,08 | 1,81 | 96,99 |
| no_vad | **1,10** | 0,9994 | 1,02 | 1,18 | 98,82 |
| no_cmn | 1,81 | 0,9984 | 2,48 | 1,20 | 96,39 |
| no_vad_no_cmn | 2,36 | 0,9973 | 2,56 | 1,76 | 95,29 |

> **Leitura:** CMN contribui claramente (remover piora EER 1,36→1,81).
> O VAD neste protocolo (utterances já trimmed) não ajuda e até atrapalha
> levemente — provavelmente por descartar duração útil em frases curtas.
> Sem as duas etapas o EER sobe para 2,36% (~1,7× o baseline).

> **Nota (correção de bug):** a primeira execução do Exp. A acusou EER ≈ 40%
> porque a "CMN" implementada fazia z-score (subtrai média **e** divide por
> desvio). O SpeechBrain usa apenas subtração de média
> (`std_norm=False`); com z-score os embeddings colapsavam (cosine ≈ 0,98
> entre falantes distintos). Corrigido em `preprocess._cmn_normalize` e
> todos os embeddings reextraídos.

---

## Configuração

Tudo em [`config.yaml`](config.yaml): pré-processamento, modelo, treino
(seeds 42/123/2026, 20 épocas, batch 64), matching (cosine, threshold da
validação) e paths. Os scripts aceitam overrides por CLI (`--epochs`,
`--seeds`, `--data_dir`, `--protocols_dir`, `--output_dir`, ...).

---

## Seeds de Reprodutibilidade

| Uso | Seed |
|---|---|
| Experimento principal | `42` |
| Validação cruzada | `123` |
| Reprodução alternativa | `2026` |

---

## Hardware e Software

| Componente | Versão |
|---|---|
| Python | 3.13 (testado) |
| PyTorch | 2.12.1 (CPU) |
| torchaudio | 2.11.1 |
| SpeechBrain | 1.1.2 |
| GPU | pendente (pipeline roda em CPU) |

---

## Referências

1. **Desplanques, B., Thienpondt, J., & Demuynck, K.** (2020). ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification. *Interspeech 2020*. https://doi.org/10.21437/Interspeech.2020-2650
2. **Nagrani, A., Chung, J. S., & Zisserman, A.** (2017). VoxCeleb: A Large-Scale Speaker Identification Dataset. *Interspeech 2017*.
3. **Chung, J. S., Nagrani, A., & Zisserman, A.** (2018). VoxCeleb2: Deep Speaker Recognition. *Interspeech 2018*.
4. **Ravanelli, M., Parcollet, T., Plantinga, P., et al.** (2021). SpeechBrain: A General-Purpose Speech Toolkit. *arXiv:2106.04624*.
5. **Panayotov, V., Chen, G., Povey, D., & Khudanpur, S.** (2015). LibriSpeech: An ASR corpus based on public domain audio books. *ICASSP 2015*.
6. **Yamagishi, J., Veaux, C., & MacDonald, K.** (2019). CSTR VCTK Corpus (version 0.92). *University of Edinburgh*. https://doi.org/10.7488/ds/2645
7. **Snyder, D., Garcia-Romero, D., Sell, G., Povey, D., & Khudanpur, S.** (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition. *ICASSP 2018*.
8. **ISO/IEC 19795-1:2006** — Biometric performance testing and reporting — Part 1: Principles and framework.

---

## Licença

MIT — ver [LICENSE](LICENSE).

---

## Versão

| Campo | Valor |
|---|---|
| **Tag de entrega** | `final-v1.0` (criar no commit final) |

---

<p align="center">
  <strong>Dupla 7</strong> — UTFPR Campus Ponta Grossa · Biometria
</p>
