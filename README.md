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
| **Normalização** | Por utterance — CMN |
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
# Pesos pré-treinados (zero-shot)
python extract_embeddings.py --dataset vctk \
    --data_dir data/vctk/wav48_silence_trimmed \
    --output_dir embeddings/ --config config.yaml

# A partir de um checkpoint fine-tuned (opcional)
python extract_embeddings.py --dataset vctk \
    --data_dir data/vctk/wav48_silence_trimmed \
    --output_dir embeddings_ft/ \
    --checkpoint checkpoints/best_model_seed42.pt --config config.yaml
```

### 3. Avaliar (Experimento A — baseline VCTK)

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
| **A** | Baseline VCTK (zero-shot) | pipeline pronto — aguarda dataset completo |
| **B** | Limiar adaptativo (DET) | parcial (threshold EER na validação já implementado) |
| **C** | Cross-dataset (VCTK → LibriSpeech) | planejado — flag `--cross-dataset` a implementar |
| **D** | Robustez a ruído (SNR 0–20 dB) | planejado — `evaluate_robustness.py` a implementar |
| **E** | Ablação do pipeline (VAD/CMN/Mel) | planejado — `ablation_study.py` a implementar |

### Resultados (preencher após execução)

| Métrica | VCTK (Exp. A) | LibriSpeech (Exp. C) |
|---|---|---|
| **EER (%)** | — | — |
| **AUC-ROC** | — | — |
| **FMR @ threshold** | — | — |
| **FNMR @ threshold** | — | — |
| **TAR @ FAR=1%** | — | — |
| **Inferência (ms/utt, CPU)** | ~51 (medido, smoke test) | — |
| **Parâmetros** | 20.767.552 | 20.767.552 |

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
