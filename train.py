"""
train.py — Fine-tuning do ECAPA-TDNN para Speaker Verification (Dupla 7 — UTFPR Biometria)

Realiza o treinamento supervisionado do ECAPA-TDNN usando AAM-Softmax Loss
na base VCTK. Suporta múltiplas seeds para reportar média ± desvio-padrão do EER.

Uso:
    python train.py --config config.yaml --seeds 42 123 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import yaml
import json
import argparse
import random
import os
from pathlib import Path
from tqdm import tqdm

from preprocess import AudioPreprocessor
from model import AAMSoftmax, load_pretrained_ecapa


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Define a seed em todas as bibliotecas relevantes para reprodutibilidade.

    Args:
        seed: Valor inteiro da seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Garante determinismo nas operações CUDA (pode reduzir desempenho)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> float:
    """Calcula a Equal Error Rate (EER) a partir dos scores de pares genuínos e impostores.

    A EER é o ponto onde a False Acceptance Rate (FAR) iguala a
    False Rejection Rate (FRR), servindo como métrica principal de avaliação.

    Args:
        genuine_scores: Array de scores de similaridade para pares genuínos.
        impostor_scores: Array de scores de similaridade para pares impostores.

    Returns:
        EER como valor entre 0 e 1 (ex: 0.05 = 5%).
    """
    # Combina todos os scores e gera os thresholds possíveis
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.sort(np.unique(all_scores))

    min_diff = float("inf")
    eer = 0.0

    for threshold in thresholds:
        # FAR: fração de impostores aceitos (score >= threshold)
        far = np.mean(impostor_scores >= threshold)
        # FRR: fração de genuínos rejeitados (score < threshold)
        frr = np.mean(genuine_scores < threshold)

        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2.0

    return float(eer)


def cosine_similarity_matrix(
    embeddings_a: torch.Tensor, embeddings_b: torch.Tensor
) -> torch.Tensor:
    """Calcula a similaridade de cosseno entre dois conjuntos de embeddings.

    Args:
        embeddings_a: Tensor de shape (N, D).
        embeddings_b: Tensor de shape (N, D).

    Returns:
        Tensor de shape (N,) com os scores de similaridade.
    """
    a_norm = nn.functional.normalize(embeddings_a, p=2, dim=1)
    b_norm = nn.functional.normalize(embeddings_b, p=2, dim=1)
    return (a_norm * b_norm).sum(dim=1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpeakerDataset(Dataset):
    """Dataset para treinamento de verificação de locutor na base VCTK.

    Lê um arquivo CSV com colunas `speaker_id` e `file_path`, cria um mapeamento
    de speaker_id para índice de classe e retorna espectrogramas com seus rótulos.

    Args:
        csv_path: Caminho para o arquivo CSV com as colunas `speaker_id` e `file_path`.
        preprocessor: Instância de :class:`AudioPreprocessor` para extração de features.
        augment: Se True, aplica SpecAugment (mascaramento temporal e de frequência).
    """

    # Comprimento fixo em frames para o collate — ajustável via config
    FIXED_FRAMES = 300

    def __init__(
        self,
        csv_path: str,
        preprocessor: "AudioPreprocessor",
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.augment = augment

        # Carrega o CSV
        self.df = pd.read_csv(csv_path)
        if "speaker_id" not in self.df.columns or "file_path" not in self.df.columns:
            raise ValueError(
                f"CSV '{csv_path}' deve conter as colunas 'speaker_id' e 'file_path'."
            )

        # Cria mapeamento speaker_id → class_idx (índice contínuo 0..N-1)
        unique_speakers = sorted(self.df["speaker_id"].unique())
        self.speaker_to_idx: dict = {spk: idx for idx, spk in enumerate(unique_speakers)}
        self.num_classes: int = len(unique_speakers)

        # Armazena como lista de tuplas para acesso rápido
        self.samples: list[tuple[str, int]] = [
            (row["file_path"], self.speaker_to_idx[row["speaker_id"]])
            for _, row in self.df.iterrows()
        ]

    def __len__(self) -> int:
        """Retorna o número total de amostras no dataset."""
        return len(self.samples)

    def _spec_augment(self, spec: torch.Tensor) -> torch.Tensor:
        """Aplica SpecAugment: mascaramento temporal e de frequência.

        Args:
            spec: Espectrograma de shape (freq_bins, time_frames).

        Returns:
            Espectrograma aumentado com a mesma shape.
        """
        freq_bins, time_frames = spec.shape

        # --- Mascaramento de frequência ---
        # Máscara aleatória em até 15% das bandas de frequência
        max_freq_mask = max(1, int(0.15 * freq_bins))
        f = random.randint(0, max_freq_mask)
        if f > 0:
            f0 = random.randint(0, freq_bins - f)
            spec[f0 : f0 + f, :] = 0.0

        # --- Mascaramento temporal ---
        # Máscara aleatória em até 15% dos frames temporais
        max_time_mask = max(1, int(0.15 * time_frames))
        t = random.randint(0, max_time_mask)
        if t > 0:
            t0 = random.randint(0, time_frames - t)
            spec[:, t0 : t0 + t] = 0.0

        return spec

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Retorna o espectrograma e o rótulo de classe para o índice dado.

        Args:
            idx: Índice da amostra.

        Returns:
            Tupla (spectrogram, label) onde spectrogram é um Tensor de shape
            (freq_bins, time_frames) e label é o índice inteiro do locutor.
        """
        file_path, label = self.samples[idx]

        # Extrai o espectrograma usando o AudioPreprocessor
        spec = self.preprocessor.process(file_path)  # Tensor (freq_bins, time_frames)

        # Aplica augmentation se habilitado e em modo de treino
        if self.augment:
            spec = self._spec_augment(spec.clone())

        return spec, label


def collate_fn(
    batch: list[tuple[torch.Tensor, int]], fixed_frames: int = SpeakerDataset.FIXED_FRAMES
) -> tuple[torch.Tensor, torch.Tensor]:
    """Função de collate que padeia/trunca espectrogramas para comprimento fixo.

    Args:
        batch: Lista de tuplas (spectrogram, label).
        fixed_frames: Número alvo de frames temporais.

    Returns:
        Tupla (spectrograms, labels, lengths) com shapes
        (batch_size, freq_bins, fixed_frames), (batch_size,) e
        (batch_size,) — lengths são comprimentos relativos em [0, 1].
    """
    specs, labels = zip(*batch)
    freq_bins = specs[0].shape[0]
    out_specs = torch.zeros(len(specs), freq_bins, fixed_frames)
    lengths = torch.zeros(len(specs), dtype=torch.float)

    for i, spec in enumerate(specs):
        n_frames = spec.shape[1]
        if n_frames >= fixed_frames:
            # Trunca: pega os primeiros `fixed_frames` frames
            out_specs[i] = spec[:, :fixed_frames]
            lengths[i] = 1.0
        else:
            # Padeia com zeros à direita
            out_specs[i, :, :n_frames] = spec
            lengths[i] = n_frames / fixed_frames

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return out_specs, labels_tensor, lengths


# ---------------------------------------------------------------------------
# Treinamento — uma época
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    classifier: "AAMSoftmax",
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Executa uma época completa de treinamento com AAM-Softmax.

    Args:
        model: Instância do ECAPA-TDNN (extrator de embeddings).
        classifier: Cabeça AAMSoftmax responsável pelo cálculo da perda.
        dataloader: DataLoader de treinamento.
        optimizer: Otimizador (ex: Adam).
        device: Dispositivo de computação (cpu/cuda).
        epoch: Número da época atual (para exibição no tqdm).

    Returns:
        Tupla (loss_medio, acuracia_media) calculados sobre todos os batches.
    """
    model.train()
    classifier.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Época {epoch:03d} [treino]", leave=False)

    for specs, labels, lengths in pbar:
        specs = specs.to(device)      # (B, freq_bins, T)
        labels = labels.to(device)    # (B,)
        lengths = lengths.to(device)  # (B,) comprimentos relativos (0–1)

        optimizer.zero_grad()

        # Forward: obtém embeddings do ECAPA-TDNN (mascara padding via lens)
        embeddings = model(specs, lens=lengths)  # (B, emb_dim)

        # Calcula perda AAM-Softmax e logits para acurácia
        loss, logits = classifier(embeddings, labels)

        # Backprop
        loss.backward()
        # Clipa gradientes para estabilidade
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(classifier.parameters()), max_norm=5.0
        )
        optimizer.step()

        # Métricas do batch
        batch_size = labels.size(0)
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()

        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += batch_size

        # Atualiza barra de progresso
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / batch_size:.3f}",
        )

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


# ---------------------------------------------------------------------------
# Validação — cálculo do EER
# ---------------------------------------------------------------------------

def validate(
    model: nn.Module,
    val_dataloader: DataLoader,
    protocols_dir: str,
    device: torch.device,
) -> float:
    """Calcula o EER de validação usando pares genuínos e impostores.

    Extrai embeddings L2-normalizados para todos os arquivos de validação,
    calcula similaridade de cosseno para cada par e computa o EER.

    Args:
        model: Instância do ECAPA-TDNN em modo de avaliação.
        val_dataloader: DataLoader de validação (sem augmentation).
        protocols_dir: Diretório contendo `genuine_pairs.csv` e `impostor_pairs.csv`.
        device: Dispositivo de computação.

    Returns:
        EER de validação como float entre 0 e 1.
    """
    model.eval()

    # --- Extrai embeddings para todos os arquivos de validação ---
    emb_map: dict[str, np.ndarray] = _extract_embeddings_by_path(
        model, val_dataloader, device
    )

    # --- Carrega pares de validação do protocolo ---
    genuine_path = os.path.join(protocols_dir, "val_genuine_pairs.csv")
    impostor_path = os.path.join(protocols_dir, "val_impostor_pairs.csv")

    if not os.path.exists(genuine_path) or not os.path.exists(impostor_path):
        raise FileNotFoundError(
            f"Pares de validação não encontrados em '{protocols_dir}'. "
            "Esperado: val_genuine_pairs.csv e val_impostor_pairs.csv "
            "(gere com generate_protocols.py)."
        )

    genuine_df = pd.read_csv(genuine_path)
    impostor_df = pd.read_csv(impostor_path)

    # --- Calcula scores de cosseno para cada par ---
    genuine_scores = _score_pairs(genuine_df, emb_map)
    impostor_scores = _score_pairs(impostor_df, emb_map)

    eer = compute_eer(genuine_scores, impostor_scores)
    return eer


def _extract_embeddings_by_path(
    model: nn.Module,
    val_dataloader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Extrai embeddings mapeados por caminho de arquivo.

    Itera pelo dataset subjacente em batches, usando os caminhos armazenados
    em `val_dataloader.dataset.samples` para criar o mapeamento.

    Args:
        model: Modelo extrator de embeddings em modo eval.
        val_dataloader: DataLoader cujo dataset possui atributo `samples`.
        device: Dispositivo de computação.

    Returns:
        Dicionário {file_path: embedding_numpy}.
    """
    model.eval()
    dataset = val_dataloader.dataset
    emb_map: dict[str, np.ndarray] = {}

    with torch.no_grad():
        batch_specs = []
        batch_paths = []
        batch_size = val_dataloader.batch_size or 32

        for i, (file_path, _label) in enumerate(dataset.samples):
            spec, _ = dataset[i]
            batch_specs.append(spec)
            batch_paths.append(file_path)

            if len(batch_specs) == batch_size or i == len(dataset.samples) - 1:
                # Monta batch com pad/trim e comprimentos relativos
                specs_tensor = _pad_trim_batch(batch_specs, SpeakerDataset.FIXED_FRAMES)
                specs_tensor = specs_tensor.to(device)

                lens = torch.tensor(
                    [
                        min(s.shape[1], SpeakerDataset.FIXED_FRAMES)
                        / SpeakerDataset.FIXED_FRAMES
                        for s in batch_specs
                    ],
                    dtype=torch.float,
                    device=device,
                )

                embeddings = model(specs_tensor, lens=lens)
                embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
                embeddings_np = embeddings.cpu().numpy()

                for path, emb in zip(batch_paths, embeddings_np):
                    emb_map[path] = emb

                batch_specs = []
                batch_paths = []

    return emb_map


def _pad_trim_batch(specs: list[torch.Tensor], fixed_frames: int) -> torch.Tensor:
    """Padeia/trunca uma lista de espectrogramas para comprimento fixo.

    Args:
        specs: Lista de tensores de shape (freq_bins, time_frames).
        fixed_frames: Comprimento alvo em frames.

    Returns:
        Tensor de shape (N, freq_bins, fixed_frames).
    """
    freq_bins = specs[0].shape[0]
    out = torch.zeros(len(specs), freq_bins, fixed_frames)
    for i, spec in enumerate(specs):
        n = spec.shape[1]
        if n >= fixed_frames:
            out[i] = spec[:, :fixed_frames]
        else:
            out[i, :, :n] = spec
    return out


def _score_pairs(
    pairs_df: pd.DataFrame, emb_map: dict[str, np.ndarray]
) -> np.ndarray:
    """Calcula scores de similaridade de cosseno para pares de arquivos.

    Aceita as colunas de par ``amostra_1``/``amostra_2`` (formato do
    generate_protocols.py) ou ``file1``/``file2`` (formato legado).

    Args:
        pairs_df: DataFrame com colunas de par de arquivos.
        emb_map: Mapeamento {file_path: embedding}.

    Returns:
        Array numpy com os scores de similaridade para cada par.
    """
    if "amostra_1" in pairs_df.columns:
        col1, col2 = "amostra_1", "amostra_2"
    else:
        col1, col2 = "file1", "file2"

    scores = []
    skipped = 0
    for _, row in pairs_df.iterrows():
        f1, f2 = str(row[col1]), str(row[col2])
        if f1 not in emb_map or f2 not in emb_map:
            skipped += 1
            continue
        e1 = emb_map[f1]
        e2 = emb_map[f2]
        # Similaridade de cosseno manual (L2 já normalizado)
        score = float(np.dot(e1, e2))
        scores.append(score)

    if skipped > 0:
        print(f"  [aviso] {skipped} pares pulados por falta de embeddings.")
    if not scores:
        raise ValueError(
            "Nenhum par pôde ser avaliado — verifique se os caminhos dos "
            "pares do protocolo batem com os caminhos do dataset."
        )

    return np.array(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# Treinamento completo com uma seed
# ---------------------------------------------------------------------------

def train_with_seed(seed: int, config: dict) -> dict:
    """Executa o pipeline completo de treinamento para uma seed específica.

    Configura o ambiente determinístico, carrega o modelo ECAPA-TDNN,
    cria os dataloaders e executa o loop de épocas com early stopping.

    Args:
        seed: Valor da seed para reprodutibilidade.
        config: Dicionário de configuração (lido de config.yaml).

    Returns:
        Dicionário com:
            - ``best_eer``: Melhor EER de validação obtido.
            - ``checkpoint_path``: Caminho para o melhor checkpoint salvo.
            - ``seed``: Seed utilizada.
    """
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"  Treinamento com seed={seed}")
    print(f"{'='*60}")

    # --- Dispositivo ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Dispositivo: {device}")

    # --- Seções do config.yaml ---
    train_cfg = config.get("training", {})
    prep_cfg = config.get("preprocessing", {})
    model_cfg = config.get("model", {})
    paths_cfg = config.get("paths", {})
    ds_cfg = config.get("datasets", {}).get("vctk", {})

    # --- Diretórios ---
    data_dir = config.get("data_dir") or ds_cfg.get("path", "data/vctk")
    output_dir = config.get("output_dir") or train_cfg.get(
        "checkpoint_dir", "checkpoints"
    )
    dataset_name = config.get("dataset", "vctk")
    protocols_root = config.get("protocols_dir") or paths_cfg.get(
        "protocols_dir", "protocols"
    )
    protocols_dir = os.path.join(protocols_root, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print(f"  Data dir      : {data_dir}")
    print(f"  Protocols dir : {protocols_dir}")
    print(f"  Checkpoints   : {output_dir}")

    # --- Preprocessador (mesmo pipeline do extract_embeddings) ---
    preprocessor = AudioPreprocessor(
        config={
            "sample_rate": prep_cfg.get("sample_rate", 16000),
            "n_mels": prep_cfg.get("n_mels", 80),
            "n_fft": prep_cfg.get("n_fft", 512),
            "win_length_ms": prep_cfg.get("win_length_ms", 25),
            "hop_length_ms": prep_cfg.get("hop_length_ms", 10),
            "fmin": prep_cfg.get("fmin", 20),
            "fmax": prep_cfg.get("fmax", 7600),
            "vad_enabled": prep_cfg.get("vad_enabled", True),
            "vad_aggressiveness": prep_cfg.get("vad_aggressiveness", 2),
            "normalize": prep_cfg.get("normalize", True),
        }
    )

    # --- Datasets (CSVs do protocolo, divisão por identidade) ---
    train_csv = os.path.join(protocols_dir, "train.csv")
    val_csv = os.path.join(protocols_dir, "validation.csv")

    train_dataset = SpeakerDataset(train_csv, preprocessor, augment=True)
    val_dataset = SpeakerDataset(val_csv, preprocessor, augment=False)

    num_classes = train_dataset.num_classes
    print(f"  Locutores de treino: {num_classes}")
    print(f"  Amostras de treino: {len(train_dataset)} | validação: {len(val_dataset)}")

    # --- Dataloaders ---
    batch_size = train_cfg.get("batch_size", 64)
    # No Windows, num_workers > 0 com spawn é lento e frágil
    default_workers = 0 if os.name == "nt" else 4
    num_workers = train_cfg.get("num_workers", default_workers)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # --- Modelo (pesos pré-treinados VoxCeleb1+2 via SpeechBrain) ---
    emb_dim = model_cfg.get("embedding_dim", 192)
    freeze_encoder = train_cfg.get("freeze_encoder", False)
    use_pretrained = train_cfg.get("fine_tune", True)

    model = load_pretrained_ecapa(
        embedding_dim=emb_dim, use_speechbrain=bool(use_pretrained)
    )

    # Checkpoint local opcional (sobrescreve os pesos pré-treinados)
    pretrained_path = config.get("pretrained_path", None)
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"  Carregando pesos de checkpoint local: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location="cpu")
        state = (
            ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            if isinstance(ckpt, dict)
            else ckpt
        )
        model.load_state_dict(state, strict=False)

    if freeze_encoder:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("  Encoder CONGELADO (apenas a cabeça AAM-Softmax é treinada).")

    model = model.to(device)

    # --- Cabeça AAM-Softmax ---
    classifier = AAMSoftmax(
        embedding_dim=emb_dim,
        n_classes=num_classes,
        margin=model_cfg.get("aam_margin", 0.2),
        scale=model_cfg.get("aam_scale", 30.0),
    ).to(device)

    # --- Otimizador e scheduler ---
    lr = train_cfg.get("learning_rate", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        trainable_params + list(classifier.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    num_epochs = config.get("epochs") or train_cfg.get("epochs", 20)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # --- Loop de épocas com early stopping ---
    best_eer = float("inf")
    best_checkpoint_path = os.path.join(output_dir, f"best_model_seed{seed}.pt")
    patience = train_cfg.get("early_stopping_patience", 5)
    patience_counter = 0
    validate_every = train_cfg.get("validate_every", 2)

    history = []

    for epoch in range(1, num_epochs + 1):
        # Treinamento
        train_loss, train_acc = train_one_epoch(
            model, classifier, train_loader, optimizer, device, epoch
        )
        scheduler.step()

        epoch_info = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_eer": None,
        }

        # Validação a cada `validate_every` épocas ou na última
        if epoch % validate_every == 0 or epoch == num_epochs:
            val_eer = validate(model, val_loader, protocols_dir, device)
            epoch_info["val_eer"] = round(val_eer * 100, 4)  # em %

            print(
                f"  Época {epoch:03d}/{num_epochs} | "
                f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
                f"EER: {val_eer*100:.2f}%"
            )

            # Salva checkpoint se melhorou
            if val_eer < best_eer:
                best_eer = val_eer
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "classifier_state_dict": classifier.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_eer": val_eer,
                        "seed": seed,
                        "config": config,
                    },
                    best_checkpoint_path,
                )
                print(f"  ✔ Checkpoint salvo: EER={val_eer*100:.2f}%")
            else:
                patience_counter += 1
                print(
                    f"  Sem melhora ({patience_counter}/{patience}). "
                    f"Melhor EER: {best_eer*100:.2f}%"
                )

            # Early stopping
            if patience_counter >= patience:
                print(f"  Early stopping ativado na época {epoch}.")
                history.append(epoch_info)
                break
        else:
            print(
                f"  Época {epoch:03d}/{num_epochs} | "
                f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f}"
            )

        history.append(epoch_info)

    # Salva histórico de treinamento desta seed
    history_path = os.path.join(output_dir, f"history_seed{seed}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\n  Seed {seed} concluída. Melhor EER: {best_eer*100:.2f}%")
    print(f"  Checkpoint: {best_checkpoint_path}")

    return {
        "seed": seed,
        "best_eer": best_eer,
        "checkpoint_path": best_checkpoint_path,
        "history_path": history_path,
    }


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def main() -> None:
    """Ponto de entrada principal do script de treinamento.

    Realiza o parse dos argumentos, carrega a configuração, executa o
    treinamento para cada seed e reporta os resultados agregados.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tuning do ECAPA-TDNN para Speaker Verification (Dupla 7 — UTFPR Biometria)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Caminho para o arquivo de configuração YAML (padrão: config.yaml).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 2026],
        help="Lista de seeds para treinamento (padrão: 42 123 2026).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Diretório raiz dos dados (sobrescreve config.yaml).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Diretório para salvar checkpoints (sobrescreve config.yaml).",
    )
    parser.add_argument(
        "--protocols_dir",
        type=str,
        default=None,
        help="Diretório raiz dos protocolos, ex.: protocols/ (sobrescreve config.yaml).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["vctk", "librispeech"],
        help="Dataset alvo (padrão: vctk).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Número de épocas (sobrescreve config.yaml).",
    )
    args = parser.parse_args()

    # --- Carrega configuração ---
    config: dict = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        print(f"Configuração carregada de: {args.config}")
    else:
        print(f"[aviso] Arquivo de configuração '{args.config}' não encontrado. Usando padrões.")

    # Sobrescreve config com argumentos de linha de comando, se fornecidos
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.protocols_dir is not None:
        config["protocols_dir"] = args.protocols_dir
    if args.dataset is not None:
        config["dataset"] = args.dataset
    if args.epochs is not None:
        config["epochs"] = args.epochs

    print(f"\nSeeds: {args.seeds}")
    print(f"Data dir: {config.get('data_dir', 'data/vctk')}")
    print(f"Output dir: {config.get('output_dir', 'checkpoints')}")

    # --- Executa treinamento para cada seed ---
    all_results: list[dict] = []
    for seed in args.seeds:
        result = train_with_seed(seed, config)
        all_results.append(result)

    # --- Agrega resultados ---
    eers = [r["best_eer"] for r in all_results]
    mean_eer = float(np.mean(eers))
    std_eer = float(np.std(eers))

    print(f"\n{'='*60}")
    print("  Resumo final do treinamento")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  Seed {r['seed']:>6}: EER = {r['best_eer']*100:.2f}%  →  {r['checkpoint_path']}")
    print(f"\n  EER médio:  {mean_eer*100:.2f}%")
    print(f"  Desvio-padrão: {std_eer*100:.2f}%")
    print(f"  Resultado: {mean_eer*100:.2f} ± {std_eer*100:.2f}%")

    # --- Salva resumo em JSON ---
    summary = {
        "seeds": args.seeds,
        "results": [
            {
                "seed": r["seed"],
                "best_eer_pct": round(r["best_eer"] * 100, 4),
                "checkpoint_path": r["checkpoint_path"],
            }
            for r in all_results
        ],
        "mean_eer_pct": round(mean_eer * 100, 4),
        "std_eer_pct": round(std_eer * 100, 4),
        "summary_str": f"{mean_eer*100:.2f} ± {std_eer*100:.2f}%",
        "config": config,
    }

    os.makedirs("results", exist_ok=True)
    summary_path = "results/training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Resumo salvo em: {summary_path}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
