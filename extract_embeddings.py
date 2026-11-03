"""
================================================================================
extract_embeddings.py — Extração de Embeddings com ECAPA-TDNN
================================================================================
Projeto   : Verificação de Locutor (Speaker Verification) — Dupla 7
Disciplina: Biometria — UTFPR

Descrição:
    Processa todos os arquivos de áudio de um dataset, extrai embeddings com o
    modelo ECAPA-TDNN (pré-treinado ou fine-tuned) e salva os resultados em
    disco no formato .npy. Também mede e reporta o tempo médio de inferência
    e gera um relatório JSON em `results/{dataset}/inference_report.json`.

Uso:
    python extract_embeddings.py --dataset vctk --data_dir data/vctk
    python extract_embeddings.py --dataset librispeech --data_dir data/librispeech \\
        --checkpoint checkpoints/best.pt --batch_size 64
================================================================================
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from model import get_model_info, load_pretrained_ecapa
from preprocess import AudioPreprocessor

# ---------------------------------------------------------------------------
# Configuração de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("extract_embeddings")

# Extensões de áudio suportadas
SUPPORTED_EXTENSIONS = {".wav", ".flac"}


# ==============================================================================
# 1. load_model — Carrega o modelo ECAPA-TDNN
# ==============================================================================

def load_model(
    checkpoint_path: str = None,
    device: str = "auto",
) -> Tuple[torch.nn.Module, torch.device, dict]:
    """
    Carrega o modelo ECAPA-TDNN a partir de um checkpoint ou dos pesos pré-treinados.

    Se ``checkpoint_path`` for fornecido, carrega o checkpoint salvo pelo
    ``train.py`` (formato ``torch.save``). Caso contrário, instancia o modelo
    via :func:`load_pretrained_ecapa` de ``model.py``.

    Parâmetros
    ----------
    checkpoint_path : str, opcional
        Caminho para o arquivo de checkpoint (``.pt`` ou ``.pth``).
        Se ``None``, usa os pesos pré-treinados via SpeechBrain.
    device : str, opcional
        Dispositivo de inferência. ``"auto"`` seleciona CUDA se disponível,
        senão CPU. Valores explícitos: ``"cpu"``, ``"cuda"``, ``"cuda:0"`` etc.
        Padrão: ``"auto"``.

    Retorna
    -------
    tuple
        - ``model`` (:class:`torch.nn.Module`): modelo em modo eval, no device.
        - ``device`` (:class:`torch.device`): dispositivo selecionado.
        - ``model_info`` (dict): dicionário com metadados do modelo
          (``n_params``, ``model_size_mb``, ``embedding_dim``, ``architecture_name``).
    """
    # --- Seleção do dispositivo ---
    if device == "auto":
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(device)

    logger.info("Dispositivo selecionado: %s", _device)

    # --- Carregamento do modelo ---
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint não encontrado: {checkpoint_path}"
            )
        logger.info("Carregando checkpoint: %s", checkpoint_path)
        checkpoint = torch.load(str(checkpoint_path), map_location=_device)

        # Suporta dois formatos comuns: dict com 'model_state_dict' ou state_dict direto
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model = load_pretrained_ecapa(use_speechbrain=False)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Checkpoint carregado com sucesso.")
    else:
        logger.info("Nenhum checkpoint fornecido — usando pesos pré-treinados.")
        model = load_pretrained_ecapa(use_speechbrain=True)

    model.to(_device)
    model.eval()

    model_info = get_model_info(model)
    logger.info(
        "Modelo pronto: %s | %d parâmetros | embedding_dim=%s",
        model_info["architecture_name"],
        model_info["n_params"],
        model_info["embedding_dim"],
    )

    return model, _device, model_info


# ==============================================================================
# 2. extract_embedding — Extrai embedding de um único espectrograma
# ==============================================================================

def extract_embedding(
    model: torch.nn.Module,
    spectrogram: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Extrai o embedding de um único espectrograma Log-Mel.

    O modelo é usado em modo de avaliação (``torch.no_grad()``), garantindo
    que não haja acúmulo de gradientes nem uso de Dropout/BatchNorm em modo
    de treino.

    Parâmetros
    ----------
    model : torch.nn.Module
        Modelo ECAPA-TDNN já carregado e em modo eval.
    spectrogram : torch.Tensor
        Tensor de shape ``(n_mels, T)`` ou ``(1, n_mels, T)`` com o
        Log-Mel Spectrogram de uma utterance.
    device : torch.device
        Dispositivo de inferência (CPU ou CUDA).

    Retorna
    -------
    np.ndarray
        Array 1-D de float32 com o embedding L2-normalizado (192-d).
    """
    model.eval()

    # Garante que o espectrograma tenha dimensão de batch
    if spectrogram.dim() == 2:
        spectrogram = spectrogram.unsqueeze(0)  # (1, n_mels, T)

    spectrogram = spectrogram.to(device)

    with torch.no_grad():
        embedding = model(spectrogram)  # (1, embedding_dim)

    # Remove dimensão de batch e converte para numpy
    embedding_np = embedding.squeeze(0).cpu().numpy().astype(np.float32)

    # L2-normalização (garante norma unitária mesmo que o modelo já normalize)
    norm = np.linalg.norm(embedding_np)
    if norm > 1e-10:
        embedding_np = embedding_np / norm

    return embedding_np


# ==============================================================================
# 3. extract_embeddings_from_list — Extração em batch de uma lista de arquivos
# ==============================================================================

def extract_embeddings_from_list(
    model: torch.nn.Module,
    file_list: List[str],
    preprocessor: AudioPreprocessor,
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """
    Extrai embeddings de uma lista de arquivos de áudio usando batching.

    Os espectrogramas de cada batch são alinhados (padded) ao comprimento
    temporal máximo do batch antes de serem enviados ao modelo, garantindo
    eficiência de processamento sem desperdício de memória.

    Métricas de tempo de inferência são coletadas por utterance e registradas
    em log ao final do processamento. Arquivos que falham em qualquer etapa
    (pré-processamento ou inferência) são contabilizados como FTA
    (Failure-To-Acquire) e registrados em aviso no log.

    Parâmetros
    ----------
    model : torch.nn.Module
        Modelo ECAPA-TDNN em modo eval.
    file_list : list of str
        Lista de caminhos absolutos ou relativos para arquivos de áudio
        (``.wav`` ou ``.flac``).
    preprocessor : AudioPreprocessor
        Instância do pré-processador de áudio configurado.
    device : torch.device
        Dispositivo de inferência (CPU ou CUDA).
    batch_size : int, opcional
        Número de utterances por batch. Padrão: 32.

    Retorna
    -------
    dict
        Dicionário ``{file_path: embedding_array}``, onde ``file_path`` é a
        string original passada na lista e ``embedding_array`` é um
        :class:`numpy.ndarray` 1-D (192,) L2-normalizado.

    Nota
    ----
    Arquivos que falham não são incluídos no dicionário de retorno. O número
    de falhas pode ser calculado como ``len(file_list) - len(result)``.
    """
    model.eval()

    embeddings_dict: Dict[str, np.ndarray] = {}
    failed_files: List[str] = []
    inference_times_ms: List[float] = []  # tempo de inferência por utterance

    # Processa arquivos em chunks de batch_size
    total_files = len(file_list)
    progress = tqdm(
        range(0, total_files, batch_size),
        desc="Extraindo embeddings",
        unit="batch",
        total=(total_files + batch_size - 1) // batch_size,
    )

    for batch_start in progress:
        batch_paths = file_list[batch_start: batch_start + batch_size]

        # --- Pré-processamento do batch ---
        valid_paths: List[str] = []
        valid_specs: List[torch.Tensor] = []

        for file_path in batch_paths:
            try:
                spec = preprocessor.process(file_path)  # (n_mels, T)
                valid_paths.append(file_path)
                valid_specs.append(spec)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FTA (pré-processamento) '%s': %s", file_path, exc)
                failed_files.append(file_path)

        if not valid_specs:
            continue

        # --- Padding para comprimento uniforme no batch ---
        max_len = max(s.shape[-1] for s in valid_specs)
        # Comprimentos relativos (0–1) antes do padding — usados como máscara
        lens = torch.tensor(
            [s.shape[-1] / max_len for s in valid_specs], dtype=torch.float
        )
        padded: List[torch.Tensor] = []
        for spec in valid_specs:
            pad_amount = max_len - spec.shape[-1]
            if pad_amount > 0:
                spec = torch.nn.functional.pad(spec, (0, pad_amount))
            padded.append(spec)

        batch_tensor = torch.stack(padded, dim=0).to(device)  # (B, n_mels, T)

        # --- Inferência com medição de tempo ---
        try:
            if device.type == "cuda":
                torch.cuda.synchronize()

            t_start = time.perf_counter()

            with torch.no_grad():
                batch_embeddings = model(
                    batch_tensor, lens=lens.to(device)
                )  # (B, embedding_dim)

            if device.type == "cuda":
                torch.cuda.synchronize()

            t_end = time.perf_counter()
            elapsed_ms = (t_end - t_start) * 1000.0
            per_utterance_ms = elapsed_ms / len(valid_paths)
            inference_times_ms.extend([per_utterance_ms] * len(valid_paths))

        except Exception as exc:  # noqa: BLE001
            logger.error("Erro de inferência no batch [%d:%d]: %s",
                         batch_start, batch_start + batch_size, exc)
            failed_files.extend(valid_paths)
            continue

        # --- L2-normalização e coleta dos resultados ---
        batch_embeddings_np = batch_embeddings.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(batch_embeddings_np, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        batch_embeddings_np = batch_embeddings_np / norms

        for file_path, embedding in zip(valid_paths, batch_embeddings_np):
            embeddings_dict[file_path] = embedding

        progress.set_postfix(
            processados=len(embeddings_dict),
            fta=len(failed_files),
        )

    # --- Relatório de tempo ---
    if inference_times_ms:
        mean_ms = float(np.mean(inference_times_ms))
        std_ms = float(np.std(inference_times_ms))
        logger.info(
            "Tempo de inferência — média: %.2f ms/utterance | std: %.2f ms",
            mean_ms, std_ms,
        )
    else:
        mean_ms, std_ms = 0.0, 0.0

    if failed_files:
        logger.warning(
            "%d arquivo(s) falharam (FTA). Primeiros 10: %s",
            len(failed_files),
            failed_files[:10],
        )

    # Armazena métricas como atributo temporário no dicionário (via wrapper)
    embeddings_dict["__meta__inference_times_ms"] = np.array(inference_times_ms)  # type: ignore[assignment]

    return embeddings_dict


# ==============================================================================
# 4. save_embeddings — Persiste embeddings em disco
# ==============================================================================

def save_embeddings(
    embeddings_dict: Dict,
    output_dir: str,
    dataset_name: str,
) -> None:
    """
    Salva embeddings em disco no formato ``.npy``, organizados por falante.

    Estrutura de diretórios criada:

    .. code-block:: text

        output_dir/
        └── dataset_name/
            ├── speaker_id_1/
            │   ├── utterance_1.npy
            │   └── utterance_2.npy
            ├── speaker_id_2/
            │   └── ...
            └── embeddings_index.csv

    O arquivo ``embeddings_index.csv`` contém as colunas:
    ``file_path``, ``embedding_path``, ``speaker_id``, ``duration_sec``.
    A coluna ``duration_sec`` é inferida a partir do nome do arquivo quando
    não disponível diretamente.

    Parâmetros
    ----------
    embeddings_dict : dict
        Dicionário ``{file_path: embedding_array}`` retornado por
        :func:`extract_embeddings_from_list` (a chave ``"__meta__*"``
        é ignorada automaticamente).
    output_dir : str
        Diretório raiz para salvar os embeddings.
    dataset_name : str
        Nome do dataset (``"vctk"`` ou ``"librispeech"``), usado como
        subdiretório dentro de ``output_dir``.
    """
    base_dir = Path(output_dir) / dataset_name
    base_dir.mkdir(parents=True, exist_ok=True)

    index_records = []

    for file_path, embedding in embeddings_dict.items():
        # Ignora chaves de metadados internas
        if str(file_path).startswith("__meta__"):
            continue

        path_obj = Path(file_path)

        # --- Inferência do speaker_id a partir da estrutura de diretórios ---
        # VCTK:       data/vctk/wav48_silence_trimmed/p225/p225_001_mic1.flac
        # LibriSpeech: data/librispeech/train-clean-100/84/121123/84-121123-0000.flac
        # Convenção: o diretório pai do arquivo é o speaker_id
        speaker_id = path_obj.parent.name

        # --- Diretório de saída por falante ---
        speaker_dir = base_dir / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)

        # --- Caminho do arquivo .npy (relativo ao índice/raiz do dataset) ---
        stem = path_obj.stem  # nome sem extensão
        npy_path = speaker_dir / f"{stem}.npy"
        np.save(str(npy_path), embedding)
        rel_npy_path = npy_path.relative_to(base_dir)

        # --- Estimativa de duração (não disponível sem reprocessar áudio) ---
        duration_sec = None  # campo reservado para uso futuro

        index_records.append(
            {
                "file_path": str(file_path),
                "embedding_path": str(rel_npy_path).replace("\\", "/"),
                "speaker_id": speaker_id,
                "duration_sec": duration_sec,
            }
        )

    # --- Salva index CSV ---
    if index_records:
        index_df = pd.DataFrame(index_records)
        index_csv = base_dir / "embeddings_index.csv"
        index_df.to_csv(str(index_csv), index=False)
        logger.info(
            "Índice salvo em: %s (%d entradas)", index_csv, len(index_records)
        )
    else:
        logger.warning("Nenhum embedding para salvar em '%s/%s'.", output_dir, dataset_name)


# ==============================================================================
# 5. _discover_audio_files — Varredura de arquivos de áudio no dataset
# ==============================================================================

def _discover_audio_files(data_dir: str, dataset: str) -> List[str]:
    """
    Descobre todos os arquivos de áudio suportados em um diretório de dataset.

    Percorre recursivamente o diretório ``data_dir`` e retorna todos os
    arquivos com extensão ``.wav`` ou ``.flac``. A lista é ordenada
    lexicograficamente para garantir reprodutibilidade.

    Parâmetros
    ----------
    data_dir : str
        Caminho para o diretório raiz do dataset.
    dataset : str
        Nome do dataset (usado apenas para mensagens de log).

    Retorna
    -------
    list of str
        Lista de caminhos absolutos de arquivos de áudio encontrados.

    Levanta
    -------
    FileNotFoundError
        Se ``data_dir`` não existir ou não for um diretório.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Diretório do dataset não encontrado: {data_dir}"
        )
    if not data_path.is_dir():
        raise NotADirectoryError(
            f"O caminho fornecido não é um diretório: {data_dir}"
        )

    audio_files = sorted(
        str(p)
        for p in data_path.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    logger.info(
        "Dataset '%s': %d arquivos de áudio encontrados em '%s'",
        dataset,
        len(audio_files),
        data_dir,
    )
    return audio_files


# ==============================================================================
# 6. _get_hardware_info — Informações sobre o hardware de inferência
# ==============================================================================

def _get_hardware_info(device: torch.device) -> dict:
    """
    Coleta informações sobre o hardware usado para inferência.

    Parâmetros
    ----------
    device : torch.device
        Dispositivo de inferência.

    Retorna
    -------
    dict
        Dicionário com campos ``type`` (``"GPU"`` ou ``"CPU"``) e
        ``name`` (nome do dispositivo).
    """
    if device.type == "cuda":
        hw_type = "GPU"
        try:
            hw_name = torch.cuda.get_device_name(device)
        except Exception:  # noqa: BLE001
            hw_name = "CUDA device (nome indisponível)"
    else:
        hw_type = "CPU"
        try:
            import platform
            hw_name = platform.processor() or "CPU (nome indisponível)"
        except Exception:  # noqa: BLE001
            hw_name = "CPU (nome indisponível)"

    return {"type": hw_type, "name": hw_name}


# ==============================================================================
# 7. main — Ponto de entrada principal
# ==============================================================================

def main() -> None:
    """
    Ponto de entrada principal do script de extração de embeddings.

    Fluxo de execução:
    1. Leitura dos argumentos de linha de comando.
    2. Carregamento da configuração YAML.
    3. Carregamento do modelo ECAPA-TDNN.
    4. Descoberta de todos os arquivos de áudio no dataset.
    5. Extração de embeddings com barra de progresso (tqdm).
    6. Salvamento dos embeddings e do índice CSV.
    7. Impressão e salvamento do relatório de inferência (JSON).
    """
    # -------------------------------------------------------------------------
    # 7.1 Argparse
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Extrai embeddings ECAPA-TDNN de um dataset de áudio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["vctk", "librispeech"],
        help="Nome do dataset a processar.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Caminho para o diretório raiz dos dados brutos.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="embeddings/",
        help="Diretório raiz para salvar os embeddings extraídos.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Caminho para um checkpoint salvo pelo train.py (opcional).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Número de utterances por batch na inferência.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Caminho para o arquivo de configuração YAML.",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 7.2 Carregamento da configuração
    # -------------------------------------------------------------------------
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info("Configuração carregada de: %s", config_path)
    else:
        logger.warning(
            "Arquivo de configuração não encontrado: %s. Usando defaults.", config_path
        )
        config = {}

    preprocessing_config = config.get("preprocessing", {})

    # -------------------------------------------------------------------------
    # 7.3 Carregamento do modelo
    # -------------------------------------------------------------------------
    model, device, model_info = load_model(
        checkpoint_path=args.checkpoint,
        device="auto",
    )

    # -------------------------------------------------------------------------
    # 7.4 Descoberta dos arquivos de áudio
    # -------------------------------------------------------------------------
    audio_files = _discover_audio_files(args.data_dir, args.dataset)

    if not audio_files:
        logger.error(
            "Nenhum arquivo de áudio encontrado em '%s'. Encerrando.", args.data_dir
        )
        return

    # -------------------------------------------------------------------------
    # 7.5 Pré-processador
    # -------------------------------------------------------------------------
    preprocessor = AudioPreprocessor(config=preprocessing_config)

    # -------------------------------------------------------------------------
    # 7.6 Extração de embeddings
    # -------------------------------------------------------------------------
    logger.info(
        "Iniciando extração de %d arquivos | batch_size=%d | device=%s",
        len(audio_files),
        args.batch_size,
        device,
    )

    t_total_start = time.perf_counter()
    embeddings_dict = extract_embeddings_from_list(
        model=model,
        file_list=audio_files,
        preprocessor=preprocessor,
        device=device,
        batch_size=args.batch_size,
    )
    t_total_end = time.perf_counter()
    total_elapsed_sec = t_total_end - t_total_start

    # Extrai metadados de tempo e remove do dict antes de salvar
    inference_times_ms: np.ndarray = embeddings_dict.pop(
        "__meta__inference_times_ms", np.array([])
    )

    n_total = len(audio_files)
    n_success = len(embeddings_dict)
    n_fta = n_total - n_success

    mean_inference_ms = float(np.mean(inference_times_ms)) if len(inference_times_ms) > 0 else 0.0
    std_inference_ms = float(np.std(inference_times_ms)) if len(inference_times_ms) > 0 else 0.0

    # -------------------------------------------------------------------------
    # 7.7 Salvamento dos embeddings
    # -------------------------------------------------------------------------
    save_embeddings(
        embeddings_dict=embeddings_dict,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
    )

    # -------------------------------------------------------------------------
    # 7.8 Informações de hardware
    # -------------------------------------------------------------------------
    hardware_info = _get_hardware_info(device)

    # -------------------------------------------------------------------------
    # 7.9 Relatório final — console
    # -------------------------------------------------------------------------
    separator = "=" * 65
    print(f"\n{separator}")
    print("  RELATÓRIO DE EXTRAÇÃO DE EMBEDDINGS")
    print(separator)
    print(f"  Dataset          : {args.dataset}")
    print(f"  Total de arquivos: {n_total}")
    print(f"  Processados (OK) : {n_success}")
    print(f"  FTA (falhas)     : {n_fta}")
    print(f"  Tempo total      : {total_elapsed_sec:.1f}s")
    print(f"  Tempo médio/utt  : {mean_inference_ms:.2f} ms ± {std_inference_ms:.2f} ms")
    print(f"  Hardware         : {hardware_info['type']} — {hardware_info['name']}")
    print(f"  Embeddings em    : {Path(args.output_dir) / args.dataset}")
    print(separator)

    # -------------------------------------------------------------------------
    # 7.10 Salvamento do relatório JSON
    # -------------------------------------------------------------------------
    report = {
        "dataset": args.dataset,
        "n_total": n_total,
        "n_success": n_success,
        "n_fta": n_fta,
        "mean_inference_ms": round(mean_inference_ms, 4),
        "std_inference_ms": round(std_inference_ms, 4),
        "total_elapsed_sec": round(total_elapsed_sec, 2),
        "hardware": hardware_info,
        "model_info": model_info,
        "batch_size": args.batch_size,
        "checkpoint": args.checkpoint,
    }

    results_dir = Path("results") / args.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "inference_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Relatório salvo em: %s", report_path)
    print(f"\n  Relatório JSON salvo em: {report_path}")
    print(separator + "\n")


# ==============================================================================
# Bloco principal
# ==============================================================================

if __name__ == "__main__":
    main()
