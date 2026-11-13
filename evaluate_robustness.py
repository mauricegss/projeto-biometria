"""
evaluate_robustness.py — Experimento D: robustez a ruído aditivo
Dupla 7 — UTFPR Biometria

Aplica ruído aditivo (branco ou babble) em níveis de SNR variados aos arquivos
do conjunto de teste e reavalia o sistema de verificação de locutor.

Condições:
  - original (embeddings já extraídos, sem degradação)
  - white / babble × SNR ∈ {20, 10, 5, 0} dB  (enunciado: ≥4 níveis:
    original, baixa, média e alta degradação)

Threshold operacional: congelado no conjunto de validação limpo (Exp. A) —
mesmo protocolo anti-leak do enunciado §13. EER/AUC são recalculados por
condição (não dependem do threshold).

Uso:
  python evaluate_robustness.py --dataset vctk
  python evaluate_robustness.py --dataset vctk --snr_levels 20 10 --noise_types white
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

from evaluate import (
    compute_metrics,
    compute_scores,
    find_threshold_on_validation,
    load_pairs,
    plot_roc_curve,
    plot_score_distributions,
    save_metrics,
)
from extract_embeddings import extract_embedding, load_model
from preprocess import AudioPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_robustness")


# ---------------------------------------------------------------------------
# Ruído
# ---------------------------------------------------------------------------

def _load_waveform_16k(path: Path, target_sr: int = 16000) -> torch.Tensor:
    """Carrega áudio mono float32 em (1, N), reamostrando se necessário."""
    data, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T.copy())
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != target_sr:
        import torchaudio.transforms as T

        resampler = T.Resample(
            orig_freq=orig_sr,
            new_freq=target_sr,
            resampling_method="sinc_interp_hann",
        )
        waveform = resampler(waveform)
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform  # (1, N)


def build_babble_pool(
    data_dir: Path,
    sample_rate: int,
    n_files: int = 40,
    seed: int = 42,
) -> List[torch.Tensor]:
    """Seleciona utterances de fala para misturar como babble."""
    rng = np.random.default_rng(seed)
    candidates = sorted(
        list(data_dir.rglob("*.wav")) + list(data_dir.rglob("*.flac"))
    )
    if not candidates:
        raise FileNotFoundError(f"Nenhum áudio em {data_dir} para babble")
    chosen = rng.choice(len(candidates), size=min(n_files, len(candidates)), replace=False)
    pool = []
    for idx in chosen:
        try:
            pool.append(_load_waveform_16k(Path(candidates[int(idx)]), sample_rate))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falha ao carregar babble '%s': %s", candidates[int(idx)], exc)
    if not pool:
        raise RuntimeError("Pool de babble vazio")
    return pool


def make_noise(
    noise_type: str,
    length: int,
    sample_rate: int,
    babble_pool: Optional[List[torch.Tensor]] = None,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Gera sinal de ruído de shape (1, length)."""
    rng = rng or np.random.default_rng()
    if noise_type == "white":
        noise = torch.randn(1, length)
        return noise / (noise.pow(2).mean().sqrt() + 1e-10)
    if noise_type == "babble":
        if not babble_pool:
            raise ValueError("babble_pool vazio")
        noise = torch.zeros(1, length)
        n_segments = int(rng.integers(3, 7))
        for _ in range(n_segments):
            seg = babble_pool[int(rng.integers(0, len(babble_pool)))]
            if seg.shape[-1] < 1600:  # < 0.1 s
                continue
            start = int(rng.integers(0, max(1, seg.shape[-1] - length)))
            chunk = seg[:, start: start + length]
            if chunk.shape[-1] < length:
                chunk = torch.nn.functional.pad(chunk, (0, length - chunk.shape[-1]))
            noise = noise + chunk
        noise = noise / (n_segments + 1e-10)
        rms = noise.pow(2).mean().sqrt()
        if rms > 0:
            noise = noise / rms
        return noise
    raise ValueError(f"Tipo de ruído desconhecido: {noise_type}")


def add_noise_at_snr(
    signal: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    """Soma ``noise`` a ``signal`` atingindo o SNR alvo (dB)."""
    if noise.shape[-1] < signal.shape[-1]:
        reps = signal.shape[-1] // noise.shape[-1] + 1
        noise = noise.repeat(1, reps)
    noise = noise[:, : signal.shape[-1]]
    ps = signal.pow(2).mean()
    pn = noise.pow(2).mean()
    if pn <= 0 or ps <= 0:
        return signal
    factor = torch.sqrt(ps / (pn * (10.0 ** (snr_db / 10.0))))
    return signal + noise * factor


# ---------------------------------------------------------------------------
# Extração de embeddings para uma condição
# ---------------------------------------------------------------------------

def extract_for_files(
    model: torch.nn.Module,
    file_list: List[str],
    preprocessor: AudioPreprocessor,
    device: torch.device,
    condition: str,
    babble_pool: Optional[List[torch.Tensor]] = None,
    snr_db: Optional[float] = None,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Extrai embeddings de ``file_list`` sob uma condição de degradação.

    ``condition == "original"`` usa o pipeline limpo (pré-processamento padrão).
    Caso contrário, o waveform é degradado em memória antes do pré-processador.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    embeddings: Dict[str, np.ndarray] = {}
    failed: List[str] = []

    # Pré-carrega waveform limpa por arquivo (reutiliza entre condições)
    # Aqui carregamos sob demanda para economizar RAM.
    n = len(file_list)
    progress = tqdm(range(0, n, batch_size), desc=f"Condição {condition}", unit="batch")

    for batch_start in progress:
        batch_paths = file_list[batch_start: batch_start + batch_size]
        valid_paths: List[str] = []
        valid_specs: List[torch.Tensor] = []

        for file_path in batch_paths:
            try:
                key = str(Path(str(file_path)))  # normaliza \ vs /
                if condition == "original":
                    spec = preprocessor.process(file_path)
                else:
                    waveform = _load_waveform_16k(Path(file_path), preprocessor.sample_rate)
                    noise = make_noise(
                        noise_type=condition.split("@")[0],
                        length=waveform.shape[-1],
                        sample_rate=preprocessor.sample_rate,
                        babble_pool=babble_pool,
                        rng=rng,
                    )
                    noisy = add_noise_at_snr(waveform, noise, float(snr_db))
                    spec = preprocessor.process_waveform(noisy, source=file_path)
                valid_paths.append(key)
                valid_specs.append(spec)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FTA '%s' [%s]: %s", file_path, condition, exc)
                failed.append(file_path)

        if not valid_specs:
            continue

        max_len = max(s.shape[-1] for s in valid_specs)
        lens = torch.tensor(
            [s.shape[-1] / max_len for s in valid_specs], dtype=torch.float
        )
        padded = []
        for spec in valid_specs:
            pad = max_len - spec.shape[-1]
            if pad > 0:
                spec = torch.nn.functional.pad(spec, (0, pad))
            padded.append(spec)
        batch_tensor = torch.stack(padded, dim=0).to(device)

        with torch.no_grad():
            batch_emb = model(batch_tensor, lens=lens.to(device))

        batch_np = batch_emb.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(batch_np, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        batch_np = batch_np / norms
        for fp, emb in zip(valid_paths, batch_np):
            embeddings[fp] = emb

        progress.set_postfix(ok=len(embeddings), fta=len(failed))

    if failed:
        logger.warning("[%s] %d FTA(s).", condition, len(failed))
    return embeddings


# ---------------------------------------------------------------------------
# Avaliação de uma condição
# ---------------------------------------------------------------------------

def evaluate_condition(
    pairs_df: pd.DataFrame,
    embeddings: Dict[str, np.ndarray],
    frozen_threshold: float,
    results_path: Path,
    condition: str,
) -> dict:
    cond_dir = results_path / condition
    cond_dir.mkdir(parents=True, exist_ok=True)
    scores_df = compute_scores(
        pairs_df, embeddings, output_path=str(cond_dir / "scores.csv")
    )
    genuine = scores_df.loc[scores_df["tipo"] == "genuino", "score"].values
    impostor = scores_df.loc[scores_df["tipo"] == "impostor", "score"].values
    if len(genuine) == 0 or len(impostor) == 0:
        raise ValueError(f"Scores vazios na condição '{condition}'")

    # EER/AUC por condição; FMR/FNMR no threshold congelado (validação limpa)
    metrics = compute_metrics(genuine, impostor, threshold=frozen_threshold)
    metrics["condition"] = condition
    metrics["frozen_threshold"] = frozen_threshold
    metrics["n_pairs_scored"] = int(len(scores_df))

    save_metrics(metrics, output_path=str(cond_dir / "metrics.json"))

    fpr = np.array(metrics["roc"]["fpr"])
    tpr = np.array(metrics["roc"]["tpr"])
    plot_roc_curve(
        fpr=fpr,
        tpr=tpr,
        auc_value=metrics["auc"],
        output_path=str(cond_dir / "roc.png"),
        eer_value=metrics["eer"],
        threshold_at_eer=metrics["threshold_at_eer"],
    )
    plot_score_distributions(
        genuine_scores=genuine,
        impostor_scores=impostor,
        threshold=frozen_threshold,
        output_path=str(cond_dir / "distributions.png"),
        eer_value=metrics["eer"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimento D — robustez a ruído aditivo (Dupla 7)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="vctk", choices=["vctk", "librispeech"])
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Dataset root (default: data/{dataset}/wav48_silence_trimmed ou train-clean-100)")
    parser.add_argument("--embeddings_dir", type=str, default="embeddings/")
    parser.add_argument("--protocols_dir", type=str, default="protocols/")
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--noise_types", type=str, nargs="+", default=["white", "babble"])
    parser.add_argument("--snr_levels", type=float, nargs="+", default=[20.0, 10.0, 5.0, 0.0])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_original", action="store_true",
                        help="Não reprocessa a condição original (usa embeddings existentes)")
    args = parser.parse_args()

    config = {}
    if Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    dataset = config.get("dataset", args.dataset)
    pre_cfg = config.get("preprocessing", {})
    sample_rate = int(pre_cfg.get("sample_rate", 16000))

    if args.data_dir is None:
        if dataset == "vctk":
            args.data_dir = "data/vctk/wav48_silence_trimmed"
        else:
            args.data_dir = "data/librispeech/train-clean-100"

    results_path = Path(args.results_dir) / dataset / "robustness"
    results_path.mkdir(parents=True, exist_ok=True)

    # Threshold congelado: metrics.json do Exp. A (validação limpa)
    clean_metrics_path = Path(args.results_dir) / dataset / "metrics.json"
    if not clean_metrics_path.exists():
        raise FileNotFoundError(
            f"Rode evaluate.py antes (Exp. A) para congelar o threshold: {clean_metrics_path}"
        )
    with open(clean_metrics_path, "r", encoding="utf-8") as f:
        clean_metrics = json.load(f)
    frozen_threshold = float(clean_metrics.get("threshold_usado", clean_metrics["threshold_at_eer"]))
    logger.info("Threshold congelado (validação limpa): %.6f", frozen_threshold)

    # Modelo + pré-processador
    model, device, _ = load_model(checkpoint_path=args.checkpoint, device="auto")
    preprocessor = AudioPreprocessor(config=pre_cfg)

    # Pares de teste
    test_pairs = load_pairs(args.protocols_dir, dataset, split="test")

    # Arquivos únicos no protocolo de teste (normaliza separadores \ vs /)
    files: List[str] = sorted(
        {
            str(Path(str(p)))
            for p in set(test_pairs["amostra_1"].astype(str))
            | set(test_pairs["amostra_2"].astype(str))
        }
    )
    logger.info("Arquivos únicos no teste: %d | pares: %d", len(files), len(test_pairs))

    # Pool de babble (fala de outros locutors do próprio dataset de treino)
    babble_pool = None
    if "babble" in args.noise_types:
        babble_pool = build_babble_pool(
            Path(args.data_dir), sample_rate, n_files=40, seed=args.seed
        )
        logger.info("Pool de babble: %d utterances", len(babble_pool))

    conditions: List[Tuple[str, Optional[float]]] = []
    if not args.skip_original:
        conditions.append(("original", None))
    for noise_type in args.noise_types:
        for snr in args.snr_levels:
            conditions.append((noise_type, float(snr)))

    # Embeddings originais já existentes (economia de tempo)
    original_cache: Optional[Dict[str, np.ndarray]] = None
    if any(c == "original" for c, _ in conditions):
        try:
            from evaluate import load_embeddings

            original_cache = load_embeddings(args.embeddings_dir, dataset)
            # Mantém apenas arquivos do teste
            original_cache = {k: v for k, v in original_cache.items() if k in set(files)}
            logger.info("Embeddings originais carregados: %d", len(original_cache))
        except FileNotFoundError:
            logger.warning("Índice de embeddings original não encontrado — reextrairá.")

    all_results = []
    t_start = time.perf_counter()

    for condition, snr in tqdm(conditions, desc="Condições"):
        label = "original" if condition == "original" else f"{condition}@{int(snr)}dB"
        logger.info("=== Condição: %s ===", label)

        if condition == "original" and original_cache is not None and len(original_cache) >= len(files) * 0.9:
            embeddings = original_cache
        else:
            embeddings = extract_for_files(
                model=model,
                file_list=files,
                preprocessor=preprocessor,
                device=device,
                condition=condition,
                babble_pool=babble_pool,
                snr_db=snr,
                batch_size=args.batch_size,
                seed=args.seed,
            )

        metrics = evaluate_condition(
            pairs_df=test_pairs,
            embeddings=embeddings,
            frozen_threshold=frozen_threshold,
            results_path=results_path,
            condition=label,
        )
        all_results.append(metrics)
        logger.info(
            "%s → EER=%.4f%% | AUC=%.4f | FMR@thr=%.4f%% | FNMR@thr=%.4f%%",
            label,
            100 * metrics["eer"],
            metrics["auc"],
            100 * metrics["fmr"],
            100 * metrics["fnmr"],
        )

    elapsed = time.perf_counter() - t_start

    # Tabela resumo
    rows = []
    for m in all_results:
        rows.append({
            "condition": m["condition"],
            "eer": m["eer"],
            "eer_pct": 100 * m["eer"],
            "auc": m["auc"],
            "fmr": m["fmr"],
            "fnmr": m["fnmr"],
            "tar_at_far_001": m.get("tar_at_far_001"),
            "d_prime": m.get("d_prime"),
            "threshold_usado": m.get("frozen_threshold", frozen_threshold),
            "n_pairs": m.get("n_pairs_scored"),
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(results_path / "robustness_summary.csv", index=False)

    summary = {
        "dataset": dataset,
        "frozen_threshold": frozen_threshold,
        "threshold_source": str(clean_metrics_path),
        "noise_types": args.noise_types,
        "snr_levels_db": args.snr_levels,
        "n_test_files": len(files),
        "n_test_pairs": len(test_pairs),
        "elapsed_sec": elapsed,
        "conditions": all_results,
    }
    with open(results_path / "robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print(f"ROBUSTEZ — {dataset} — threshold congelado={frozen_threshold:.4f}")
    print("=" * 90)
    print(f"{'Condição':<18} {'EER%':>8} {'AUC':>8} {'FMR%':>8} {'FNMR%':>8} {'TAR@1%':>8}")
    print("-" * 90)
    for m in all_results:
        print(
            f"{m['condition']:<18} {100*m['eer']:>8.3f} {m['auc']:>8.4f} "
            f"{100*m['fmr']:>8.3f} {100*m['fnmr']:>8.3f} "
            f"{100*m.get('tar_at_far_001', float('nan')):>8.2f}"
        )
    print("=" * 90)
    print(f"[CONCLUÍDO] Resultados em {results_path.resolve()} ({elapsed:.0f}s)\n")


if __name__ == "__main__":
    main()
