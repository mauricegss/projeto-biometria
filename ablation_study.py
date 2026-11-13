"""
ablation_study.py — Experimento E: estudo de ablação do pipeline
Dupla 7 — UTFPR Biometria

Remove ou desabilita etapas do pré-processamento e mede o impacto no
desempenho de verificação de locutor (enunciado §16.5).

Variantes:
  - full        : pipeline completo (VAD + CMN) — referência (Exp. A)
  - no_vad      : sem Voice Activity Detection (áudio completo)
  - no_cmn      : sem Cepstral Mean Normalization
  - no_vad_no_cmn : sem VAD e sem CMN

Para cada variante reextraí embeddings de val+test com o preprocessor
modificado, congela threshold na validação da própria variante e avalia
no teste (mesmo protocolo anti-leak do enunciado §13).

Uso:
  python ablation_study.py --dataset vctk
  python ablation_study.py --dataset vctk --variants full no_vad
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from evaluate import (
    compute_metrics,
    compute_scores,
    find_threshold_on_validation,
    load_pairs,
    save_metrics,
)
from extract_embeddings import extract_embedding, load_model
from preprocess import AudioPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ablation_study")

VARIANTS: Dict[str, Dict[str, bool]] = {
    "full": {"vad_enabled": True, "normalize": True},
    "no_vad": {"vad_enabled": False, "normalize": True},
    "no_cmn": {"vad_enabled": True, "normalize": False},
    "no_vad_no_cmn": {"vad_enabled": False, "normalize": False},
}


def extract_embeddings_for_csvs(
    model,
    file_list: List[str],
    preprocessor: AudioPreprocessor,
    device,
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """Extrai embeddings de uma lista de arquivos (batching + pad)."""
    model.eval()
    embeddings: Dict[str, np.ndarray] = {}
    failed = 0

    for batch_start in tqdm(range(0, len(file_list), batch_size), desc="Extraindo", unit="batch"):
        batch_paths = file_list[batch_start: batch_start + batch_size]
        valid_paths: List[str] = []
        valid_specs = []
        for fp in batch_paths:
            try:
                spec = preprocessor.process(fp)
                valid_paths.append(fp)
                valid_specs.append(spec)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FTA '%s': %s", fp, exc)
                failed += 1
        if not valid_specs:
            continue
        max_len = max(s.shape[-1] for s in valid_specs)
        lens = torch.tensor([s.shape[-1] / max_len for s in valid_specs], dtype=torch.float)
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

    logger.info("Extraídos=%d FTA=%d", len(embeddings), failed)
    return embeddings


def unique_files_from_pairs(*pairs_dfs: pd.DataFrame) -> List[str]:
    files = set()
    for df in pairs_dfs:
        files |= set(df["amostra_1"].astype(str))
        files |= set(df["amostra_2"].astype(str))
    return sorted(files)


def evaluate_variant(
    name: str,
    embeddings: Dict[str, np.ndarray],
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    results_path: Path,
) -> dict:
    out = results_path / name
    out.mkdir(parents=True, exist_ok=True)
    # Threshold na VALIDAÇÃO da variante (anti-leak)
    val_scores = compute_scores(
        val_pairs, embeddings, output_path=str(out / "val_scores.csv")
    )
    val_g = val_scores.loc[val_scores["tipo"] == "genuino", "score"].values
    val_i = val_scores.loc[val_scores["tipo"] == "impostor", "score"].values
    if len(val_g) == 0 or len(val_i) == 0:
        raise ValueError(f"Valores de validação vazios na variante '{name}'")
    threshold = find_threshold_on_validation(val_g, val_i)

    test_scores = compute_scores(
        test_pairs, embeddings, output_path=str(out / "scores.csv")
    )
    test_g = test_scores.loc[test_scores["tipo"] == "genuino", "score"].values
    test_i = test_scores.loc[test_scores["tipo"] == "impostor", "score"].values
    metrics = compute_metrics(test_g, test_i, threshold=threshold)
    metrics["variant"] = name
    metrics["threshold_usado"] = threshold

    save_metrics(metrics, output_path=str(out / "metrics.json"))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimento E — ablação do pipeline (Dupla 7)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="vctk", choices=["vctk", "librispeech"])
    parser.add_argument("--embeddings_dir", type=str, default="embeddings/")
    parser.add_argument("--protocols_dir", type=str, default="protocols/")
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--variants", type=str, nargs="+",
                        default=["full", "no_vad", "no_cmn", "no_vad_no_cmn"],
                        choices=list(VARIANTS.keys()))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--reuse_full", action="store_true",
                        help="Reusa results/{dataset}/metrics.json como baseline 'full'")
    args = parser.parse_args()

    config = {}
    if Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    dataset = config.get("dataset", args.dataset)
    base_pre_cfg = dict(config.get("preprocessing", {}))

    results_path = Path(args.results_dir) / dataset / "ablation"
    results_path.mkdir(parents=True, exist_ok=True)

    val_pairs = load_pairs(args.protocols_dir, dataset, split="val")
    test_pairs = load_pairs(args.protocols_dir, dataset, split="test")
    files = unique_files_from_pairs(val_pairs, test_pairs)
    logger.info("Arquivos únicos (val+test): %d", len(files))

    model, device, _ = load_model(checkpoint_path=args.checkpoint, device="auto")

    all_metrics = []
    t0 = time.perf_counter()

    for variant in args.variants:
        logger.info("=== Variante: %s ===", variant)

        if variant == "full" and args.reuse_full:
            ref = Path(args.results_dir) / dataset / "metrics.json"
            if ref.exists():
                with open(ref, "r", encoding="utf-8") as f:
                    m = json.load(f)
                m["variant"] = "full"
                m["reused"] = True
                # Salva cópia no diretório de ablação (JSON puro — sem replot)
                out = results_path / "full"
                out.mkdir(parents=True, exist_ok=True)
                with open(out / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(m, f, indent=2, ensure_ascii=False)
                all_metrics.append(m)
                logger.info("Reusado metrics.json do Exp. A (full)")
                continue

        overrides = VARIANTS[variant]
        pre_cfg = {**base_pre_cfg, **overrides}
        preprocessor = AudioPreprocessor(config=pre_cfg)

        emb_start = time.perf_counter()
        embeddings = extract_embeddings_for_csvs(
            model=model,
            file_list=files,
            preprocessor=preprocessor,
            device=device,
            batch_size=args.batch_size,
        )
        logger.info(
            "Extração %s: %.1fs (%d embeddings)",
            variant, time.perf_counter() - emb_start, len(embeddings),
        )

        metrics = evaluate_variant(variant, embeddings, val_pairs, test_pairs, results_path)
        all_metrics.append(metrics)
        logger.info(
            "%s → EER=%.4f%% | FMR=%.4f%% | FNMR=%.4f%% | thr=%.4f",
            variant, 100 * metrics["eer"], 100 * metrics["fmr"],
            100 * metrics["fnmr"], metrics["threshold_usado"],
        )

    elapsed = time.perf_counter() - t0

    rows = []
    for m in all_metrics:
        rows.append({
            "variant": m.get("variant"),
            "eer": m.get("eer"),
            "eer_pct": 100 * m.get("eer", float("nan")),
            "auc": m.get("auc"),
            "fmr": m.get("fmr"),
            "fnmr": m.get("fnmr"),
            "tar_at_far_001": m.get("tar_at_far_001"),
            "d_prime": m.get("d_prime"),
            "threshold_usado": m.get("threshold_usado"),
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(results_path / "ablation_summary.csv", index=False)

    with open(results_path / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset,
                "variants": args.variants,
                "n_files": len(files),
                "elapsed_sec": elapsed,
                "results": all_metrics,
            },
            f, indent=2, ensure_ascii=False,
        )

    print("\n" + "=" * 90)
    print(f"ABLAÇÃO — {dataset}")
    print("=" * 90)
    print(f"{'Variante':<18} {'EER%':>8} {'AUC':>8} {'FMR%':>8} {'FNMR%':>8} {'TAR@1%':>8}")
    print("-" * 90)
    for m in all_metrics:
        print(
            f"{m.get('variant', '?'):<18} {100*m.get('eer', float('nan')):>8.3f} "
            f"{m.get('auc', float('nan')):>8.4f} {100*m.get('fmr', float('nan')):>8.3f} "
            f"{100*m.get('fnmr', float('nan')):>8.3f} "
            f"{100*m.get('tar_at_far_001', float('nan')):>8.2f}"
        )
    print("=" * 90)
    print(f"[CONCLUÍDO] Resultados em {results_path.resolve()} ({elapsed:.0f}s)\n")


if __name__ == "__main__":
    main()
