"""
evaluate.py — Avaliação completa do sistema de Speaker Verification
Dupla 7 — UTFPR Biometria

Carrega pares de teste/validação, calcula scores de similaridade de cosseno,
computa métricas biométricas (EER, AUC, TAR@FAR, d-prime), gera gráficos e
salva resultados em JSON/CSV.
"""

import numpy as np
import pandas as pd
import json
import yaml
import argparse
import os
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List


# ---------------------------------------------------------------------------
# Funções de similaridade e scores
# ---------------------------------------------------------------------------

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Calcula a similaridade de cosseno entre dois embeddings L2-normalizados.

    Quando os vetores já estão normalizados, a similaridade de cosseno é
    equivalente ao produto interno (dot product) entre eles.

    Parâmetros
    ----------
    emb1 : np.ndarray
        Primeiro embedding (deve estar L2-normalizado).
    emb2 : np.ndarray
        Segundo embedding (deve estar L2-normalizado).

    Retorna
    -------
    float
        Valor de similaridade no intervalo [-1, 1].
    """
    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-10)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-10)
    return float(np.dot(emb1, emb2))


def compute_scores(
    pairs_df: pd.DataFrame,
    embeddings_dict: Dict[str, np.ndarray],
    output_path: str = "results/scores.csv",
) -> pd.DataFrame:
    """
    Calcula scores de similaridade de cosseno para cada par de amostras.

    Itera sobre cada linha de ``pairs_df``, busca os embeddings correspondentes
    em ``embeddings_dict`` e computa a similaridade de cosseno. Pares cujos
    embeddings não forem encontrados são silenciosamente ignorados (warning
    impresso no terminal).

    Parâmetros
    ----------
    pairs_df : pd.DataFrame
        DataFrame com colunas ``amostra_1``, ``amostra_2`` e ``tipo``
        (``"genuino"`` ou ``"impostor"``).
    embeddings_dict : Dict[str, np.ndarray]
        Mapa de identificador de arquivo → vetor de embedding.
    output_path : str, optional
        Caminho para salvar o arquivo ``scores.csv``.

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas: ``amostra_1``, ``amostra_2``, ``tipo``, ``score``.
    """
    rows = []
    missing = set()

    for _, row in pairs_df.iterrows():
        # Normaliza separadores de caminho (\ vs /) para casar com o índice
        a1 = str(Path(str(row["amostra_1"])))
        a2 = str(Path(str(row["amostra_2"])))
        tipo = row["tipo"]

        if a1 not in embeddings_dict:
            missing.add(a1)
            continue
        if a2 not in embeddings_dict:
            missing.add(a2)
            continue

        score = cosine_similarity(embeddings_dict[a1], embeddings_dict[a2])
        rows.append({"amostra_1": a1, "amostra_2": a2, "tipo": tipo, "score": score})

    if missing:
        print(f"[AVISO] {len(missing)} embeddings não encontrados: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")

    scores_df = pd.DataFrame(rows, columns=["amostra_1", "amostra_2", "tipo", "score"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(output_path, index=False)
    print(f"[INFO] Scores salvos em: {output_path} ({len(scores_df)} pares)")

    return scores_df


# ---------------------------------------------------------------------------
# Funções de métricas biométricas
# ---------------------------------------------------------------------------

def compute_eer(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula a Equal Error Rate (EER) por interpolação linear.

    A EER é o ponto na curva FMR/FNMR onde FMR ≈ FNMR. A interpolação linear
    entre os dois pontos mais próximos da interseção é utilizada para maior
    precisão.

    Parâmetros
    ----------
    genuine_scores : np.ndarray
        Scores de pares genuínos.
    impostor_scores : np.ndarray
        Scores de pares impostores.

    Retorna
    -------
    tuple
        (eer_value, threshold_at_eer, fmr_curve, fnmr_curve, thresholds)
        - eer_value       : Taxa de erro no ponto EER (float, 0–1).
        - threshold_at_eer: Threshold correspondente ao EER.
        - fmr_curve       : Array de FMR para cada threshold.
        - fnmr_curve      : Array de FNMR para cada threshold.
        - thresholds      : Array de thresholds avaliados.
    """
    # Conjunto único de thresholds
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(all_scores.min() - 1e-6, all_scores.max() + 1e-6, 1000)

    fmr_curve = np.array([
        np.mean(impostor_scores >= t) for t in thresholds
    ])
    fnmr_curve = np.array([
        np.mean(genuine_scores < t) for t in thresholds
    ])

    # Diferença entre as curvas
    diff = fmr_curve - fnmr_curve

    # Encontra o índice onde a diferença muda de sinal (cruzamento)
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        # Fallback: mínimo da diferença absoluta
        idx = np.argmin(np.abs(diff))
        eer_value = float((fmr_curve[idx] + fnmr_curve[idx]) / 2)
        threshold_at_eer = float(thresholds[idx])
    else:
        idx = sign_changes[0]
        # Interpolação linear entre os dois pontos do cruzamento
        t0, t1 = thresholds[idx], thresholds[idx + 1]
        d0, d1 = diff[idx], diff[idx + 1]

        if d1 != d0:
            t_eer = t0 - d0 * (t1 - t0) / (d1 - d0)
        else:
            t_eer = (t0 + t1) / 2

        threshold_at_eer = float(t_eer)
        fmr_at_eer = float(np.mean(impostor_scores >= t_eer))
        fnmr_at_eer = float(np.mean(genuine_scores < t_eer))
        eer_value = float((fmr_at_eer + fnmr_at_eer) / 2)

    return eer_value, threshold_at_eer, fmr_curve, fnmr_curve, thresholds


def compute_metrics(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float = None,
) -> dict:
    """
    Computa o conjunto completo de métricas biométricas do sistema.

    Métricas calculadas:
    - EER e threshold correspondente
    - FMR e FNMR no threshold fornecido (ou no EER, se None)
    - Curva ROC (FPR, TPR, thresholds)
    - AUC (área sob a curva ROC)
    - TAR@FAR=1% (Taxa de Aceitação Verdadeira a 1% de FAR)
    - Estatísticas das distribuições de scores (min, max, mean, std)
    - d-prime (separabilidade entre distribuições genuína e impostora)

    Parâmetros
    ----------
    genuine_scores : np.ndarray
        Scores de pares genuínos (positivos).
    impostor_scores : np.ndarray
        Scores de pares impostores (negativos).
    threshold : float, optional
        Threshold de decisão. Se None, usa o threshold do EER.

    Retorna
    -------
    dict
        Dicionário com todas as métricas calculadas.
    """
    genuine_scores = np.asarray(genuine_scores, dtype=float)
    impostor_scores = np.asarray(impostor_scores, dtype=float)

    # --- EER ---
    eer_value, threshold_at_eer, fmr_curve, fnmr_curve, eer_thresholds = compute_eer(
        genuine_scores, impostor_scores
    )

    if threshold is None:
        threshold = threshold_at_eer

    # --- FMR e FNMR no threshold escolhido ---
    fmr = float(np.mean(impostor_scores >= threshold))
    fnmr = float(np.mean(genuine_scores < threshold))

    # --- Curva ROC e AUC ---
    y_true = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores)),
    ])
    y_scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    auc_value = float(auc(fpr, tpr))

    # --- TAR @ FAR = 1% ---
    tar_at_far_01 = float(np.interp(0.01, fpr, tpr))

    # --- Estatísticas de distribuição ---
    dist_genuinos = {
        "min": float(genuine_scores.min()),
        "max": float(genuine_scores.max()),
        "mean": float(genuine_scores.mean()),
        "std": float(genuine_scores.std()),
        "n": int(len(genuine_scores)),
    }
    dist_impostores = {
        "min": float(impostor_scores.min()),
        "max": float(impostor_scores.max()),
        "mean": float(impostor_scores.mean()),
        "std": float(impostor_scores.std()),
        "n": int(len(impostor_scores)),
    }

    # --- d-prime ---
    pooled_std = np.sqrt(
        (genuine_scores.std() ** 2 + impostor_scores.std() ** 2) / 2 + 1e-10
    )
    d_prime = float(
        abs(genuine_scores.mean() - impostor_scores.mean()) / pooled_std
    )

    return {
        "eer": eer_value,
        "threshold_at_eer": threshold_at_eer,
        "threshold_usado": threshold,
        "fmr": fmr,
        "fnmr": fnmr,
        "auc": auc_value,
        "tar_at_far_001": tar_at_far_01,
        "d_prime": d_prime,
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "eer_curve": {
            "fmr": fmr_curve.tolist(),
            "fnmr": fnmr_curve.tolist(),
            "thresholds": eer_thresholds.tolist(),
        },
        "distribuicao_genuinos": dist_genuinos,
        "distribuicao_impostores": dist_impostores,
    }


def find_threshold_on_validation(
    val_genuine_scores: np.ndarray,
    val_impostor_scores: np.ndarray,
) -> float:
    """
    Define o threshold congelado a partir do ponto de EER no conjunto de validação.

    O threshold é calculado uma única vez no conjunto de validação e depois
    aplicado fixo ao conjunto de teste, evitando vazamento de informação.

    Parâmetros
    ----------
    val_genuine_scores : np.ndarray
        Scores genuínos do conjunto de validação.
    val_impostor_scores : np.ndarray
        Scores impostores do conjunto de validação.

    Retorna
    -------
    float
        Threshold congelado no ponto de EER da validação.
    """
    _, threshold_at_eer, _, _, _ = compute_eer(val_genuine_scores, val_impostor_scores)
    print(f"[INFO] Threshold congelado na validação (EER): {threshold_at_eer:.6f}")
    return threshold_at_eer


# ---------------------------------------------------------------------------
# Funções de visualização
# ---------------------------------------------------------------------------

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_value: float,
    output_path: str,
    eer_value: float = None,
    threshold_at_eer: float = None,
) -> None:
    """
    Gera e salva a curva ROC do sistema biométrico.

    Exibe a curva ROC com ponto EER marcado, linha de referência aleatória,
    grade, rótulos em português e AUC na legenda.

    Parâmetros
    ----------
    fpr : np.ndarray
        Taxa de Falsa Aceitação (False Positive Rate) por threshold.
    tpr : np.ndarray
        Taxa de Verdadeira Aceitação (True Positive Rate / TAR) por threshold.
    auc_value : float
        Área sob a curva ROC.
    output_path : str
        Caminho de saída do arquivo PNG.
    eer_value : float, optional
        Valor do EER para marcação na curva.
    threshold_at_eer : float, optional
        Threshold correspondente ao EER (usado apenas na anotação).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color="royalblue", lw=2,
            label=f"Curva ROC (AUC = {auc_value:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Classificador aleatório")

    # Marca ponto EER
    if eer_value is not None:
        eer_fpr = eer_value
        eer_tpr = 1.0 - eer_value
        ax.scatter(
            [eer_fpr], [eer_tpr],
            color="red", zorder=5, s=80,
            label=f"EER = {eer_value * 100:.2f}%",
        )
        ax.annotate(
            f"EER\n{eer_value * 100:.2f}%",
            xy=(eer_fpr, eer_tpr),
            xytext=(eer_fpr + 0.05, eer_tpr - 0.1),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="red"),
            color="red",
        )

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Taxa de Falsa Aceitação (FAR / FPR)", fontsize=12)
    ax.set_ylabel("Taxa de Verdadeira Aceitação (TAR / TPR)", fontsize=12)
    ax.set_title("Curva ROC — Verificação de Locutor", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Curva ROC salva em: {output_path}")


def plot_score_distributions(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float,
    output_path: str,
    eer_value: float = None,
) -> None:
    """
    Gera e salva o histograma/KDE das distribuições de scores.

    Exibe as distribuições de scores genuínos e impostores sobrepostas com
    KDE, linha vertical no threshold, áreas de erro (FMR e FNMR) sombreadas
    em cores distintas, e legenda com EER e threshold.

    Parâmetros
    ----------
    genuine_scores : np.ndarray
        Scores dos pares genuínos.
    impostor_scores : np.ndarray
        Scores dos pares impostores.
    threshold : float
        Threshold de decisão (linha vertical).
    output_path : str
        Caminho de saída do arquivo PNG.
    eer_value : float, optional
        Valor do EER para exibição na legenda.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # KDE + histograma
    sns.histplot(
        genuine_scores, kde=True, ax=ax, color="steelblue", alpha=0.4,
        label=f"Genuínos (n={len(genuine_scores)})", stat="density", bins=40,
    )
    sns.histplot(
        impostor_scores, kde=True, ax=ax, color="tomato", alpha=0.4,
        label=f"Impostores (n={len(impostor_scores)})", stat="density", bins=40,
    )

    # Linha do threshold
    ax.axvline(
        threshold, color="black", linestyle="--", lw=2,
        label=f"Threshold = {threshold:.4f}",
    )

    # Área FMR: impostores com score >= threshold (falsos positivos)
    x_fmr = np.linspace(threshold, max(impostor_scores.max(), genuine_scores.max()), 300)
    try:
        from scipy.stats import gaussian_kde
        kde_imp = gaussian_kde(impostor_scores)
        ax.fill_between(x_fmr, kde_imp(x_fmr), alpha=0.3, color="orange",
                        label="Área FMR (falsos positivos)")
    except Exception:
        pass

    # Área FNMR: genuínos com score < threshold (falsos negativos)
    x_fnmr = np.linspace(min(impostor_scores.min(), genuine_scores.min()), threshold, 300)
    try:
        kde_gen = gaussian_kde(genuine_scores)
        ax.fill_between(x_fnmr, kde_gen(x_fnmr), alpha=0.3, color="purple",
                        label="Área FNMR (falsos negativos)")
    except Exception:
        pass

    # Legenda e rótulos
    title = "Distribuição de Scores — Verificação de Locutor"
    if eer_value is not None:
        title += f"\nEER = {eer_value * 100:.2f}%"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Score de Similaridade de Cosseno", fontsize=12)
    ax.set_ylabel("Densidade", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Distribuição de scores salva em: {output_path}")


# ---------------------------------------------------------------------------
# Análise de erros
# ---------------------------------------------------------------------------

def analyze_errors(
    scores_df: pd.DataFrame,
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float,
    n_errors: int = 3,
) -> dict:
    """
    Identifica e retorna os piores erros de classificação do sistema.

    Erros analisados:
    - Falsos Matches (FM): pares impostores aceitos (score >= threshold), ordenados
      pelo score mais alto (pior caso).
    - Falsos Non-Matches (FNM): pares genuínos rejeitados (score < threshold),
      ordenados pelo score mais baixo (pior caso).

    Parâmetros
    ----------
    scores_df : pd.DataFrame
        DataFrame com colunas ``amostra_1``, ``amostra_2``, ``tipo``, ``score``.
    genuine_scores : np.ndarray
        Array de scores genuínos (usado apenas para referência).
    impostor_scores : np.ndarray
        Array de scores impostores (usado apenas para referência).
    threshold : float
        Threshold de decisão.
    n_errors : int, optional
        Número de piores erros a retornar para cada categoria. Default: 3.

    Retorna
    -------
    dict
        Dicionário com chaves ``falsos_matches`` e ``falsos_non_matches``,
        cada uma contendo uma lista de dicts com os campos:
        ``arquivo1``, ``arquivo2``, ``score``, ``threshold``, ``tipo_erro``.
    """
    fm_list = []
    fnm_list = []

    for _, row in scores_df.iterrows():
        score = row["score"]
        tipo = row["tipo"]
        entry = {
            "arquivo1": row["amostra_1"],
            "arquivo2": row["amostra_2"],
            "score": float(score),
            "threshold": float(threshold),
        }
        if tipo == "impostor":
            if score >= threshold:
                entry["tipo_erro"] = "Falso Match (FM)"
                fm_list.append(entry)
        elif tipo in ("genuino", "genuíno", "genuine"):
            if score < threshold:
                entry["tipo_erro"] = "Falso Non-Match (FNM)"
                fnm_list.append(entry)

    # Piores FM: scores mais altos (mais confiantes no erro)
    fm_list.sort(key=lambda x: x["score"], reverse=True)
    # Piores FNM: scores mais baixos (mais distantes do threshold)
    fnm_list.sort(key=lambda x: x["score"])

    result = {
        "falsos_matches": fm_list[:n_errors],
        "falsos_non_matches": fnm_list[:n_errors],
        "total_fm": len(fm_list),
        "total_fnm": len(fnm_list),
        "threshold": float(threshold),
    }

    print(f"[INFO] Análise de erros: {len(fm_list)} FM | {len(fnm_list)} FNM")
    return result


# ---------------------------------------------------------------------------
# Salvamento de resultados
# ---------------------------------------------------------------------------

def save_metrics(metrics_dict: dict, output_path: str) -> None:
    """
    Serializa e salva o dicionário de métricas em formato JSON indentado.

    Campos com arrays numpy são convertidos automaticamente para listas Python.
    Campos da curva ROC completa são omitidos do JSON final para manter o
    arquivo legível (ficam disponíveis apenas no objeto em memória).

    Parâmetros
    ----------
    metrics_dict : dict
        Dicionário com todas as métricas calculadas por ``compute_metrics``.
    output_path : str
        Caminho de saída do arquivo ``metrics.json``.
    """
    # Cria uma cópia compacta sem os arrays longos de ROC
    compact = {k: v for k, v in metrics_dict.items() if k not in ("roc", "eer_curve")}

    # Resumo compacto das curvas
    fpr = np.array(metrics_dict["roc"]["fpr"])
    tpr = np.array(metrics_dict["roc"]["tpr"])
    compact["roc_resumo"] = {
        "n_pontos": len(fpr),
        "fpr_min": float(fpr.min()),
        "fpr_max": float(fpr.max()),
        "tpr_min": float(tpr.min()),
        "tpr_max": float(tpr.max()),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Métricas salvas em: {output_path}")


# ---------------------------------------------------------------------------
# Utilitários de carregamento
# ---------------------------------------------------------------------------

def load_embeddings(embeddings_dir: str, dataset: str) -> Dict[str, np.ndarray]:
    """
    Carrega todos os embeddings a partir do índice CSV e arquivos .npy.

    O índice CSV deve conter as colunas ``file_path`` (identificador do
    arquivo; ``arquivo`` é aceito por compatibilidade) e ``embedding_path``.

    Parâmetros
    ----------
    embeddings_dir : str
        Diretório raiz dos embeddings.
    dataset : str
        Nome do dataset (subdiretório dentro de embeddings_dir).

    Retorna
    -------
    Dict[str, np.ndarray]
        Mapa de identificador → vetor de embedding.
    """
    index_path = Path(embeddings_dir) / dataset / "embeddings_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Índice de embeddings não encontrado: {index_path}")

    index_df = pd.read_csv(index_path)
    embeddings_dict: Dict[str, np.ndarray] = {}

    # Coluna-chave: file_path (extract_embeddings.py) ou arquivo (legado)
    key_col = "file_path" if "file_path" in index_df.columns else "arquivo"

    for _, row in index_df.iterrows():
        key = str(Path(str(row[key_col])))  # normaliza separadores (\ vs /)
        emb_path = Path(str(row["embedding_path"]))

        # Candidatos: caminho como está (abs/relativo ao cwd), relativo ao
        # índice e relativo ao diretório raiz de embeddings.
        candidates = [
            emb_path,
            index_path.parent / emb_path,
            Path(embeddings_dir) / dataset / emb_path,
        ]

        loaded = False
        for cand in candidates:
            if cand.is_file():
                try:
                    embeddings_dict[key] = np.load(str(cand))
                    loaded = True
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"[AVISO] Falha ao carregar {cand}: {e}")
                    break
        if not loaded:
            print(f"[AVISO] Embedding não encontrado para '{key}': {row['embedding_path']}")

    print(f"[INFO] {len(embeddings_dict)} embeddings carregados de '{dataset}'")
    return embeddings_dict


def load_pairs(protocols_dir: str, dataset: str, split: str) -> pd.DataFrame:
    """
    Carrega pares genuínos e impostores de um split (test ou val).

    Espera os arquivos:
    - ``{protocols_dir}/{dataset}/{split}/genuine_pairs.csv``
    - ``{protocols_dir}/{dataset}/{split}_genuine_pairs.csv``

    ATENÇÃO: o fallback para a raiz de ``{dataset}/`` foi removido
    propositalmente para impedir que pares de teste sejam usados
    acidentalmente como validação (vazamento de informação — enunciado §13).

    Parâmetros
    ----------
    protocols_dir : str
        Diretório raiz dos protocolos.
    dataset : str
        Nome do dataset.
    split : str
        Partição desejada: ``"test"`` ou ``"val"``.

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas ``amostra_1``, ``amostra_2``, ``tipo``.
    """
    base = Path(protocols_dir) / dataset

    # Apenas caminhos explícitos do split (sem fallback para a raiz)
    candidates_genuine = [
        base / split / "genuine_pairs.csv",
        base / f"{split}_genuine_pairs.csv",
    ]
    candidates_impostor = [
        base / split / "impostor_pairs.csv",
        base / f"{split}_impostor_pairs.csv",
    ]

    genuine_path = next((p for p in candidates_genuine if p.exists()), None)
    impostor_path = next((p for p in candidates_impostor if p.exists()), None)

    dfs = []
    if genuine_path:
        df_g = pd.read_csv(genuine_path)
        df_g["tipo"] = "genuino"
        dfs.append(df_g)
        print(f"[INFO] Pares genuínos ({split}): {len(df_g)} — {genuine_path}")
    else:
        print(f"[AVISO] Pares genuínos não encontrados para split '{split}'")

    if impostor_path:
        df_i = pd.read_csv(impostor_path)
        df_i["tipo"] = "impostor"
        dfs.append(df_i)
        print(f"[INFO] Pares impostores ({split}): {len(df_i)} — {impostor_path}")
    else:
        print(f"[AVISO] Pares impostores não encontrados para split '{split}'")

    if not dfs:
        raise FileNotFoundError(
            f"Nenhum arquivo de pares encontrado para dataset='{dataset}', split='{split}'"
        )

    pairs_df = pd.concat(dfs, ignore_index=True)

    # Garante colunas mínimas
    if "amostra_1" not in pairs_df.columns or "amostra_2" not in pairs_df.columns:
        # Tenta renomear colunas genéricas
        cols = pairs_df.columns.tolist()
        if len(cols) >= 2:
            pairs_df = pairs_df.rename(columns={cols[0]: "amostra_1", cols[1]: "amostra_2"})

    return pairs_df


# ---------------------------------------------------------------------------
# Impressão de resultados no terminal
# ---------------------------------------------------------------------------

def print_results_table(metrics: dict, dataset: str, threshold: float) -> None:
    """
    Imprime tabela padronizada de resultados no terminal.

    Parâmetros
    ----------
    metrics : dict
        Dicionário de métricas retornado por ``compute_metrics``.
    dataset : str
        Nome do dataset avaliado.
    threshold : float
        Threshold de decisão utilizado.
    """
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  RESULTADOS — Speaker Verification | Dataset: {dataset.upper()}")
    print(sep)
    print(f"  {'Métrica':<35} {'Valor':>12}")
    print(f"  {'-'*35} {'-'*12}")
    print(f"  {'EER':<35} {metrics['eer'] * 100:>11.2f}%")
    print(f"  {'Threshold (EER)':<35} {metrics['threshold_at_eer']:>12.6f}")
    print(f"  {'Threshold usado':<35} {threshold:>12.6f}")
    print(f"  {'FMR @ threshold':<35} {metrics['fmr'] * 100:>11.2f}%")
    print(f"  {'FNMR @ threshold':<35} {metrics['fnmr'] * 100:>11.2f}%")
    print(f"  {'AUC-ROC':<35} {metrics['auc']:>12.4f}")
    print(f"  {'TAR @ FAR=1%':<35} {metrics['tar_at_far_001'] * 100:>11.2f}%")
    print(f"  {'d-prime':<35} {metrics['d_prime']:>12.4f}")
    print(f"  {'N genuínos':<35} {metrics['distribuicao_genuinos']['n']:>12}")
    print(f"  {'N impostores':<35} {metrics['distribuicao_impostores']['n']:>12}")
    print(f"  {'Score genuíno médio':<35} {metrics['distribuicao_genuinos']['mean']:>12.4f}")
    print(f"  {'Score impostor médio':<35} {metrics['distribuicao_impostores']['mean']:>12.4f}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Ponto de entrada principal do script de avaliação.

    Orquestra o pipeline completo:
    1. Leitura de argumentos e configuração
    2. Carregamento de embeddings
    3. Carregamento de pares (validação e teste)
    4. Definição do threshold (validação ou argumento externo)
    5. Cálculo de scores no conjunto de teste
    6. Cálculo de métricas biométricas
    7. Geração de gráficos (ROC, distribuições)
    8. Salvamento de scores.csv e metrics.json
    9. Análise de erros e salvamento de error_analysis.json
    10. Impressão de tabela de resultados
    """
    parser = argparse.ArgumentParser(
        description="Avaliação completa do sistema de Speaker Verification — UTFPR Biometria"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="vctk",
        choices=["vctk", "librispeech"],
        help="Dataset a avaliar (default: vctk)",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="embeddings/",
        help="Diretório raiz dos embeddings (default: embeddings/)",
    )
    parser.add_argument(
        "--protocols_dir",
        type=str,
        default="protocols/",
        help="Diretório raiz dos protocolos de pares (default: protocols/)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/",
        help="Diretório de saída dos resultados (default: results/)",
    )
    parser.add_argument(
        "--val_threshold",
        type=float,
        default=None,
        help="Threshold pré-definido. Se omitido, calcula no conjunto de validação.",
    )
    parser.add_argument(
        "--threshold_from",
        type=str,
        default=None,
        help=(
            "Caminho para metrics.json de OUTRO dataset (cross-dataset). "
            "Carrega threshold_usado da base de origem e aplica na destino, "
            "sem recalcular na validação da destino (enunciado §11/§13)."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Arquivo de configuração YAML (default: config.yaml)",
    )
    args = parser.parse_args()

    # --- Carrega configuração opcional ---
    config = {}
    if Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        print(f"[INFO] Configuração carregada de: {args.config}")

    dataset = config.get("dataset", args.dataset)
    embeddings_dir = config.get("embeddings_dir", args.embeddings_dir)
    protocols_dir = config.get("protocols_dir", args.protocols_dir)
    results_dir = config.get("results_dir", args.results_dir)

    results_path = Path(results_dir) / dataset
    results_path.mkdir(parents=True, exist_ok=True)

    # --- 1. Carrega embeddings ---
    print(f"\n[ETAPA 1] Carregando embeddings ({dataset})...")
    embeddings_dict = load_embeddings(embeddings_dir, dataset)

    # --- 3. Carrega pares de teste ---
    print(f"\n[ETAPA 2] Carregando pares de teste...")
    test_pairs = load_pairs(protocols_dir, dataset, split="test")

    # --- 4. Define threshold ---
    threshold = args.val_threshold
    threshold_origin = None
    if threshold is None and args.threshold_from:
        tf_path = Path(args.threshold_from)
        if not tf_path.exists():
            raise FileNotFoundError(f"metrics.json de origem não encontrado: {tf_path}")
        with open(tf_path, "r", encoding="utf-8") as f:
            src_metrics = json.load(f)
        threshold = float(src_metrics.get("threshold_usado", src_metrics.get("threshold_at_eer")))
        threshold_origin = str(tf_path)
        print(f"[ETAPA 3] Threshold CROSS-DATASET de {tf_path}: {threshold:.6f}")
    if threshold is None:
        print(f"\n[ETAPA 3] Calculando threshold no conjunto de validação...")
        try:
            val_pairs = load_pairs(protocols_dir, dataset, split="val")
            val_scores_df = compute_scores(
                val_pairs,
                embeddings_dict,
                output_path=str(results_path / "val_scores.csv"),
            )
            val_genuine = val_scores_df.loc[
                val_scores_df["tipo"] == "genuino", "score"
            ].values
            val_impostor = val_scores_df.loc[
                val_scores_df["tipo"] == "impostor", "score"
            ].values
            threshold = find_threshold_on_validation(val_genuine, val_impostor)
        except FileNotFoundError as e:
            print(f"[AVISO] Validação não disponível: {e}")
            print("[INFO] Threshold será calculado diretamente no conjunto de teste (EER).")
            threshold = None  # será definido após compute_metrics
    else:
        print(f"[ETAPA 3] Usando threshold: {threshold}")

    # --- 5. Calcula scores no teste ---
    print(f"\n[ETAPA 4] Calculando scores no conjunto de teste...")
    scores_df = compute_scores(
        test_pairs,
        embeddings_dict,
        output_path=str(results_path / "scores.csv"),
    )

    genuine_scores = scores_df.loc[scores_df["tipo"] == "genuino", "score"].values
    impostor_scores = scores_df.loc[scores_df["tipo"] == "impostor", "score"].values

    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        raise ValueError(
            "Scores genuínos ou impostores vazios. Verifique os pares e embeddings."
        )

    # --- 6. Computa métricas ---
    print(f"\n[ETAPA 5] Computando métricas biométricas...")
    metrics = compute_metrics(genuine_scores, impostor_scores, threshold=threshold)

    # Atualiza threshold se foi calculado internamente
    if threshold is None:
        threshold = metrics["threshold_at_eer"]

    # --- 7. Gera gráficos ---
    print(f"\n[ETAPA 6] Gerando gráficos...")
    fpr = np.array(metrics["roc"]["fpr"])
    tpr = np.array(metrics["roc"]["tpr"])

    plot_roc_curve(
        fpr=fpr,
        tpr=tpr,
        auc_value=metrics["auc"],
        output_path=str(results_path / "roc.png"),
        eer_value=metrics["eer"],
        threshold_at_eer=metrics["threshold_at_eer"],
    )

    plot_score_distributions(
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
        threshold=threshold,
        output_path=str(results_path / "distributions.png"),
        eer_value=metrics["eer"],
    )

    # --- 8. Salva métricas ---
    print(f"[ETAPA 7] Salvando métricas...")
    if threshold_origin:
        metrics["threshold_origin"] = threshold_origin
        metrics["threshold_cross_dataset"] = True
    save_metrics(metrics, output_path=str(results_path / "metrics.json"))

    # --- 9. Analisa erros ---
    print(f"\n[ETAPA 8] Analisando erros de classificação...")
    error_analysis = analyze_errors(
        scores_df=scores_df,
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
        threshold=threshold,
        n_errors=3,
    )

    error_path = results_path / "error_analysis.json"
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=4, ensure_ascii=False)
    print(f"[INFO] Análise de erros salva em: {error_path}")

    # --- 10. Imprime tabela ---
    print_results_table(metrics, dataset=dataset, threshold=threshold)

    print(f"[CONCLUÍDO] Todos os resultados salvos em: {results_path.resolve()}\n")


if __name__ == "__main__":
    main()
