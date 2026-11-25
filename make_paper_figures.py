"""
Gera as figuras e tabelas do artigo e da apresentação em ``results/paper/``.

Tudo vem dos resultados já salvos em ``results/`` e dos metadados das bases:

- ``errors_librispeech.png``: log-Mel dos 3 falsos matches e 3 falsos
  non-matches de maior erro no LibriSpeech (pares da Dupla 8, Exp. B);
- ``speaker_6209.png``: score médio de cada utterance do capítulo 34601
  contra as outras utterances do locutor 6209 (hipótese da análise de erros);
- ``robustness.png``: EER por SNR no VCTK (Exp. D);
- ``pipeline_example.png``: uma amostra em cada etapa do pré-processamento;
- ``error_cases.csv``: os 12 casos analisados (6 por base), com score,
  duração, sexo, sotaque e microfone quando disponíveis;
- ``error_stats.json``: estatísticas agregadas usadas na discussão.

Requer ``data/librispeech/train-clean-100`` (áudio) e
``data/vctk/speaker-info.txt`` (metadados do VCTK 0.92).

Uso:
    python make_paper_figures.py
"""

import json
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

OUT = Path("results/paper")
LIBRI_ERRORS = Path("results/dupla8/librispeech/error_analysis.json")
LIBRI_SCORES = Path("results/dupla8/librispeech/scores.csv")
VCTK_ERRORS = Path("results/vctk/error_analysis.json")
VCTK_SCORES = Path("results/vctk/scores.csv")
ROBUSTNESS = Path("results/vctk/robustness/robustness_summary.csv")
LIBRI_SPEAKERS = Path("data/librispeech/SPEAKERS.TXT")
VCTK_SPEAKERS = Path("data/vctk/speaker-info.txt")
PIPELINE_SAMPLE = "data/librispeech/train-clean-100/6209/34599/6209-34599-0000.flac"

plt.rcParams.update({"font.size": 9, "figure.dpi": 200, "savefig.bbox": "tight"})


def posix(p: str) -> str:
    return str(p).replace("\\", "/")


def libri_speakers() -> dict:
    sex = {}
    for line in LIBRI_SPEAKERS.read_text(encoding="utf-8").splitlines():
        if line.startswith(";"):
            continue
        parts = [x.strip() for x in line.split("|")]
        if len(parts) >= 2:
            sex[parts[0]] = parts[1]
    return sex


def vctk_speakers() -> dict:
    meta = {}
    for line in VCTK_SPEAKERS.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0].startswith("p"):
            meta[parts[0]] = {"sexo": parts[2], "sotaque": parts[3]}
    return meta


def logmel(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    mel = librosa.feature.melspectrogram(
        y=y, sr=16000, n_fft=512, win_length=400, hop_length=160,
        n_mels=80, fmin=20, fmax=7600,
    )
    return librosa.power_to_db(mel, ref=np.max)


def error_cases() -> pd.DataFrame:
    sex = libri_speakers()
    vmeta = vctk_speakers()
    rows = []
    for base, path in (("LibriSpeech", LIBRI_ERRORS), ("VCTK", VCTK_ERRORS)):
        data = json.loads(path.read_text(encoding="utf-8"))
        for kind, key in (("FM", "falsos_matches"), ("FNM", "falsos_non_matches")):
            for i, case in enumerate(data[key][:3], start=1):
                row = {"base": base, "tipo": kind, "caso": f"{kind}{i}",
                       "score": case["score"], "threshold": case["threshold"]}
                for side in (1, 2):
                    f = posix(case[f"arquivo{side}"])
                    row[f"amostra_{side}"] = Path(f).stem
                    if base == "LibriSpeech":
                        spk = f.split("/")[3]
                        row[f"sexo_{side}"] = sex.get(spk, "")
                        row[f"duracao_{side}_s"] = round(sf.info(f).duration, 1)
                    else:
                        spk = f.split("/")[-2]
                        row[f"sexo_{side}"] = vmeta.get(spk, {}).get("sexo", "")
                        row[f"sotaque_{side}"] = vmeta.get(spk, {}).get("sotaque", "")
                        row[f"microfone_{side}"] = Path(f).stem.split("_")[-1]
                rows.append(row)
    return pd.DataFrame(rows)


def plot_errors(cases: pd.DataFrame) -> None:
    data = json.loads(LIBRI_ERRORS.read_text(encoding="utf-8"))
    picked = [("FM", c) for c in data["falsos_matches"][:3]] + \
             [("FNM", c) for c in data["falsos_non_matches"][:3]]
    fig, axes = plt.subplots(len(picked), 2, figsize=(7.2, 8.4))
    for r, (kind, case) in enumerate(picked):
        for c, side in enumerate((1, 2)):
            f = posix(case[f"arquivo{side}"])
            ax = axes[r, c]
            librosa.display.specshow(logmel(f), sr=16000, hop_length=160,
                                     x_axis="time", y_axis="mel", fmin=20,
                                     fmax=7600, ax=ax, cmap="magma")
            ax.set_title(Path(f).stem, fontsize=7)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelsize=6)
        axes[r, 0].set_ylabel(f"{kind}{r % 3 + 1}\nscore {case['score']:.3f}", fontsize=8)
    fig.suptitle("Log-Mel dos erros no LibriSpeech (threshold 0,353)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "errors_librispeech.png")
    plt.close(fig)


def speaker_6209(stats: dict) -> None:
    s = pd.read_csv(LIBRI_SCORES)
    g = s[(s.tipo == "genuino") & s.amostra_1.str.contains("/6209/")]
    stem = lambda p: Path(posix(p)).stem  # noqa: E731
    rows = []
    for a, b, sc in zip(g.amostra_1, g.amostra_2, g.score):
        rows += [(stem(a), stem(b), sc), (stem(b), stem(a), sc)]
    r = pd.DataFrame(rows, columns=["utt", "other", "score"])
    r["ch"] = r.utt.str.split("-").str[1]
    r["och"] = r.other.str.split("-").str[1]
    per_utt = r[(r.ch == "34601") & (r.och != "34601")].groupby("utt").score.mean()
    per_utt.index = [int(u.split("-")[2]) for u in per_utt.index]
    per_utt = per_utt.sort_index()

    thr = json.loads(LIBRI_ERRORS.read_text(encoding="utf-8"))["threshold"]
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    colors = ["#c0392b" if v < thr else "#2c7fb8" for v in per_utt.values]
    ax.bar(per_utt.index.astype(str), per_utt.values, color=colors)
    ax.axhline(thr, color="black", lw=0.8, ls="--", label=f"threshold {thr:.3f}")
    ax.set_xlabel("utterance do capítulo 34601")
    ax.set_ylabel("score médio")
    ax.set_title("Locutor 6209: score de cada utterance do cap. 34601 contra os caps. 34599/34600")
    ax.tick_params(axis="x", labelsize=6)
    ax.legend(fontsize=7)
    fig.savefig(OUT / "speaker_6209.png")
    plt.close(fig)

    has = (g.amostra_1.str.contains("/34601/")) | (g.amostra_2.str.contains("/34601/"))
    stats["speaker_6209"] = {
        "utterances_abaixo_do_threshold": [int(i) for i, v in per_utt.items() if v < thr],
        "score_medio_pares_com_34601": float(g[has].score.mean()),
        "score_medio_pares_sem_34601": float(g[~has].score.mean()),
    }


def aggregate_stats(stats: dict) -> None:
    sex = libri_speakers()
    s = pd.read_csv(LIBRI_SCORES)
    thr = json.loads(LIBRI_ERRORS.read_text(encoding="utf-8"))["threshold"]
    spk = lambda p: posix(p).split("/")[3]  # noqa: E731
    gen = s[s.tipo == "genuino"]
    imp = s[s.tipo == "impostor"].copy()
    imp["mesmo_sexo"] = [sex[spk(a)] == sex[spk(b)] for a, b in zip(imp.amostra_1, imp.amostra_2)]
    fnm = gen[gen.score < thr]
    stats["librispeech"] = {
        "n_fm": int((imp.score >= thr).sum()),
        "n_fnm": int(len(fnm)),
        "fnm_por_locutor": fnm.amostra_1.map(spk).value_counts().to_dict(),
        "fnm_com_cap_34601": int((fnm.amostra_1.str.contains("/34601/") |
                                  fnm.amostra_2.str.contains("/34601/")).sum()),
        "fmr_mesmo_sexo": float((imp[imp.mesmo_sexo].score >= thr).mean()),
        "fmr_sexo_diferente": float((imp[~imp.mesmo_sexo].score >= thr).mean()),
        "fracao_fm_mesmo_sexo": float(imp[imp.score >= thr].mesmo_sexo.mean()),
        "fracao_impostores_mesmo_sexo": float(imp.mesmo_sexo.mean()),
    }

    vmeta = vctk_speakers()
    v = pd.read_csv(VCTK_SCORES)
    vthr = json.loads(VCTK_ERRORS.read_text(encoding="utf-8"))["threshold"]
    vspk = lambda p: posix(p).split("/")[-2]  # noqa: E731
    mic = lambda p: Path(posix(p)).stem.split("_")[-1]  # noqa: E731
    vimp = v[v.tipo == "impostor"].copy()
    vimp["mesmo_sexo"] = [vmeta[vspk(a)]["sexo"] == vmeta[vspk(b)]["sexo"]
                          for a, b in zip(vimp.amostra_1, vimp.amostra_2)]
    vimp["mesmo_sotaque"] = [vmeta[vspk(a)]["sotaque"] == vmeta[vspk(b)]["sotaque"]
                             for a, b in zip(vimp.amostra_1, vimp.amostra_2)]
    vfm = vimp[vimp.score >= vthr]
    vgen = v[v.tipo == "genuino"].copy()
    vgen["mic_diferente"] = [mic(a) != mic(b) for a, b in zip(vgen.amostra_1, vgen.amostra_2)]
    pairs = vfm.apply(lambda r: "-".join(sorted((vspk(r.amostra_1), vspk(r.amostra_2)))), axis=1)
    stats["vctk"] = {
        "n_fm": int(len(vfm)),
        "n_fnm": int((vgen.score < vthr).sum()),
        "fracao_fm_mesmo_sexo": float(vfm.mesmo_sexo.mean()),
        "fracao_impostores_mesmo_sexo": float(vimp.mesmo_sexo.mean()),
        "fracao_fm_mesmo_sotaque": float(vfm.mesmo_sotaque.mean()),
        "fracao_impostores_mesmo_sotaque": float(vimp.mesmo_sotaque.mean()),
        "fm_por_par_de_locutores": pairs.value_counts().head(5).to_dict(),
        "fnm_com_microfones_diferentes": int(vgen[vgen.score < vthr].mic_diferente.sum()),
        "score_genuino_mesmo_mic": float(vgen[~vgen.mic_diferente].score.mean()),
        "score_genuino_mic_diferente": float(vgen[vgen.mic_diferente].score.mean()),
    }


def plot_robustness() -> None:
    df = pd.read_csv(ROBUSTNESS)
    base = df[df.condition == "original"].eer_pct.iloc[0]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for noise, marker in (("white", "o"), ("babble", "s")):
        d = df[df.condition.str.startswith(noise)].copy()
        d["snr"] = d.condition.str.extract(r"@(\d+)dB").astype(int)
        d = d.sort_values("snr", ascending=False)
        x = ["limpo"] + [f"{s} dB" for s in d.snr]
        ax.plot(x, [base] + d.eer_pct.tolist(), marker=marker, label=noise)
    ax.set_ylabel("EER (%)")
    ax.set_xlabel("condição")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "robustness.png")
    plt.close(fig)


def plot_pipeline_example(path: str = PIPELINE_SAMPLE) -> None:
    import torch
    import yaml

    from preprocess import AudioPreprocessor

    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    pre = AudioPreprocessor(config=config.get("preprocessing", {}))
    raw = pre._load_and_resample(Path(path))
    voiced = pre._apply_vad(raw)
    spec = pre.process(path)
    to_np = lambda t: t.squeeze().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)  # noqa: E731

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 3.6))
    y = to_np(raw)
    axes[0].plot(np.arange(len(y)) / 16000, y, lw=0.3, color="#555")
    axes[0].set_title(f"(a) sinal mono 16 kHz · {Path(path).stem} · {len(y) / 16000:.1f} s", fontsize=8)
    v = to_np(voiced)
    axes[1].plot(np.arange(len(v)) / 16000, v, lw=0.3, color="#2c7fb8")
    axes[1].set_title(f"(b) após o VAD · {len(v) / 16000:.1f} s de fala", fontsize=8)
    axes[2].imshow(to_np(spec), aspect="auto", origin="lower", cmap="magma",
                   extent=[0, len(v) / 16000, 0, 80])
    axes[2].set_title("(c) log-Mel 80 bandas com CMN (entrada da ECAPA-TDNN)", fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=6)
    axes[2].set_xlabel("tempo (s)", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "pipeline_example.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_pipeline_example()
    stats: dict = {}
    cases = error_cases()
    cases.to_csv(OUT / "error_cases.csv", index=False)
    plot_errors(cases)
    speaker_6209(stats)
    aggregate_stats(stats)
    plot_robustness()
    (OUT / "error_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Figuras e tabelas salvas em {OUT}/")


if __name__ == "__main__":
    main()
