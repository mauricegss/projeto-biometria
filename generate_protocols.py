"""
generate_protocols.py — Geração de Protocolos Experimentais para Speaker Verification
Dupla 7 — UTFPR Biometria

IMPORTANTE: Os pares gerados por este script (genuínos e impostores) devem ser
compartilhados com a Dupla 8 para garantir avaliação comparável entre sistemas.
Os arquivos de protocolo ficam em protocols/{dataset}/ e devem ser entregues
junto com os resultados experimentais.

Saídas (VCTK):
  protocols/vctk/train.csv, validation.csv, test.csv
  protocols/vctk/val_genuine_pairs.csv,  val_impostor_pairs.csv
  protocols/vctk/test_genuine_pairs.csv, test_impostor_pairs.csv
Saídas (LibriSpeech — apenas teste/cross-dataset):
  protocols/librispeech/test_genuine_pairs.csv, test_impostor_pairs.csv

Datasets suportados:
  - VCTK      : estrutura p{ID}/*.wav
  - LibriSpeech: estrutura {speaker_id}/{chapter_id}/*.flac

Uso:
  python generate_protocols.py --dataset vctk --data_dir /data/VCTK-Corpus
  python generate_protocols.py --dataset librispeech --data_dir /data/LibriSpeech/test-clean
"""

import argparse
import os
import random
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Descoberta de falantes
# ---------------------------------------------------------------------------

def discover_speakers(dataset_dir: str, dataset_type: str) -> Dict[str, List[str]]:
    """
    Descobre falantes e suas utterances no diretório do dataset.

    Para VCTK: cada subdiretório com prefixo 'p' é um falante; arquivos .wav
               ou .flac dentro do subdiretório são as utterances.
    Para LibriSpeech: estrutura {speaker_id}/{chapter_id}/*.flac

    Parâmetros
    ----------
    dataset_dir : str
        Caminho raiz do dataset.
    dataset_type : str
        Tipo do dataset: 'vctk' ou 'librispeech'.

    Retorna
    -------
    Dict[str, List[str]]
        Dicionário {speaker_id: [lista de caminhos absolutos de arquivo]}.
        Falantes com menos de 5 utterances são descartados.
    """
    dataset_type = dataset_type.lower()
    speakers: Dict[str, List[str]] = {}

    if dataset_type == "vctk":
        for entry in sorted(os.listdir(dataset_dir)):
            entry_path = os.path.join(dataset_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            # Subdiretórios de falantes têm o formato p{ID}
            if not entry.startswith("p"):
                continue
            speaker_id = entry
            files = sorted(
                os.path.join(entry_path, f)
                for f in os.listdir(entry_path)
                if f.lower().endswith((".wav", ".flac"))
            )
            if len(files) >= 5:
                speakers[speaker_id] = files

    elif dataset_type == "librispeech":
        for speaker_id in sorted(os.listdir(dataset_dir)):
            speaker_path = os.path.join(dataset_dir, speaker_id)
            if not os.path.isdir(speaker_path):
                continue
            files: List[str] = []
            for chapter_id in sorted(os.listdir(speaker_path)):
                chapter_path = os.path.join(speaker_path, chapter_id)
                if not os.path.isdir(chapter_path):
                    continue
                files.extend(
                    sorted(
                        os.path.join(chapter_path, f)
                        for f in os.listdir(chapter_path)
                        if f.lower().endswith(".flac")
                    )
                )
            if len(files) >= 5:
                speakers[speaker_id] = files
    else:
        raise ValueError(
            f"dataset_type inválido: '{dataset_type}'. Use 'vctk' ou 'librispeech'."
        )

    print(f"[discover_speakers] {len(speakers)} falantes encontrados em '{dataset_dir}'.")
    return speakers


# ---------------------------------------------------------------------------
# 2. Divisão de falantes em treino / validação / teste
# ---------------------------------------------------------------------------

def split_speakers(
    speakers: Dict[str, List[str]],
    dataset_name: str,
    output_dir: str = "protocols",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Divide falantes em conjuntos de treino, validação e teste.

    A divisão é SEMPRE por falante (nunca por utterance), garantindo que não haja
    vazamento de informação entre os splits.

    Os splits são salvos em:
        {output_dir}/{dataset_name}/train.csv
        {output_dir}/{dataset_name}/validation.csv
        {output_dir}/{dataset_name}/test.csv

    Formato CSV: speaker_id, file_path

    Parâmetros
    ----------
    speakers : Dict[str, List[str]]
        Dicionário {speaker_id: [lista de arquivos]} retornado por discover_speakers.
    dataset_name : str
        Nome do dataset (usado para nomear o subdiretório de saída).
    output_dir : str
        Diretório raiz onde os protocolos serão salvos.
    train_ratio : float
        Proporção de falantes para treino (default 0.70).
    val_ratio : float
        Proporção de falantes para validação (default 0.15).
    test_ratio : float
        Proporção de falantes para teste (default 0.15).
    seed : int
        Semente para reprodutibilidade (default 42).

    Retorna
    -------
    Tuple[Dict, Dict, Dict]
        (train_speakers, val_speakers, test_speakers)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, (
        "As proporções train_ratio + val_ratio + test_ratio devem somar 1.0"
    )

    random.seed(seed)
    np.random.seed(seed)

    speaker_ids = sorted(speakers.keys())
    random.shuffle(speaker_ids)

    n = len(speaker_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # Teste recebe o restante para evitar perda por arredondamento
    n_test = n - n_train - n_val

    train_ids = speaker_ids[:n_train]
    val_ids = speaker_ids[n_train : n_train + n_val]
    test_ids = speaker_ids[n_train + n_val :]

    train_speakers = {sid: speakers[sid] for sid in train_ids}
    val_speakers = {sid: speakers[sid] for sid in val_ids}
    test_speakers = {sid: speakers[sid] for sid in test_ids}

    # Persistência dos splits
    split_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(split_dir, exist_ok=True)

    for split_name, split_dict in [
        ("train", train_speakers),
        ("validation", val_speakers),
        ("test", test_speakers),
    ]:
        rows = [
            {"speaker_id": sid, "file_path": fp}
            for sid, fps in split_dict.items()
            for fp in fps
        ]
        csv_path = os.path.join(split_dir, f"{split_name}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(
            f"[split_speakers] {split_name}: {len(split_dict)} falantes → {csv_path}"
        )

    print(
        f"[split_speakers] Total: {n_train} treino / {n_val} validação / {n_test} teste"
    )
    return train_speakers, val_speakers, test_speakers


# ---------------------------------------------------------------------------
# 3. Pares genuínos
# ---------------------------------------------------------------------------

def generate_genuine_pairs(
    speakers_dict: Dict[str, List[str]],
    n_pairs_per_speaker: int = 10,
    max_pairs: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Gera pares de utterances genuínas (mesmo falante).

    Para cada falante são gerados até `n_pairs_per_speaker` pares escolhidos
    aleatoriamente dentre todas as combinações possíveis de utterances distintas.
    O total de pares genuínos é limitado a `max_pairs`.

    Parâmetros
    ----------
    speakers_dict : Dict[str, List[str]]
        Dicionário {speaker_id: [lista de arquivos]}.
    n_pairs_per_speaker : int
        Número máximo de pares por falante (default 10).
    max_pairs : int
        Limite global de pares genuínos (default 5000).
    seed : int
        Semente para reprodutibilidade (default 42).

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas: amostra_1, amostra_2, tipo
        (tipo="genuine"; score deixado em branco — será preenchido em evaluate.py)
    """
    random.seed(seed)
    np.random.seed(seed)

    records = []
    for speaker_id in sorted(speakers_dict.keys()):
        files = speakers_dict[speaker_id]
        if len(files) < 2:
            continue

        all_combos = list(combinations(files, 2))
        n_sample = min(n_pairs_per_speaker, len(all_combos))
        sampled = random.sample(all_combos, n_sample)

        for f1, f2 in sampled:
            records.append({"amostra_1": f1, "amostra_2": f2, "tipo": "genuine"})

        if len(records) >= max_pairs:
            break

    # Aplica limite global
    if len(records) > max_pairs:
        random.seed(seed)
        records = random.sample(records, max_pairs)

    df = pd.DataFrame(records, columns=["amostra_1", "amostra_2", "tipo"])
    print(f"[generate_genuine_pairs] {len(df)} pares genuínos gerados.")
    return df


# ---------------------------------------------------------------------------
# 4. Pares impostores
# ---------------------------------------------------------------------------

def generate_impostor_pairs(
    speakers_dict: Dict[str, List[str]],
    genuine_df: pd.DataFrame = None,
    n_pairs: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Gera pares de utterances impostoras (falantes diferentes).

    Amostra aleatoriamente pares de falantes distintos e seleciona uma utterance
    de cada. Garante que nenhum par genuíno seja duplicado na lista de impostores.

    Parâmetros
    ----------
    speakers_dict : Dict[str, List[str]]
        Dicionário {speaker_id: [lista de arquivos]}.
    genuine_df : pd.DataFrame, opcional
        DataFrame de pares genuínos para evitar colisões (geralmente não ocorre,
        mas verificado por segurança).
    n_pairs : int
        Número de pares impostores a gerar (default 5000).
    seed : int
        Semente para reprodutibilidade (default 42).

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas: amostra_1, amostra_2, tipo
        (tipo="impostor"; score deixado em branco — será preenchido em evaluate.py)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Conjunto de pares genuínos para checagem de colisão
    genuine_set = set()
    if genuine_df is not None and not genuine_df.empty:
        for _, row in genuine_df.iterrows():
            genuine_set.add((row["amostra_1"], row["amostra_2"]))
            genuine_set.add((row["amostra_2"], row["amostra_1"]))

    speaker_ids = sorted(speakers_dict.keys())
    if len(speaker_ids) < 2:
        raise ValueError("São necessários pelo menos 2 falantes para gerar pares impostores.")

    records = []
    max_attempts = n_pairs * 20  # Evita loop infinito em datasets pequenos
    attempts = 0

    while len(records) < n_pairs and attempts < max_attempts:
        attempts += 1
        sid1, sid2 = random.sample(speaker_ids, 2)
        f1 = random.choice(speakers_dict[sid1])
        f2 = random.choice(speakers_dict[sid2])

        # Verificação de colisão com pares genuínos (segurança extra)
        if (f1, f2) in genuine_set or (f2, f1) in genuine_set:
            continue

        records.append({"amostra_1": f1, "amostra_2": f2, "tipo": "impostor"})

    df = pd.DataFrame(records, columns=["amostra_1", "amostra_2", "tipo"])
    print(f"[generate_impostor_pairs] {len(df)} pares impostores gerados.")
    if len(df) < n_pairs:
        print(
            f"[generate_impostor_pairs] AVISO: apenas {len(df)}/{n_pairs} pares gerados "
            f"(dataset pequeno ou muitas colisões)."
        )
    return df


# ---------------------------------------------------------------------------
# 5. Persistência do protocolo
# ---------------------------------------------------------------------------

def save_protocol(pairs_df: pd.DataFrame, output_path: str) -> None:
    """
    Salva o protocolo de pares em CSV padronizado.

    O campo `score` NÃO é incluído neste arquivo — ele será calculado e
    adicionado por evaluate.py durante a avaliação dos modelos.

    Parâmetros
    ----------
    pairs_df : pd.DataFrame
        DataFrame com colunas amostra_1, amostra_2, tipo.
    output_path : str
        Caminho completo do arquivo CSV de saída.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # Garante que apenas as colunas do protocolo sejam salvas (sem score)
    cols = [c for c in ["amostra_1", "amostra_2", "tipo"] if c in pairs_df.columns]
    pairs_df[cols].to_csv(output_path, index=False)
    print(f"[save_protocol] Protocolo salvo em: {output_path}")


# ---------------------------------------------------------------------------
# 6. Ponto de entrada principal
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Ponto de entrada CLI para geração dos protocolos experimentais.

    Para VCTK: realiza a divisão treino/validação/teste e gera pares genuínos
               e impostores para o conjunto de teste.
    Para LibriSpeech: gera apenas pares de teste (sem divisão treino/val, pois
                      é utilizado exclusivamente como conjunto de avaliação
                      cross-dataset).

    Exemplo de uso:
        python generate_protocols.py \\
            --dataset vctk \\
            --data_dir /data/VCTK-Corpus/wav48_silence_trimmed \\
            --output_dir protocols/

        python generate_protocols.py \\
            --dataset librispeech \\
            --data_dir /data/LibriSpeech/test-clean \\
            --output_dir protocols/
    """
    parser = argparse.ArgumentParser(
        description="Geração de protocolos experimentais — Speaker Verification (Dupla 7)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["vctk", "librispeech"],
        help="Dataset alvo: 'vctk' ou 'librispeech'.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Caminho raiz do dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="protocols",
        help="Diretório de saída dos protocolos (default: protocols/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente para reprodutibilidade (default: 42).",
    )
    parser.add_argument(
        "--n_genuine_per_speaker",
        type=int,
        default=10,
        help="Pares genuínos por falante (default: 10).",
    )
    parser.add_argument(
        "--n_impostor_pairs",
        type=int,
        default=5000,
        help="Total de pares impostores (default: 5000).",
    )

    args = parser.parse_args()

    # Fixa seeds globais
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(dataset_out_dir, exist_ok=True)

    print("=" * 60)
    print(f"Dataset     : {args.dataset.upper()}")
    print(f"Diretório   : {args.data_dir}")
    print(f"Saída       : {args.output_dir}")
    print(f"Seed        : {args.seed}")
    print("=" * 60)

    # ------------------------------------------------------------------ VCTK
    if args.dataset == "vctk":
        speakers = discover_speakers(args.data_dir, "vctk")

        train_spk, val_spk, test_spk = split_speakers(
            speakers=speakers,
            dataset_name="vctk",
            output_dir=args.output_dir,
            seed=args.seed,
        )

        print("\n--- Gerando pares de VALIDAÇÃO (para congelamento do threshold) ---")
        val_genuine_df = generate_genuine_pairs(
            speakers_dict=val_spk,
            n_pairs_per_speaker=args.n_genuine_per_speaker,
            seed=args.seed,
        )
        val_impostor_df = generate_impostor_pairs(
            speakers_dict=val_spk,
            genuine_df=val_genuine_df,
            n_pairs=args.n_impostor_pairs,
            seed=args.seed,
        )
        save_protocol(val_genuine_df, os.path.join(dataset_out_dir, "val_genuine_pairs.csv"))
        save_protocol(val_impostor_df, os.path.join(dataset_out_dir, "val_impostor_pairs.csv"))

        print("\n--- Gerando pares de TESTE ---")
        genuine_df = generate_genuine_pairs(
            speakers_dict=test_spk,
            n_pairs_per_speaker=args.n_genuine_per_speaker,
            seed=args.seed,
        )
        impostor_df = generate_impostor_pairs(
            speakers_dict=test_spk,
            genuine_df=genuine_df,
            n_pairs=args.n_impostor_pairs,
            seed=args.seed,
        )

        save_protocol(genuine_df, os.path.join(dataset_out_dir, "test_genuine_pairs.csv"))
        save_protocol(impostor_df, os.path.join(dataset_out_dir, "test_impostor_pairs.csv"))

        # Resumo estatístico
        print("\n" + "=" * 60)
        print("RESUMO — VCTK")
        print(f"  Falantes treino     : {len(train_spk)}")
        print(f"  Falantes validação  : {len(val_spk)}")
        print(f"  Falantes teste      : {len(test_spk)}")
        print(f"  Pares val genuínos  : {len(val_genuine_df)}")
        print(f"  Pares val impostores: {len(val_impostor_df)}")
        print(f"  Pares test genuínos : {len(genuine_df)}")
        print(f"  Pares test impostores: {len(impostor_df)}")
        print("=" * 60)

    # ----------------------------------------------------------- LibriSpeech
    elif args.dataset == "librispeech":
        # LibriSpeech é usado apenas para avaliação cross-dataset
        # Não há divisão treino/val — todos os falantes participam do protocolo
        speakers = discover_speakers(args.data_dir, "librispeech")

        print("\n--- LibriSpeech: apenas protocolo de teste (cross-dataset) ---")
        genuine_df = generate_genuine_pairs(
            speakers_dict=speakers,
            n_pairs_per_speaker=args.n_genuine_per_speaker,
            seed=args.seed,
        )
        impostor_df = generate_impostor_pairs(
            speakers_dict=speakers,
            genuine_df=genuine_df,
            n_pairs=args.n_impostor_pairs,
            seed=args.seed,
        )

        save_protocol(genuine_df, os.path.join(dataset_out_dir, "test_genuine_pairs.csv"))
        save_protocol(impostor_df, os.path.join(dataset_out_dir, "test_impostor_pairs.csv"))

        # Resumo estatístico
        print("\n" + "=" * 60)
        print("RESUMO — LibriSpeech")
        print(f"  Total de falantes   : {len(speakers)}")
        print(f"  Pares genuínos      : {len(genuine_df)}")
        print(f"  Pares impostores    : {len(impostor_df)}")
        print("=" * 60)

    print("\nProtocolos gerados com sucesso. Compartilhe os CSVs com a Dupla 8.")


if __name__ == "__main__":
    main()

