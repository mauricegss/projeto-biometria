"""
Importa os pares de teste do LibriSpeech definidos pela Dupla 8 (Experimento B).

Pelo enunciado (§11, regra 8), as duas CNNs de uma modalidade são avaliadas
nos mesmos pares de cada base. O LibriSpeech é a base principal da Dupla 8,
então os pares dela valem para o nosso modelo nessa base.

O script baixa os CSVs de um commit fixo do repositório da Dupla 8, confere o
SHA-256 dos arquivos originais e grava em ``protocols/dupla8/librispeech/``:

- ``test_genuine_pairs.csv`` / ``test_impostor_pairs.csv`` no formato deste
  repositório (caminhos com o prefixo do ``--data_dir`` e coluna ``tipo``);
- ``sha256.csv`` com os hashes dos arquivos originais e o commit de origem.

Uso:
    python import_dupla8_protocols.py
"""

import argparse
import csv
import hashlib
import io
import sys
import urllib.request
from pathlib import Path

REPO = "SandWoodJones/Biometria"
# "add speaker-disjoint splits and verification pairs" (20/11/2026)
COMMIT = "6e1f8170e960df78bfe53717810d73c960ca7e53"
SOURCE_DIR = "protocols/librispeech"
FILES = {
    "test_genuine_pairs.csv": "genuine",
    "test_impostor_pairs.csv": "impostor",
}
EXPECTED_SHA256 = {
    "test_genuine_pairs.csv": "b16477d07b47f78ca75eaa163968ffbc2706cf2bb6552ae193b6aad05f4056ee",
    "test_impostor_pairs.csv": "b438d3b1b0dbf0d38cffb8f533af19726598e77a4d0b76f94bb6b00c0f4fdb2d",
}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data_dir", default="data/librispeech/train-clean-100",
                        help="Raiz do LibriSpeech, usada como prefixo dos caminhos.")
    parser.add_argument("--output_dir", default="protocols/dupla8/librispeech")
    parser.add_argument("--commit", default=COMMIT, help="Commit do repositório da Dupla 8.")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = Path(args.data_dir)
    hashes = []

    for name, tipo in FILES.items():
        url = f"https://raw.githubusercontent.com/{REPO}/{args.commit}/{SOURCE_DIR}/{name}"
        raw = fetch(url)
        digest = hashlib.sha256(raw).hexdigest()
        expected = EXPECTED_SHA256.get(name)
        if expected and digest != expected:
            sys.exit(f"SHA-256 diferente do esperado para {name}: {digest}")
        hashes.append((name, digest))

        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
        if not rows or {"amostra_1", "amostra_2"} - rows[0].keys():
            sys.exit(f"{name}: colunas amostra_1/amostra_2 não encontradas")

        with open(out / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["amostra_1", "amostra_2", "tipo"])
            for r in rows:
                writer.writerow([
                    (prefix / r["amostra_1"]).as_posix(),
                    (prefix / r["amostra_2"]).as_posix(),
                    tipo,
                ])
        print(f"{name}: {len(rows)} pares · sha256 {digest}")

    with open(out / "sha256.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["arquivo", "sha256", "origem"])
        for name, digest in hashes:
            writer.writerow([name, digest, f"{REPO}@{args.commit}/{SOURCE_DIR}/{name}"])


if __name__ == "__main__":
    main()
