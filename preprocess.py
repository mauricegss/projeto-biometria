"""
preprocess.py
=============
Módulo de pré-processamento de áudio para o projeto de Speaker Verification.
Dupla 7 — UTFPR Biometria

Responsabilidades:
  - Leitura de áudio WAV/FLAC via soundfile (compatível com torchaudio >= 2.9,
    que delega ``torchaudio.load`` ao torchcodec)
  - Conversão para mono 16 kHz
  - Voice Activity Detection (VAD) via webrtcvad (fallback por energia)
  - Geração de Log-Mel Spectrogram (80 mel-banks, 25 ms / 10 ms)
  - Normalização por utterance (CMN — Cepstral Mean Normalization)

Uso via linha de comando:
  python preprocess.py --dataset_dir /caminho/dataset --output_dir /caminho/saida
"""

from __future__ import annotations

import argparse
import logging
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # backend sem GUI para salvar PNGs
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Tentativa de importar webrtcvad (opcional)
# ---------------------------------------------------------------------------
try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False
    logging.warning(
        "webrtcvad não está instalado. "
        "O VAD por energia (-40 dBFS) será usado como fallback."
    )

# ---------------------------------------------------------------------------
# Configuração de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess")

# ---------------------------------------------------------------------------
# Constantes / configuração padrão
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, object] = {
    "sample_rate": 16000,          # Taxa de amostragem alvo (Hz)
    "mono": True,                  # Converter para mono
    "vad_aggressiveness": 2,       # Agressividade do webrtcvad (0–3)
    "vad_enabled": True,           # Habilitar VAD
    "n_mels": 80,                  # Número de mel-banks
    "win_length_ms": 25,           # Tamanho da janela em ms
    "hop_length_ms": 10,           # Hop length em ms
    "n_fft": 512,                  # Tamanho da FFT
    "fmin": 20,                    # Frequência mínima do mel-filterbank (Hz)
    "fmax": 7600,                  # Frequência máxima do mel-filterbank (Hz)
    "normalize": True,             # Normalizar (CMN)
    "min_duration_sec": 1.0,       # Duração mínima após VAD (segundos)
}

# Formatos de áudio suportados
SUPPORTED_EXTENSIONS = {".wav", ".flac"}

# Tamanho de frame aceito pelo webrtcvad (em ms); apenas 10, 20 ou 30 ms
_VAD_FRAME_MS = 30


# ===========================================================================
# Classe principal
# ===========================================================================

class AudioPreprocessor:
    """
    Pré-processador de áudio para Speaker Verification.

    Realiza a cadeia completa:
      1. Leitura do arquivo (WAV ou FLAC)
      2. Resample para 16 kHz mono
      3. VAD para remover segmentos de silêncio
      4. Log-Mel Spectrogram
      5. Normalização por utterance (CMN)

    Parameters
    ----------
    config : dict, optional
        Dicionário de configuração. Valores ausentes são preenchidos
        com :data:`DEFAULT_CONFIG`.

    Examples
    --------
    >>> preprocessor = AudioPreprocessor()
    >>> spec = preprocessor.process("audio.wav")
    >>> print(spec.shape)  # (n_mels, T)
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        # Mescla config fornecido com os defaults
        self.config: Dict = {**DEFAULT_CONFIG, **(config or {})}

        self.sample_rate: int = int(self.config["sample_rate"])
        self.vad_enabled: bool = bool(self.config["vad_enabled"])
        self.vad_aggressiveness: int = int(self.config["vad_aggressiveness"])
        self.normalize: bool = bool(self.config["normalize"])
        self.min_duration_sec: float = float(self.config["min_duration_sec"])

        # Calcula win_length e hop_length em amostras
        self.win_length: int = int(
            self.config["win_length_ms"] * self.sample_rate / 1000
        )
        self.hop_length: int = int(
            self.config["hop_length_ms"] * self.sample_rate / 1000
        )
        self.n_fft: int = int(self.config["n_fft"])
        self.n_mels: int = int(self.config["n_mels"])
        self.fmin: float = float(self.config["fmin"])
        self.fmax: float = float(self.config["fmax"])

        # Transforma torchaudio para Log-Mel
        self._mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            power=2.0,
            normalized=False,
        )
        self._amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        logger.info(
            "AudioPreprocessor inicializado — sr=%d Hz, n_mels=%d, "
            "win=%d ms, hop=%d ms, VAD=%s (webrtcvad=%s)",
            self.sample_rate,
            self.n_mels,
            self.config["win_length_ms"],
            self.config["hop_length_ms"],
            self.vad_enabled,
            WEBRTCVAD_AVAILABLE,
        )

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def process(self, audio_path: str) -> torch.Tensor:
        """
        Processa um único arquivo de áudio e retorna o Log-Mel Spectrogram.

        Parameters
        ----------
        audio_path : str
            Caminho para o arquivo WAV ou FLAC.

        Returns
        -------
        torch.Tensor
            Tensor de shape ``(n_mels, T)`` com o Log-Mel Spectrogram
            normalizado (CMN).

        Raises
        ------
        FileNotFoundError
            Se o arquivo não for encontrado.
        ValueError
            Se o áudio for muito curto após VAD (< ``min_duration_sec`` s).
        RuntimeError
            Para qualquer outro erro de processamento.
        """
        path = Path(audio_path)

        # 1. Validação do arquivo
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Formato não suportado: '{path.suffix}'. "
                f"Suportados: {SUPPORTED_EXTENSIONS}"
            )

        # 2. Leitura e resample
        waveform = self._load_and_resample(path)

        return self.process_waveform(waveform, source=str(path))

    def process_waveform(
        self, waveform: torch.Tensor, source: str = "<memory>"
    ) -> torch.Tensor:
        """
        Processa um waveform já carregado (1, N) e retorna o Log-Mel Spectrogram.

        Usado pelo pipeline de robustez (Exp. D) para injetar áudio degradado
        em memória sem regravar arquivos.

        Parameters
        ----------
        waveform : torch.Tensor
            Waveform 1-D de shape ``(1, N)`` em float32 (amostrado em
            ``self.sample_rate``).
        source : str
            Identificador usado em mensagens de erro/log.

        Returns
        -------
        torch.Tensor
            Tensor ``(n_mels, T)`` com Log-Mel normalizado (CMN).
        """
        # 3. VAD
        if self.vad_enabled:
            waveform = self._apply_vad(waveform)

        # 4. Verificação de duração mínima
        min_samples = int(self.min_duration_sec * self.sample_rate)
        if waveform.shape[-1] < min_samples:
            raise ValueError(
                f"Áudio muito curto após VAD: "
                f"{waveform.shape[-1] / self.sample_rate:.2f}s "
                f"(mínimo={self.min_duration_sec}s) — arquivo: {source}"
            )

        # 5. Log-Mel Spectrogram
        spectrogram = self._compute_log_mel(waveform)

        # 6. Normalização CMN
        if self.normalize:
            spectrogram = self._cmn_normalize(spectrogram)

        return spectrogram  # shape: (n_mels, T)

    def process_batch(self, audio_paths: List[str]) -> List[torch.Tensor]:
        """
        Processa uma lista de arquivos de áudio.

        Arquivos que falham são registrados em log e substituídos por ``None``
        na lista de retorno (não interrompem o processamento dos demais).

        Parameters
        ----------
        audio_paths : list of str
            Lista de caminhos para arquivos de áudio.

        Returns
        -------
        list of torch.Tensor or None
            Lista de espectrogramas. Posições com erro têm valor ``None``.
        """
        results: List[Optional[torch.Tensor]] = []
        for path in tqdm(audio_paths, desc="Processando batch", unit="arquivo"):
            try:
                spec = self.process(path)
                results.append(spec)
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("Falha (FTA) em '%s': %s", path, exc)
                results.append(None)
            except Exception as exc:  # noqa: BLE001
                logger.error("Erro inesperado em '%s': %s", path, exc)
                results.append(None)
        return results

    def save_spectrogram(self, spec: torch.Tensor, output_path: str) -> None:
        """
        Salva um espectrograma Log-Mel como imagem PNG.

        A imagem é renderizada com o colormap *magma* e eixos desligados
        para exibição compacta.

        Parameters
        ----------
        spec : torch.Tensor
            Tensor de shape ``(n_mels, T)`` contendo o espectrograma.
        output_path : str
            Caminho de saída do arquivo PNG (incluindo extensão).
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        data = spec.cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
        img = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            cmap="magma",
            interpolation="nearest",
        )
        plt.colorbar(img, ax=ax, format="%+.1f dB")
        ax.set_xlabel("Frames de tempo")
        ax.set_ylabel("Mel-banks")
        ax.set_title("Log-Mel Spectrogram")
        plt.tight_layout()
        plt.savefig(str(out), bbox_inches="tight")
        plt.close(fig)
        logger.debug("Espectrograma salvo em: %s", out)

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _load_and_resample(self, path: Path) -> torch.Tensor:
        """
        Carrega áudio do disco e reamostra para ``self.sample_rate`` Hz mono.

        Parameters
        ----------
        path : Path
            Caminho do arquivo.

        Returns
        -------
        torch.Tensor
            Waveform 1-D de shape ``(1, N)`` em float32, normalizado para
            o intervalo [-1, 1].
        """
        try:
            # soundfile: (n_samples, n_channels) — evita a dependência de
            # torchcodec exigida por torchaudio >= 2.9
            data, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as exc:
            raise RuntimeError(
                f"Erro ao carregar '{path}': {exc}"
            ) from exc

        # (n_samples, n_channels) -> (n_channels, n_samples)
        waveform = torch.from_numpy(data.T.copy())

        # Converter para mono (média dos canais)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample se necessário (ex.: VCTK 48 kHz → 16 kHz)
        if orig_sr != self.sample_rate:
            logger.debug(
                "Reamostrando de %d Hz para %d Hz — '%s'",
                orig_sr, self.sample_rate, path.name,
            )
            resampler = T.Resample(
                orig_freq=orig_sr,
                new_freq=self.sample_rate,
                resampling_method="sinc_interp_hann",
            )
            waveform = resampler(waveform)

        # Normalizar amplitude para [-1, 1] (previne clipping nos filtros)
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        return waveform  # shape: (1, N)

    # ------------------------------------------------------------------
    # VAD — webrtcvad
    # ------------------------------------------------------------------

    def _apply_vad(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Remove segmentos de silêncio usando VAD.

        Utiliza ``webrtcvad`` se disponível; caso contrário, recorre ao
        VAD por energia com threshold de -40 dBFS.

        Parameters
        ----------
        waveform : torch.Tensor
            Waveform de shape ``(1, N)`` em float32.

        Returns
        -------
        torch.Tensor
            Waveform com frames de silêncio removidos, shape ``(1, M)``.
        """
        if WEBRTCVAD_AVAILABLE:
            try:
                return self._vad_webrtc(waveform)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "webrtcvad falhou (%s). Usando VAD por energia.", exc
                )
        return self._vad_energy(waveform)

    def _vad_webrtc(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        VAD via webrtcvad.

        Divide o waveform em frames de ``_VAD_FRAME_MS`` ms, classifica
        cada frame como fala ou silêncio e concatena apenas os frames de fala.

        Parameters
        ----------
        waveform : torch.Tensor
            Waveform de shape ``(1, N)``.

        Returns
        -------
        torch.Tensor
            Waveform filtrado de shape ``(1, M)``.
        """
        vad = webrtcvad.Vad(self.vad_aggressiveness)

        audio_np = waveform.squeeze(0).numpy()

        # webrtcvad requer PCM 16-bit com SR em {8000, 16000, 32000, 48000}
        frame_samples = int(self.sample_rate * _VAD_FRAME_MS / 1000)
        # Converter float32 → int16
        audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)

        speech_frames: List[np.ndarray] = []
        for start in range(0, len(audio_int16) - frame_samples + 1, frame_samples):
            frame = audio_int16[start: start + frame_samples]
            raw_bytes = struct.pack(f"{len(frame)}h", *frame)
            try:
                is_speech = vad.is_speech(raw_bytes, self.sample_rate)
            except Exception:
                # Frame inválido: inclui por segurança
                is_speech = True
            if is_speech:
                speech_frames.append(audio_np[start: start + frame_samples])

        if not speech_frames:
            logger.debug("VAD webrtcvad: nenhum frame de fala detectado; retornando original.")
            return waveform

        speech = np.concatenate(speech_frames, axis=0)
        return torch.from_numpy(speech).unsqueeze(0)

    def _vad_energy(
        self,
        waveform: torch.Tensor,
        threshold_db: float = -40.0,
    ) -> torch.Tensor:
        """
        VAD simplificado baseado em energia RMS por frame.

        Frames cuja energia (em dBFS) estiver acima de ``threshold_db``
        são considerados fala.

        Parameters
        ----------
        waveform : torch.Tensor
            Waveform de shape ``(1, N)``.
        threshold_db : float
            Threshold de energia em dBFS para separar fala de silêncio.

        Returns
        -------
        torch.Tensor
            Waveform filtrado de shape ``(1, M)``.
        """
        frame_samples = int(self.sample_rate * _VAD_FRAME_MS / 1000)
        audio = waveform.squeeze(0)

        speech_frames: List[torch.Tensor] = []
        for start in range(0, len(audio) - frame_samples + 1, frame_samples):
            frame = audio[start: start + frame_samples]
            rms = frame.pow(2).mean().sqrt()
            if rms > 0:
                energy_db = 20 * torch.log10(rms).item()
            else:
                energy_db = -120.0
            if energy_db > threshold_db:
                speech_frames.append(frame)

        if not speech_frames:
            logger.debug("VAD por energia: nenhum frame de fala detectado; retornando original.")
            return waveform

        return torch.cat(speech_frames, dim=0).unsqueeze(0)

    # ------------------------------------------------------------------
    # Espectrograma e normalização
    # ------------------------------------------------------------------

    def _compute_log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Calcula o Log-Mel Spectrogram a partir do waveform.

        Parameters
        ----------
        waveform : torch.Tensor
            Waveform de shape ``(1, N)``.

        Returns
        -------
        torch.Tensor
            Log-Mel Spectrogram de shape ``(n_mels, T)``.
        """
        # MelSpectrogram → shape (1, n_mels, T)
        mel = self._mel_transform(waveform)
        # Converter potência → dB
        log_mel = self._amplitude_to_db(mel)
        # Remover dimensão de canal → (n_mels, T)
        return log_mel.squeeze(0)

    def _cmn_normalize(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Cepstral Mean Normalization (CMN): normaliza por utterance.

        Subtrai apenas a média ao longo do eixo temporal (por canal mel).
        Equivale ao ``InputNormalization(norm_type='sentence', std_norm=False)``
        do SpeechBrain — **não** se deve dividir pelo desvio (z-score), pois
        isso colapsa os embeddings do ECAPA pré-treinado.

        Parameters
        ----------
        spec : torch.Tensor
            Tensor de shape ``(n_mels, T)``.

        Returns
        -------
        torch.Tensor
            Tensor normalizado de shape ``(n_mels, T)``.
        """
        mean = spec.mean(dim=-1, keepdim=True)
        return spec - mean


# ===========================================================================
# Função de processamento em lote de dataset
# ===========================================================================

def preprocess_dataset(
    dataset_dir: str,
    output_dir: str,
    config: Optional[Dict] = None,
) -> Tuple[int, int]:
    """
    Processa recursivamente todos os arquivos de áudio de um diretório
    e salva os espectrogramas como tensores ``.pt`` e imagens ``.png``.

    Arquivos que falham são registrados em ``<output_dir>/failed_files.txt``.

    Parameters
    ----------
    dataset_dir : str
        Diretório raiz do dataset (VCTK, LibriSpeech, etc.).
    output_dir : str
        Diretório de saída para os espectrogramas.
    config : dict, optional
        Configuração do pré-processamento. Defaults em :data:`DEFAULT_CONFIG`.

    Returns
    -------
    tuple of (int, int)
        ``(n_success, n_failed)`` — contagem de arquivos processados com
        sucesso e com falha.
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Diretório do dataset não encontrado: {dataset_dir}"
        )

    # Coletar todos os arquivos de áudio suportados
    audio_files: List[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        audio_files.extend(dataset_path.rglob(f"*{ext}"))

    if not audio_files:
        logger.warning(
            "Nenhum arquivo de áudio encontrado em '%s'.", dataset_dir
        )
        return 0, 0

    logger.info(
        "Encontrados %d arquivos de áudio em '%s'.",
        len(audio_files), dataset_dir,
    )

    preprocessor = AudioPreprocessor(config)
    failed_log_path = output_path / "failed_files.txt"

    n_success = 0
    n_failed = 0

    with open(failed_log_path, "w", encoding="utf-8") as failed_log:
        failed_log.write("# Arquivos que falharam no pré-processamento\n")
        failed_log.write(f"# Dataset: {dataset_dir}\n\n")

        for audio_file in tqdm(
            audio_files,
            desc="Pré-processando dataset",
            unit="arquivo",
        ):
            # Preservar estrutura de subdiretórios no output
            relative = audio_file.relative_to(dataset_path)
            out_dir = output_path / relative.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            stem = audio_file.stem
            tensor_out = out_dir / f"{stem}.pt"
            png_out = out_dir / f"{stem}.png"

            try:
                spec = preprocessor.process(str(audio_file))

                # Salvar tensor
                torch.save(spec, str(tensor_out))

                # Salvar imagem PNG
                preprocessor.save_spectrogram(spec, str(png_out))

                n_success += 1

            except FileNotFoundError as exc:
                msg = f"[FILE_NOT_FOUND] {audio_file} — {exc}"
                logger.error(msg)
                failed_log.write(msg + "\n")
                n_failed += 1

            except ValueError as exc:
                msg = f"[FTA] {audio_file} — {exc}"
                logger.warning(msg)
                failed_log.write(msg + "\n")
                n_failed += 1

            except Exception as exc:  # noqa: BLE001
                msg = f"[ERROR] {audio_file} — {type(exc).__name__}: {exc}"
                logger.error(msg)
                failed_log.write(msg + "\n")
                n_failed += 1

    logger.info(
        "Concluído: %d sucesso(s), %d falha(s). "
        "Log de falhas: '%s'",
        n_success, n_failed, failed_log_path,
    )
    return n_success, n_failed


# ===========================================================================
# CLI
# ===========================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    """Constrói o parser de argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        prog="preprocess",
        description=(
            "Pré-processa um dataset de áudio para Speaker Verification "
            "(UTFPR Biometria — Dupla 7)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Diretório raiz do dataset de áudio (WAV/FLAC).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Diretório de saída para os espectrogramas (.pt e .png).",
    )
    # Parâmetros configuráveis via CLI
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_CONFIG["sample_rate"])
    parser.add_argument("--n_mels", type=int, default=DEFAULT_CONFIG["n_mels"])
    parser.add_argument("--win_length_ms", type=int, default=DEFAULT_CONFIG["win_length_ms"])
    parser.add_argument("--hop_length_ms", type=int, default=DEFAULT_CONFIG["hop_length_ms"])
    parser.add_argument("--n_fft", type=int, default=DEFAULT_CONFIG["n_fft"])
    parser.add_argument("--fmin", type=float, default=DEFAULT_CONFIG["fmin"])
    parser.add_argument("--fmax", type=float, default=DEFAULT_CONFIG["fmax"])
    parser.add_argument(
        "--vad_aggressiveness",
        type=int,
        choices=[0, 1, 2, 3],
        default=DEFAULT_CONFIG["vad_aggressiveness"],
        help="Nível de agressividade do webrtcvad (0=menos, 3=mais agressivo).",
    )
    parser.add_argument(
        "--no_vad",
        action="store_true",
        help="Desabilita o VAD (processa o áudio completo).",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Desabilita a normalização CMN.",
    )
    parser.add_argument(
        "--min_duration_sec",
        type=float,
        default=DEFAULT_CONFIG["min_duration_sec"],
        help="Duração mínima do áudio após VAD (em segundos).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Habilita logging detalhado (DEBUG).",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Montar config a partir dos argumentos
    cli_config: Dict = {
        "sample_rate": args.sample_rate,
        "mono": True,
        "vad_aggressiveness": args.vad_aggressiveness,
        "vad_enabled": not args.no_vad,
        "n_mels": args.n_mels,
        "win_length_ms": args.win_length_ms,
        "hop_length_ms": args.hop_length_ms,
        "n_fft": args.n_fft,
        "fmin": args.fmin,
        "fmax": args.fmax,
        "normalize": not args.no_normalize,
        "min_duration_sec": args.min_duration_sec,
    }

    logger.info("=== Iniciando pré-processamento do dataset ===")
    logger.info("  dataset_dir : %s", args.dataset_dir)
    logger.info("  output_dir  : %s", args.output_dir)
    logger.info("  config      : %s", cli_config)

    try:
        n_ok, n_fail = preprocess_dataset(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            config=cli_config,
        )
    except FileNotFoundError as e:
        logger.critical("Diretório do dataset inválido: %s", e)
        sys.exit(1)

    logger.info(
        "=== Finalizado === Sucesso: %d | Falhas: %d",
        n_ok, n_fail,
    )
    sys.exit(0 if n_fail == 0 else 1)
