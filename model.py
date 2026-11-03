"""
================================================================================
model.py — ECAPA-TDNN para Speaker Verification
================================================================================
Projeto    : Verificação de Locutor (Speaker Verification) — Dupla 7
Disciplina : Biometria — UTFPR
Arquitetura: ECAPA-TDNN (Emphasized Channel Attention, Propagation and
             Aggregation in TDNN Based Speaker Verification)
Referência : Desplanques, B., Thienpondt, J., & Demuynck, K. (2020).
             ECAPA-TDNN: Emphasized Channel Attention, Propagation and
             Aggregation in TDNN Based Speaker Verification.
             Interspeech 2020. https://arxiv.org/abs/2005.07143

Implementação : speechbrain.lobes.models.ECAPA_TDNN (SpeechBrain >= 1.0)
Origem dos pesos pré-treinados : speechbrain/spkrec-ecapa-voxceleb (HuggingFace Hub)
Dataset de pré-treinamento     : VoxCeleb1 + VoxCeleb2
Função de perda                : AAM-Softmax (Additive Angular Margin Softmax)
Dimensão do embedding          : 192 (padrão)
Número de parâmetros           : 20.767.552 (backbone, medido em runtime)

Fluxo de dados:
    entrada (B, 80, T)  →  backbone SpeechBrain (B, T, 80)  →  embedding (B, 192)
    embeddings são L2-normalizados no forward.
================================================================================
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------

PRETRAINED_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

# Canais compatíveis com o checkpoint pré-treinado do Hub
# (blocks.0.conv: 80→1024, mfa: 3072→3072).
DEFAULT_CHANNELS: list[int] = [1024, 1024, 1024, 1024, 3072]

# Dimensão do embedding do checkpoint pré-treinado
PRETRAINED_EMBEDDING_DIM = 192


# ==============================================================================
# 1. AAMSoftmax — Additive Angular Margin Softmax
# ==============================================================================

class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax (AAM-Softmax / ArcFace).

    Função de perda que adiciona uma margem angular aditiva para melhorar
    a discriminabilidade dos embeddings no espaço hiperesférico.

    Parâmetros
    ----------
    embedding_dim : int
        Dimensão dos embeddings de entrada.
    n_classes : int
        Número de classes (locutores) no conjunto de treinamento.
    margin : float, opcional
        Margem angular aditiva em radianos. Padrão: 0.2.
    scale : float, opcional
        Fator de escala (temperatura inversa). Padrão: 30.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_classes: int,
        margin: float = 0.2,
        scale: float = 30.0,
    ):
        super(AAMSoftmax, self).__init__()
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.FloatTensor(n_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)   # limiar para estabilidade numérica
        self.mm = math.sin(math.pi - margin) * margin

        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula a perda AAM-Softmax.

        Parâmetros
        ----------
        embeddings : torch.Tensor
            Embeddings L2-normalizados com shape (batch, embedding_dim).
        labels : torch.Tensor
            Rótulos de classe com shape (batch,).

        Retorna
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(loss, logits)`` — escalar da perda e logits (batch, n_classes)
            já escalados (usados para calcular acurácia no treinamento).
        """
        # Normaliza os pesos do classificador
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Produto interno no espaço normalizado = cosseno do ângulo
        cosine = F.linear(embeddings, weight_norm)          # (batch, n_classes)
        sine = torch.sqrt(1.0 - cosine.clamp(-1.0, 1.0) ** 2)

        # cos(θ + m) = cos θ · cos m − sin θ · sin m
        phi = cosine * self.cos_m - sine * self.sin_m

        # Garante monotonicidade quando θ + m > π
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Aplica margem apenas à classe correta
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale

        loss = self.criterion(logits, labels)
        return loss, logits


# ==============================================================================
# 2. ECAPATDNN — Modelo Principal (wrapper do backbone SpeechBrain)
# ==============================================================================

class ECAPATDNN(nn.Module):
    """
    ECAPA-TDNN para extração de embeddings de locutor.

    Utiliza a implementação oficial ``speechbrain.lobes.models.ECAPA_TDNN``
    como backbone, carregando os pesos pré-treinados em VoxCeleb1+VoxCeleb2
    (ver :func:`load_pretrained_ecapa`).

    A interface segue o pipeline do projeto: entrada é um Log-Mel Spectrogram
    de shape ``(batch, n_mels, time)`` e a saída é um embedding
    L2-normalizado de shape ``(batch, embedding_dim)``.

    Parâmetros
    ----------
    input_size : int, opcional
        Número de bandas Mel de entrada. Padrão: 80.
    embedding_dim : int, opcional
        Dimensão do vetor de embedding. Padrão: 192.
    channels : list[int] ou None, opcional
        Lista de canais do backbone ECAPA. Padrão: compatível com o
        checkpoint pré-treinado (``[1024, 1024, 1024, 1024, 3072]``).
    n_classes : int ou None, opcional
        Se fornecido, instancia a cabeça AAM-Softmax interna (treino).
    """

    def __init__(
        self,
        input_size: int = 80,
        embedding_dim: int = PRETRAINED_EMBEDDING_DIM,
        channels: Optional[list[int]] = None,
        n_classes: Optional[int] = None,
    ):
        super(ECAPATDNN, self).__init__()

        from speechbrain.lobes.models.ECAPA_TDNN import (  # noqa: PLC0415
            ECAPA_TDNN as _SB_ECAPA_TDNN,
        )

        self.embedding_dim = embedding_dim
        self.input_size = input_size

        self.backbone = _SB_ECAPA_TDNN(
            input_size=input_size,
            lin_neurons=embedding_dim,
            channels=list(channels or DEFAULT_CHANNELS),
        )

        self.classifier: Optional[AAMSoftmax] = (
            AAMSoftmax(embedding_dim, n_classes) if n_classes is not None else None
        )

    def forward(
        self, x: torch.Tensor, lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extrai o embedding L2-normalizado.

        Parâmetros
        ----------
        x : torch.Tensor
            Log-Mel Spectrogram com shape (batch, n_mels, time).
        lens : torch.Tensor, opcional
            Comprimentos relativos (0–1) de cada utterance no batch,
            usados para mascarar padding no pooling. Padrão: sem máscara.

        Retorna
        -------
        torch.Tensor
            Embedding L2-normalizado com shape (batch, embedding_dim).
        """
        # Backbone espera (batch, time, n_mels)
        emb = self.backbone(x.transpose(1, 2), lengths=lens)  # (B, 1, D)
        emb = emb.squeeze(1)                                   # (B, D)
        return F.normalize(emb, p=2, dim=1)

    def forward_with_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Propagação direta com cálculo da perda AAM-Softmax.

        Requer que ``n_classes`` tenha sido especificado na instanciação.

        Retorna
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(embedding, loss, logits)``.
        """
        if self.classifier is None:
            raise RuntimeError(
                "Modelo instanciado sem `n_classes`. "
                "Forneça `n_classes` no construtor para usar `forward_with_loss`."
            )
        embed = self.forward(x, lens=lens)
        loss, logits = self.classifier(embed, labels)
        return embed, loss, logits


# ==============================================================================
# 3. load_pretrained_ecapa — Carregamento de pesos pré-treinados
# ==============================================================================

def _infer_channels(state_dict: dict) -> list[int]:
    """Deriva a lista de canais do backbone a partir do state_dict do Hub."""
    c0 = int(state_dict["blocks.0.conv.conv.weight"].shape[0])
    c1 = int(state_dict["mfa.conv.conv.weight"].shape[0])
    return [c0, c0, c0, c0, c1]


def load_pretrained_ecapa(
    embedding_dim: int = PRETRAINED_EMBEDDING_DIM,
    use_speechbrain: bool = True,
    device: str = "cpu",
) -> ECAPATDNN:
    """
    Instancia o ECAPA-TDNN e carrega os pesos pré-treinados do SpeechBrain.

    Com ``use_speechbrain=True``, baixa (ou usa o cache de) o checkpoint
    ``speechbrain/spkrec-ecapa-voxceleb`` e carrega os pesos no backbone.
    Em caso de falha (sem SpeechBrain, sem conexão), instancia o modelo com
    pesos aleatórios e emite aviso — o usuário deve estar ciente de que o
    modelo NÃO estará pré-treinado nesse caso.

    Parâmetros
    ----------
    embedding_dim : int, opcional
        Dimensão do embedding de saída. Padrão: 192 (idêntico ao checkpoint).
    use_speechbrain : bool, opcional
        Se True, tenta carregar os pesos pré-treinados. Padrão: True.
    device : str, opcional
        Dispositivo usado no carregamento inicial. Padrão: ``"cpu"``.

    Retorna
    -------
    ECAPATDNN
        Instância do modelo com pesos carregados (ou aleatórios, se falhar).
    """
    if not use_speechbrain:
        model = ECAPATDNN(input_size=80, embedding_dim=embedding_dim)
        _print_model_info(model, pretrained=False)
        return model

    try:
        from speechbrain.inference import EncoderClassifier  # noqa: PLC0415

        print(f"[INFO] Carregando pesos pré-treinados: {PRETRAINED_SOURCE} ...")
        classifier = EncoderClassifier.from_hparams(
            source=PRETRAINED_SOURCE,
            run_opts={"device": device},
        )
        pretrained_sd = classifier.mods.embedding_model.state_dict()

        channels = _infer_channels(pretrained_sd)
        model = ECAPATDNN(
            input_size=80, embedding_dim=embedding_dim, channels=channels
        )

        # Carrega apenas os tensores com chave E shape compatíveis
        model_sd = model.backbone.state_dict()
        compatible = {
            k: v
            for k, v in pretrained_sd.items()
            if k in model_sd and model_sd[k].shape == v.shape
        }
        model.backbone.load_state_dict(compatible, strict=False)

        n_loaded = len(compatible)
        n_total = len(pretrained_sd)
        if n_loaded != n_total:
            print(
                f"[AVISO] {n_total - n_loaded}/{n_total} tensores não "
                "carregados (shape incompatível — embedding_dim diferente do checkpoint?)."
            )
        else:
            print(f"[INFO] Pesos pré-treinados carregados ({n_loaded}/{n_total} tensores).")

        _print_model_info(model, pretrained=True)
        return model

    except Exception as exc:  # noqa: BLE001
        print(f"[AVISO] Falha ao carregar pesos pré-treinados: {exc}")
        print(
            "[AVISO] Usando ECAPA-TDNN com pesos ALEATÓRIOS — "
            "o modelo NÃO está pré-treinado."
        )
        model = ECAPATDNN(input_size=80, embedding_dim=embedding_dim)
        _print_model_info(model, pretrained=False)
        return model


def _print_model_info(model: nn.Module, pretrained: bool) -> None:
    """Imprime informações resumidas do modelo."""
    info = get_model_info(model)
    print(f"[INFO] Arquitetura  : {info['architecture_name']}")
    print(f"[INFO] Pré-treinado : {'sim (VoxCeleb1+2)' if pretrained else 'NÃO'}")
    print(f"[INFO] Parâmetros   : {info['n_params']:,}")
    print(f"[INFO] Tamanho est. : {info['model_size_mb']:.2f} MB")
    print(f"[INFO] Embedding dim: {info['embedding_dim']}")


# ==============================================================================
# 4. get_model_info — Informações do modelo
# ==============================================================================

def get_model_info(model: nn.Module) -> dict:
    """
    Retorna um dicionário com informações sobre o modelo.

    Retorna
    -------
    dict
        ``n_params``, ``model_size_mb``, ``embedding_dim``,
        ``architecture_name``, ``pretrained_source``.
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = n_params * 4 / (1024 ** 2)  # float32 = 4 bytes

    embedding_dim = getattr(model, "embedding_dim", None)

    return {
        "n_params": n_params,
        "model_size_mb": model_size_mb,
        "embedding_dim": embedding_dim,
        "architecture_name": model.__class__.__name__,
        "pretrained_source": PRETRAINED_SOURCE,
    }


# ==============================================================================
# Bloco principal — teste rápido
# ==============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("ECAPA-TDNN — Teste Rápido (inicialização aleatória)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    model = ECAPATDNN(
        input_size=80,
        embedding_dim=192,
        n_classes=None,
    ).to(device)
    model.eval()

    # Tensor dummy: batch=2, 80 coeficientes Mel, 300 frames (~3 s a 100 fps)
    dummy_input = torch.randn(2, 80, 300, device=device)
    print(f"\nEntrada : {list(dummy_input.shape)}  (batch, n_mels, frames)")

    with torch.no_grad():
        embeddings = model(dummy_input)

    print(f"Embedding: {list(embeddings.shape)}  (batch, embedding_dim)")

    assert embeddings.shape == (2, 192), (
        f"Shape inesperado: {embeddings.shape}. Esperado: (2, 192)."
    )
    print("Shape do embedding: OK")

    norms = embeddings.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"Embeddings não estão L2-normalizados. Normas: {norms}"
    )
    print("Normalização L2   : OK")

    # Testa forward com máscara de comprimento (lens)
    lens = torch.tensor([1.0, 0.5], device=device)
    with torch.no_grad():
        emb_masked = model(dummy_input, lens=lens)
    assert emb_masked.shape == (2, 192)
    print("Forward com lens  : OK")

    # Testa AAM-Softmax (loss, logits)
    clf = AAMSoftmax(embedding_dim=192, n_classes=10).to(device)
    labels = torch.randint(0, 10, (2,), device=device)
    loss, logits = clf(embeddings, labels)
    assert loss.dim() == 0 and logits.shape == (2, 10)
    print("AAM-Softmax       : OK (loss, logits)")

    print()
    info = get_model_info(model)
    print(f"Arquitetura   : {info['architecture_name']}")
    print(f"Parâmetros    : {info['n_params']:,}")
    print(f"Tamanho est.  : {info['model_size_mb']:.2f} MB")
    print(f"Embedding dim : {info['embedding_dim']}")
    print("=" * 60)
    print("Todos os testes passaram com sucesso!")
    sys.exit(0)
