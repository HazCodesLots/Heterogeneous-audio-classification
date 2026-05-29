import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # pyrefly: ignore [missing-import]
    from panns_inference.models import Cnn14
except ImportError as e:
    raise ImportError(
        "Install panns_inference or vendor the official PANN Cnn14 model.\n"
        "pip install panns-inference"
    ) from e


class PANNClassifierHead(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, num_classes=23, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class PANNBST(nn.Module):
    """
    PANN Cnn14 encoder + BST classifier head.

    Input:
        waveform: (B, T) float tensor, mono, 32 kHz
    Output:
        logits: (B, num_classes)
    """

    def __init__(
        self,
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        num_classes=23,
        pretrained_classes=527,
        head_hidden_dim=512,
        dropout=0.2,
        freeze_backbone=True,
    ):
        super().__init__()

        self.backbone = Cnn14(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=pretrained_classes,
        )

        self.head = PANNClassifierHead(
            in_dim=2048,
            hidden_dim=head_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def extract_embedding(self, waveform):
        """
        Returns clip embedding from PANN backbone.
        Expected waveform shape: (B, T)
        """
        out = self.backbone(waveform)
        if isinstance(out, dict):
            if "embedding" in out:
                return out["embedding"]
            if "clipwise_output" in out and out["clipwise_output"].ndim == 2:
                raise KeyError("Backbone output missing 'embedding'.")
        raise TypeError("Unexpected PANN backbone output format.")

    def forward(self, waveform, return_embedding=False):
        # STFT and LogMel inside Cnn14 are numerically unstable in float16.
        # We must disable AMP during the backbone extraction to prevent NaNs.
        with torch.amp.autocast("cuda", enabled=False):
            emb = self.extract_embedding(waveform.float())
            
        logits = self.head(emb)
        if return_embedding:
            return {"logits": logits, "embedding": emb}
        return logits


def build_model(
    num_classes=23,
    freeze_backbone=True,
    dropout=0.2,
    head_hidden_dim=512,
    **kwargs
):
    return PANNBST(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        head_hidden_dim=head_hidden_dim,
    )
