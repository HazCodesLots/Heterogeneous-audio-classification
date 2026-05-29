import torch
import torch.nn as nn
import torchaudio.functional as F_audio

try:
    from transformers import ClapAudioModel, ClapFeatureExtractor
except ImportError as e:
    raise ImportError("pip install transformers") from e

class CLAPClassifierHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=512, num_classes=23, dropout=0.2):
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

class CLAPBST(nn.Module):
    """
    LAION CLAP audio encoder + BST classifier head.
    
    Input:
        input_features: (B, 1, T, 64) Mel-spectrograms from ClapFeatureExtractor,
                        or (B, T) raw 32kHz waveforms (dynamically processed).
    """
    def __init__(
        self,
        num_classes=23,
        head_hidden_dim=512,
        dropout=0.2,
        freeze_backbone=True,
    ):
        super().__init__()
        
        # Load the HuggingFace pre-trained CLAP Audio model (HTSAT unfused)
        self.backbone = ClapAudioModel.from_pretrained("laion/clap-htsat-unfused")
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # The HTSAT audio encoder outputs a 768-dim pooled representation
        self.head = CLAPClassifierHead(
            in_dim=768, 
            hidden_dim=head_hidden_dim, 
            num_classes=num_classes, 
            dropout=dropout
        )

        import torchaudio.transforms as T
        # Native GPU MelSpectrogram precisely matching LAION-CLAP specs
        self.mel_extractor = T.MelSpectrogram(
            sample_rate=48000,
            n_fft=1024,
            hop_length=480,
            f_min=50,
            f_max=14000,
            n_mels=64,
            norm="slaney",
            mel_scale="slaney"
        )
        self.db_transform = T.AmplitudeToDB()
        
    def forward(self, inputs, return_embedding=False):
        # inputs can be (B, T) raw waveforms at 32kHz. 
        if inputs.ndim == 2:
            # Resample 32kHz -> 48kHz (runs entirely on GPU)
            inputs_48k = F_audio.resample(inputs, orig_freq=32000, new_freq=48000)
            
            # Extract features entirely on GPU
            # We MUST disable FP16 Mixed Precision here because squaring (power=2.0) 
            # the audio frequencies will overflow 16-bit floats and cause NaN losses!
            with torch.autocast(device_type=inputs.device.type, enabled=False):
                inputs_48k = inputs_48k.to(torch.float32)
                mel = self.mel_extractor(inputs_48k)  # (B, 64, 1001)
                mel = self.db_transform(mel)          # Convert to decibels
                
            # HuggingFace CLAP strictly requires exactly 1000 frames max
            if mel.shape[2] > 1000:
                mel = mel[:, :, :1000]
            
            # HuggingFace expects (B, 1, 1000, 64), so we transpose and add channel dim
            # Cast back to the original dtype (FP16 or FP32)
            inputs = mel.transpose(1, 2).unsqueeze(1).to(inputs.dtype)
            
        # The backbone expects `input_features`
        outputs = self.backbone(input_features=inputs)
        
        # pooler_output is the global pooled 768-dim embedding
        emb = outputs.pooler_output
        
        logits = self.head(emb)
        
        if return_embedding:
            return {"logits": logits, "embedding": emb}
        return logits

def build_model(num_classes=23, freeze_backbone=True, **kwargs):
    return CLAPBST(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        **kwargs
    )
