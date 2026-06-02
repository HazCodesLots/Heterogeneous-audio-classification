import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from metrics import hierarchical_precision_recall_f, top_level_accuracy, second_level_accuracy

# Maps each of the 23 fine-grained class indices to one of 5 coarse parent indices.
# Order matches BST_CLASSES: m(0), is(1), sp(2), fx(3), ss(4)
# ['m-sp','m-si','m-m', 'is-p','is-s','is-w','is-k','is-e',
#  'sp-s','sp-c','sp-p', 'fx-o','fx-v','fx-m','fx-h','fx-a','fx-n','fx-ex','fx-el',
#  'ss-n','ss-i','ss-u','ss-s']
FINE_TO_COARSE = [0,0,0, 1,1,1,1,1, 2,2,2, 3,3,3,3,3,3,3,3, 4,4,4,4]
NUM_COARSE = 5

def mixup_batch(inputs, labels, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(inputs.size(0), device=inputs.device)
    return lam * inputs + (1 - lam) * inputs[idx], labels, labels[idx], lam

def mixup_criterion(criterion, logits, labels_a, labels_b, lam):
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)

def spec_augment(waveform, sample_rate=32000, freq_mask_param=20, time_mask_param=80, num_masks=2):
    """
    Apply SpecAugment-style time masking directly on the waveform.
    Frequency masking is applied inside the PANN backbone on the mel-spectrogram.
    This masks contiguous time segments of the raw waveform to regularize training.
    """
    T = waveform.shape[-1]
    for _ in range(num_masks):
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, max(1, T - t))
        waveform[..., t0:t0 + t] = 0.0
    return waveform

class BSTLightningModule(pl.LightningModule):
    def __init__(self, model, lr=1e-4, warmup_epochs=10, mixup_alpha=0.0,
                 label_smoothing=0.1, class_weights=None, spec_augment=False,
                 coarse_weight=0.3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.mixup_alpha = mixup_alpha
        self.use_spec_augment = spec_augment
        self.coarse_weight = coarse_weight
        self.strict_loading = False

        fine_to_coarse = torch.tensor(FINE_TO_COARSE, dtype=torch.long)
        self.register_buffer("fine_to_coarse", fine_to_coarse)

        self.save_hyperparameters(ignore=['model', 'class_weights'])
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=label_smoothing)
        # Coarse loss has no class weights (5 balanced top-level categories)
        self.coarse_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels, confidences, _ = batch

        if self.use_spec_augment and inputs.ndim == 2:
            inputs = spec_augment(inputs)

        if self.mixup_alpha > 0.0:
            inputs, labels_a, labels_b, lam = mixup_batch(inputs, labels, alpha=self.mixup_alpha)
            logits = self(inputs)
            fine_loss = mixup_criterion(self.criterion, logits, labels_a, labels_b, lam)

            coarse_a = self.fine_to_coarse[labels_a]
            coarse_b = self.fine_to_coarse[labels_b]

            coarse_logits = self._fine_to_coarse_logits(logits)
            coarse_loss = mixup_criterion(self.coarse_criterion, coarse_logits, coarse_a, coarse_b, lam)
        else:
            logits = self(inputs)
            fine_loss = self.criterion(logits, labels)
            coarse_labels = self.fine_to_coarse[labels]
            coarse_logits = self._fine_to_coarse_logits(logits)
            coarse_loss = self.coarse_criterion(coarse_logits, coarse_labels)

        loss = fine_loss + self.coarse_weight * coarse_loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=inputs.size(0))
        self.log('train_coarse_loss', coarse_loss, on_step=False, on_epoch=True, logger=True, batch_size=inputs.size(0))
        return loss

    def _fine_to_coarse_logits(self, fine_logits):
        """Aggregate 23 fine-grained logits into 5 coarse logits via scatter-sum."""
        B = fine_logits.size(0)
        coarse_logits = torch.zeros(B, NUM_COARSE, device=fine_logits.device, dtype=fine_logits.dtype)
        coarse_logits.scatter_add_(1, self.fine_to_coarse.unsqueeze(0).expand(B, -1), fine_logits)
        return coarse_logits

    def validation_step(self, batch, batch_idx):
        inputs, labels, _, _ = batch
        logits = self(inputs)
        loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=inputs.size(0))

        self.validation_step_outputs.append({
            "preds": preds.cpu(),
            "labels": labels.cpu()
        })
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs]).numpy()
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs]).numpy()

        hmetrics = hierarchical_precision_recall_f(all_labels, all_preds, lam=0.75)
        acc = second_level_accuracy(all_labels, all_preds)
        top_acc = top_level_accuracy(all_labels, all_preds)

        self.log("val_hF", hmetrics["hF"], prog_bar=True, logger=True)
        self.log("val_hP", hmetrics["hP"], logger=True)
        self.log("val_hR", hmetrics["hR"], logger=True)
        self.log("val_acc", acc, logger=True)
        self.log("val_top_acc", top_acc, logger=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_hF",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self):
        """
        Manually implement linear LR warmup for the first `warmup_epochs`.
        After warmup, ReduceLROnPlateau takes over full control.
        """
        if self.current_epoch < self.warmup_epochs:
            warmup_lr = self.lr * (self.current_epoch + 1) / self.warmup_epochs
            for pg in self.optimizers().param_groups:
                pg['lr'] = warmup_lr
