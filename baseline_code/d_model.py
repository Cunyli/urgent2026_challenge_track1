from typing import Any
import pytorch_lightning as L
from torch.optim.optimizer import Optimizer
# from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torchaudio
from baseline_code.models import *
from baseline_code.config import Config 
from espnet2.enh.loss.criterions.time_domain import SISNRLoss, MultiResL1SpecLoss

try:
    from pesq import PesqError, pesq
except ImportError:
    PesqError = None
    pesq = None
PESQ_EXCEPTIONS = (ValueError, RuntimeError) if PesqError is None else (PesqError, ValueError, RuntimeError)


def metric_prefix(stage):
    return "Training" if stage == "train" else "Validation"


class SEModel(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()

        self.save_hyperparameters()
        self.cfg = cfg
        
        if self.cfg.se_model == "bsrnn":
            self.se_model = BSRNN_SE(** self.cfg.model_configs)
        else:
            self.se_model = None
            raise TypeError
        self.mr_l1_loss = MultiResL1SpecLoss(window_sz=[256, 512, 768, 1024], eps = 1.0e-6,normalize_variance=True, time_domain_weight=0.5)
        self.sisnr_loss = SISNRLoss()
        self._warned_missing_pesq = False




    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        return

    def on_after_backward(self):
        return
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):


        all_norm = 0.
        all_size = 1e-5
        for n, p in self.named_parameters():
            if p.grad is not None and  not torch.isnan(p.grad).any():
                all_norm += p.grad.norm().item() * p.view(-1).size()[0]
                all_size += p.view(-1).size()[0]
        self.log("Training/Grad Norm", all_norm / all_size, on_step=False, on_epoch=True)


        # check if has grad of NaN
        grad_has_nan = any(
            torch.isnan(p.grad).any() 
            for p in self.parameters() 
            if p.grad is not None
        )
        if grad_has_nan:
            rank = torch.distributed.get_rank()
            print(f'RANK {rank}: NaN in grad has been decected, reset grad to zero')
            optimizer.zero_grad()
            
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
    
    def _compute_valid_pesq(self, clean_speech, se_speech, fs, speech_length, batch_idx):
        max_batches = int(getattr(self.cfg, "valid_pesq_batches", 0) or 0)
        if max_batches <= 0 or batch_idx >= max_batches or self.trainer.sanity_checking:
            return None
        if pesq is None:
            if not self._warned_missing_pesq:
                self.print("Skipping Validation/PESQ Score because the optional 'pesq' package is not installed.")
                self._warned_missing_pesq = True
            return None

        sample_rate = int(fs.detach().cpu().item()) if torch.is_tensor(fs) else int(fs)
        if sample_rate not in (8000, 16000):
            return None
        mode = "wb" if sample_rate == 16000 else "nb"

        scores = []
        for sample_idx in range(clean_speech.shape[0]):
            sample_length = min(
                int(speech_length[sample_idx].detach().cpu().item()),
                clean_speech.shape[-1],
                se_speech.shape[-1],
            )
            ref = clean_speech[sample_idx, :sample_length].detach().cpu().numpy()
            deg = se_speech[sample_idx, :sample_length].detach().cpu().numpy()
            try:
                scores.append(pesq(sample_rate, ref, deg, mode))
            except PESQ_EXCEPTIONS as exc:
                self.print(f"Skipping Validation/PESQ Score sample {sample_idx} in batch {batch_idx}: {exc}")

        if not scores:
            return None
        return torch.tensor(sum(scores) / len(scores), device=clean_speech.device)

    def _multi_res_l1_components(self, target, estimate):
        assert target.shape == estimate.shape, (target.shape, estimate.shape)
        half_precision = (torch.float16, torch.bfloat16)
        if target.dtype in half_precision or estimate.dtype in half_precision:
            target = target.float()
            estimate = estimate.float()
        if self.mr_l1_loss.normalize_variance:
            target = target / torch.std(target, dim=1, keepdim=True)
            estimate = estimate / torch.std(estimate, dim=1, keepdim=True)

        scaling_factor = torch.sum(estimate * target, -1, keepdim=True) / (
            torch.sum(estimate**2, -1, keepdim=True) + self.mr_l1_loss.eps
        )
        scaled_estimate = estimate * scaling_factor

        if self.mr_l1_loss.reduction == "sum":
            time_domain_loss = torch.sum((scaled_estimate - target).abs(), dim=-1)
        else:
            time_domain_loss = torch.mean((scaled_estimate - target).abs(), dim=-1)

        if len(self.mr_l1_loss.stft_encoders) == 0:
            magnitude_loss = torch.zeros_like(time_domain_loss)
        else:
            magnitude_loss = torch.zeros_like(time_domain_loss)
            for stft_enc in self.mr_l1_loss.stft_encoders:
                target_mag = self.mr_l1_loss.get_magnitude(stft_enc(target)[0])
                estimate_mag = self.mr_l1_loss.get_magnitude(stft_enc(scaled_estimate)[0])
                if self.mr_l1_loss.reduction == "sum":
                    current_loss = torch.sum((estimate_mag - target_mag).abs(), dim=(1, 2))
                else:
                    current_loss = torch.mean((estimate_mag - target_mag).abs(), dim=(1, 2))
                magnitude_loss += current_loss
            magnitude_loss = magnitude_loss / len(self.mr_l1_loss.stft_encoders)

        total_loss = (
            self.mr_l1_loss.time_domain_weight * time_domain_loss
            + (1 - self.mr_l1_loss.time_domain_weight) * magnitude_loss
        )
        return total_loss, time_domain_loss, magnitude_loss

    def forward_step(self, batch, stage='train', batch_idx=0):

        clean_speech, noisy_speech, fs, speech_length = batch
        batch_size = len(clean_speech)
        B, C, T = clean_speech.shape
        assert C == 1
        clean_speech = clean_speech.view(B, T).float()
        noisy_speech = noisy_speech.view(B, T).float()


        se_speech = self.se_model(noisy_speech, speech_length, fs)[0]


        loss_per_sample, time_domain_loss, magnitude_loss = self._multi_res_l1_components(clean_speech, se_speech)
        loss = loss_per_sample.mean()
        if torch.isnan(loss):
            print('NaN in loss has been decected, skip')
            return se_speech.mean() * 0  # Skip current step

        with torch.no_grad():
            sisnr_loss = self.sisnr_loss(clean_speech, se_speech).mean()

        prefix = metric_prefix(stage)
        self.log(
            f"{prefix}/Loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=stage == "val",
        )
        self.log(
            f"{prefix}/Time Domain L1 Loss",
            time_domain_loss.detach().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=stage == "val",
        )
        self.log(
            f"{prefix}/MR STFT Magnitude Loss",
            magnitude_loss.detach().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=stage == "val",
        )
        self.log(
            f"{prefix}/SI-SNR",
            -sisnr_loss.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=stage == "val",
        )
        if stage == "val":
            self.log(
                "val_loss",
                loss.detach(),
                on_step=False,
                on_epoch=True,
                logger=False,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )

        if stage == "val":
            valid_pesq = self._compute_valid_pesq(clean_speech, se_speech, fs, speech_length, batch_idx)
            if valid_pesq is not None:
                self.log(
                    "Validation/PESQ Score",
                    valid_pesq,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size,
                    sync_dist=True,
                )
        
        return loss

    def training_step(self, batch):

        loss = self.forward_step(batch)

        return loss

    def validation_step(self, batch, batch_idx=0):
        loss = self.forward_step(batch, stage='val', batch_idx=batch_idx)

        return {'loss': loss.detach()}

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            eps=self.cfg.adam_epsilon,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.cfg.lr_step_size, gamma=self.cfg.lr_gamma)

        return [optimizer], [scheduler]
