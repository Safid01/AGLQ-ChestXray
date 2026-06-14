from pathlib import Path

import torch

from dcgnet.metrics import compute_metrics


class Trainer:
    """Training and validation loop for multi-label chest X-ray models."""

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        class_names: list[str],
        checkpoint_dir: str | Path,
        checkpoint_name: str,
        print_frequency: int = 20,
        early_stopping_patience: int | None = None,
        threshold_config: dict | None = None,
        graph_auxiliary_weight: float = 0.0,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.class_names = class_names
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_name = checkpoint_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_map = -1.0
        self.best_epoch = 0
        self.print_frequency = print_frequency
        self.early_stopping_patience = early_stopping_patience
        self.threshold_config = threshold_config or {}
        self.graph_auxiliary_weight = graph_auxiliary_weight

    @staticmethod
    def _xpu_sync() -> None:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()

    @staticmethod
    def _xpu_clear_cache() -> None:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

    @staticmethod
    def _get_logits(model_output):
        if isinstance(model_output, dict):
            return model_output["logits"]
        return model_output

    def _compute_loss(self, model_output, labels):
        logits = self._get_logits(model_output)
        loss = self.criterion(logits, labels)

        if (
            isinstance(model_output, dict)
            and self.graph_auxiliary_weight > 0
            and "graph_logits" in model_output
        ):
            graph_loss = self.criterion(model_output["graph_logits"], labels)
            loss = loss + self.graph_auxiliary_weight * graph_loss

        return loss, logits

    @staticmethod
    def _format_threshold(metrics: dict) -> str:
        per_class_threshold = metrics.get("per_class_threshold")
        if isinstance(per_class_threshold, dict) and per_class_threshold:
            values = [float(value) for value in per_class_threshold.values()]
            return f"per-class mean {sum(values) / len(values):.2f}"
        return f"{metrics.get('threshold', 0.5):.2f}"

    def train_one_epoch(self, train_loader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            model_output = self.model(images)
            loss, _ = self._compute_loss(model_output, labels)
            loss.backward()
            self.optimizer.step()

            self._xpu_sync()
            loss_value = loss.item()
            total_loss += loss_value * images.size(0)

            if batch_index == 1 or batch_index % self.print_frequency == 0 or batch_index == num_batches:
                print(
                    f"Epoch {epoch:03d} train batch {batch_index}/{num_batches} | "
                    f"loss: {loss_value:.4f}"
                )

            if batch_index % 200 == 0:
                self._xpu_clear_cache()

        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def validate(self, val_loader) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []

        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            model_output = self.model(images)
            loss, logits = self._compute_loss(model_output, labels)

            self._xpu_sync()
            total_loss += loss.item() * images.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        self._xpu_clear_cache()
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(all_logits, all_labels, self.class_names, self.threshold_config)
        metrics["loss"] = total_loss / len(val_loader.dataset)
        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict) -> None:
        checkpoint_path = self.checkpoint_dir / self.checkpoint_name
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "best_map": self.best_map,
                "best_epoch": self.best_epoch,
                "metrics": metrics,
            },
            checkpoint_path,
        )

    def fit(self, train_loader, val_loader, epochs: int) -> None:
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            val_map = val_metrics["mAP"]
            if val_map > self.best_map:
                self.best_map = val_map
                self.best_epoch = epoch
                self.save_checkpoint(epoch, val_metrics)
                checkpoint_text = "saved"
                epochs_without_improvement = 0
            else:
                checkpoint_text = "not saved"
                epochs_without_improvement += 1

            print(
                f"Epoch {epoch:03d} | "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_metrics['loss']:.4f} | "
                f"val mAP: {val_metrics['mAP']:.4f} | "
                f"val AUROC: {val_metrics['macro_auroc']:.4f} | "
                f"val F1: {val_metrics['macro_f1']:.4f} | "
                f"F1 threshold: {self._format_threshold(val_metrics)} | "
                f"best checkpoint: {checkpoint_text}"
            )

            if (
                self.early_stopping_patience is not None
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch:03d}. "
                    f"No validation mAP improvement for {self.early_stopping_patience} epochs."
                )
                break

        print(f"Best epoch: {self.best_epoch:03d}")
        print(f"Best validation mAP: {self.best_map:.4f}")
