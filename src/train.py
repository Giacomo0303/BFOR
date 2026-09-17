import os

import torch
from tqdm import tqdm


def train_epoch(model, dataloader, loss_fn, optimizer, scaler, device, epoch=None):
    model.train()
    running_loss = 0.0

    desc = f"Epoch {epoch:03d} [Train]" if epoch is not None else "Training"
    pbar = tqdm(dataloader, desc=desc, leave=False)

    for batch_idx, (x, y) in enumerate(pbar):
        optimizer.zero_grad()

        x = x.to(device, non_blocking=True)
        y = [b.to(device, non_blocking=True) for b in y]

        device_type = "cuda" if "cuda" in str(device) else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
            out = model(x)

        batch_loss = loss_fn(out, y)

        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_val = batch_loss.item()
        running_loss += loss_val
        current_avg = running_loss / (batch_idx + 1)

        pbar.set_postfix(
            {
                "loss": f"{loss_val:.4f}",
                "avg": f"{current_avg:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
        )

    return running_loss / len(dataloader)


def validate(model, dataloader, loss_fn, device, epoch=None):
    model.eval()
    running_loss = 0.0

    desc = f"Epoch {epoch:03d} [Val]  " if epoch is not None else "Validating"
    pbar = tqdm(dataloader, desc=desc, leave=False)
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = [b.to(device, non_blocking=True) for b in y]

            with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                out = model(x)

            batch_loss = loss_fn(out, y)

            loss_val = batch_loss.item()
            running_loss += loss_val
            current_avg = running_loss / (batch_idx + 1)

            pbar.set_postfix(
                {"val_loss": f"{loss_val:.4f}", "avg": f"{current_avg:.4f}"}
            )

    return running_loss / len(dataloader)


def run_training(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    lr_scheduler,
    scaler,
    early_stopping,
    epochs,
    device,
    start_epoch=0,
):
    print(f"\n{'=' * 65}")
    print(f"Starting B-FOR training for {epochs} epochs on device: {device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"{'=' * 65}\n")

    for epoch in range(start_epoch, epochs):
        current_lr = optimizer.param_groups[0]["lr"]

        # 1. Train one epoch with progress bar
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch + 1,
        )

        # 2. Validation with progress bar
        val_loss = validate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch + 1,
        )

        # 3. Learning rate scheduler step
        lr_scheduler.step(val_loss)

        # 4. Summary print
        print(
            f"Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # 5. Check early stopping & save checkpoint
        early_stopping(
            validation_loss=val_loss,
            current_epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            scaler=scaler,
        )

        if early_stopping.early_stop:
            print(f"\nTraining early-stopped at epoch {epoch + 1}!")
            break

    print(f"\n{'=' * 65}")
    print("Training finished!")
    early_stopping.load_best_weights(model)
    print(f"{'=' * 65}\n")
    return model


def resume_training(path, model, optimizer=None, scheduler=None, scaler=None):
    """
    Resumes training from a saved checkpoint.
    Loads the model, optimizer, scheduler, and scaler states,
    and returns (start_epoch, best_val_loss).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    checkpoint = torch.load(path, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_val_loss = checkpoint.get("val_loss", torch.inf)

    print(f"--> Successfully resumed from {path}")
    print(f"    Resuming at epoch {start_epoch} (best val loss: {best_val_loss:.4f})")

    return start_epoch, best_val_loss


def load_model(model, path):
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


class EarlyStopping:
    def __init__(self, path="best_model.pt", patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.min_validation_loss = torch.inf
        self.path = path

        dirname = os.path.dirname(self.path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    def __call__(
        self,
        validation_loss,
        current_epoch,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
    ):
        if validation_loss + self.min_delta >= self.min_validation_loss:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered at epoch {current_epoch}!")
        else:
            prev_loss = self.min_validation_loss
            self.min_validation_loss = validation_loss
            self.counter = 0
            print(
                f"--> Val loss improved ({prev_loss:.4f} --> {validation_loss:.4f}). "
                f"Saving checkpoint to {self.path}..."
            )
            checkpoint = {
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": validation_loss,
            }
            if optimizer is not None:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            if scaler is not None:
                checkpoint["scaler_state_dict"] = scaler.state_dict()

            torch.save(checkpoint, self.path)

    def load_best_weights(self, model):
        checkpoint = torch.load(self.path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model weights from {self.path}")
        return model
