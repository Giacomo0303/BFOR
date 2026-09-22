import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import run_config as cfg
from src.datasets import PascalVOC, StratifiedBatchSampler
from src.loss import BFOR_Loss
from src.model import BFOR_model
from src.train import EarlyStopping, run_training


def main():
    # 1. Datasets & Loaders
    train_set = PascalVOC(
        path=cfg.DATA_PATH, split="train", train_size=cfg.TRAIN_SIZE, seed=cfg.SEED
    )
    val_set = PascalVOC(
        path=cfg.DATA_PATH, split="val", train_size=cfg.TRAIN_SIZE, seed=cfg.SEED
    )

    train_sampler = StratifiedBatchSampler(
        dataset=train_set,
        batch_size=cfg.BATCH_SIZE,
        sml_cutoff=cfg.SCALE_CUTOFF_SML,
        lrg_cutoff=cfg.SCALE_CUTOFF_LRG,
        seed=cfg.SEED,
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_sampler=train_sampler,
        collate_fn=PascalVOC.collate_fn,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=PascalVOC.collate_fn,
        num_workers=cfg.NUM_WORKERS // 2,
        pin_memory=True,
    )

    # 2. Model, Loss, Optimizer, Scheduler, Scaler
    model = BFOR_model(n_channels=cfg.N_CHANNELS, drop_rate=cfg.DROP_RATE).to(
        cfg.DEVICE
    )
    loss = BFOR_Loss(
        alpha=cfg.ALPHA,
        lambda_ctr=cfg.LAMBDA_CTR,
        k=cfg.K,
        device=cfg.DEVICE,
        lambda_bg=cfg.LAMBDA_BG_SCALE,
        sml_cutoff=cfg.SCALE_CUTOFF_SML,
        lrg_cutoff=cfg.SCALE_CUTOFF_LRG,
    )

    optimizer = Adam(params=model.parameters(), lr=cfg.LR)
    lr_scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        factor=cfg.LR_FACTOR,
        patience=cfg.LR_PATIENCE,
        min_lr=cfg.MIN_LR_SCHEDULER,
    )
    scaler = torch.amp.GradScaler(cfg.DEVICE)

    early_stopping = EarlyStopping(
        path=cfg.SAVE_PATH, patience=cfg.EARLY_STOPPING_PATIENCE
    )

    # 3. Launch Training
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        early_stopping=early_stopping,
        epochs=cfg.EPOCHS,
        device=cfg.DEVICE,
        max_grad_norm=cfg.MAX_GRAD_NORM,
    )


if __name__ == "__main__":
    main()
