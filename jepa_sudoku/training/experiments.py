from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from jepa_sudoku.data.datamodule import LinearMaskCurriculum, SudokuDataConfig
from jepa_sudoku.model.models import Encoder, Predictor, SudokuRepresentation, TransformerConfig

from .lightning_data import LightningSudokuDataModule
from .lightning_module import LightningTrainConfig, SudokuLightningModule

ONE_BATCH_OVERFIT_OVERRIDES: dict[str, Any] = {
    "data": {
        "num_samples": 24,
        "num_cells_to_mask": 2,
        "seed": 123,
        "unique_solution": False,
        "randomize_mask_per_access": False,
        "batch_size": 24,
        "num_workers": 0,
        "shuffle": False,
        "pin_memory": False,
        "drop_last": True,
    },
    "curriculum": {
        "enabled": False,
        "mode": "adaptive",
        "max_mask": 81,
        "num_steps": 20,
        "step": 1,
        "patience": 32,
        "min_delta": 1.0e-6,
    },
    "training": {
        "max_epochs": 2000,
        "learning_rate": 1e-4,
    },
    "trainer": {
        "accelerator": "cpu",
        "devices": 1,
        "strategy": "auto",
        "precision": "32-true",
        "deterministic": True,
        "benchmark": False,
        "log_every_n_steps": 1,
        "num_sanity_val_steps": 0,
        "fast_dev_run": False,
        "limit_train_batches": 1.0,
    },
}


@dataclass(frozen=True)
class ExperimentResult:
    history: list[float]
    is_global_zero: bool
    checkpoint_path: str | None = None


def build_data_config(data_config: DictConfig, curriculum_config: DictConfig) -> SudokuDataConfig:
    curriculum = None
    if curriculum_config.enabled and curriculum_config.mode == "linear":
        curriculum = LinearMaskCurriculum(
            start=data_config.num_cells_to_mask,
            max_mask=curriculum_config.max_mask,
            num_steps=curriculum_config.num_steps,
        )

    if curriculum_config.enabled and curriculum_config.mode not in {"linear", "adaptive"}:
        raise ValueError(
            f"Unsupported curriculum.mode={curriculum_config.mode}. Use 'linear' or 'adaptive'."
        )

    resolved = SudokuDataConfig(
        num_samples=data_config.num_samples,
        num_cells_to_mask=data_config.num_cells_to_mask,
        dataset_path=data_config.dataset_path,
        seed=data_config.seed,
        unique_solution=data_config.unique_solution,
        randomize_mask_per_access=data_config.randomize_mask_per_access,
        mask_cells_curriculum=curriculum,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        shuffle=data_config.shuffle,
        pin_memory=data_config.pin_memory,
        drop_last=data_config.drop_last,
    )
    if resolved.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    return resolved


def _build_transformer_config(model_config: DictConfig) -> TransformerConfig:
    return TransformerConfig(
        context_size=model_config.context_size,
        n_heads=model_config.n_heads,
        head_dim=model_config.head_dim,
        n_layers=model_config.n_layers,
        d_ff=model_config.d_ff,
        dropout=model_config.dropout,
    )


def build_components(config: DictConfig) -> tuple[SudokuRepresentation, Encoder, Predictor]:
    encoder_config = _build_transformer_config(config.model.encoder)
    predictor_config = _build_transformer_config(config.model.predictor)
    if encoder_config.d_model != predictor_config.d_model:
        raise ValueError(
            "Encoder and predictor must use the same d_model so they can share the representation."
        )
    if config.ssp.dim != encoder_config.d_model:
        raise ValueError(
            f"ssp.dim ({config.ssp.dim}) must match model d_model ({encoder_config.d_model})."
        )

    representation = SudokuRepresentation(
        d_model=encoder_config.d_model,
        seed=config.ssp.seed,
    )
    encoder = Encoder(encoder_config)
    predictor = Predictor(predictor_config)
    return representation, encoder, predictor


def build_train_config(config: DictConfig) -> LightningTrainConfig:
    return LightningTrainConfig(
        learning_rate=config.training.learning_rate,
        curriculum_enabled=config.curriculum.enabled,
        curriculum_mode=config.curriculum.mode,
        curriculum_step=config.curriculum.step,
        curriculum_patience=config.curriculum.patience,
        curriculum_min_delta=config.curriculum.min_delta,
        curriculum_max_mask=config.curriculum.max_mask,
    )


def build_data_module(config: DictConfig) -> LightningSudokuDataModule:
    train_config = build_data_config(config.data, config.curriculum)
    return LightningSudokuDataModule(
        train_config=train_config,
        adaptive_curriculum_enabled=(
            config.curriculum.enabled and config.curriculum.mode == "adaptive"
        ),
    )


def build_lightning_module(config: DictConfig) -> SudokuLightningModule:
    representation, encoder, predictor = build_components(config)
    return SudokuLightningModule(
        encoder=encoder,
        predictor=predictor,
        representation=representation,
        config=build_train_config(config),
    )


def _build_callbacks(config: DictConfig) -> list[pl.Callback]:
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint.dirpath,
        filename=config.checkpoint.filename,
        monitor=config.checkpoint.monitor,
        mode=config.checkpoint.mode,
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    return [checkpoint_callback]


def _build_mlflow_logger(config: DictConfig) -> MLFlowLogger:
    try:
        import mlflow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "MLflow logging is required for training. Install the 'mlflow' package first."
        ) from exc

    return MLFlowLogger(
        experiment_name=config.logging.mlflow_experiment_name,
        run_name=config.logging.mlflow_run_name,
        tracking_uri=config.logging.mlflow_tracking_uri,
        save_dir=config.logging.mlflow_save_dir,
        log_model=False,
    )


def build_trainer(config: DictConfig) -> pl.Trainer:
    return pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        strategy=config.trainer.strategy,
        precision=config.trainer.precision,
        deterministic=config.trainer.deterministic,
        benchmark=config.trainer.benchmark,
        max_epochs=config.training.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        enable_checkpointing=True,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        fast_dev_run=config.trainer.fast_dev_run,
        limit_train_batches=config.trainer.limit_train_batches,
        callbacks=_build_callbacks(config),
        logger=_build_mlflow_logger(config),
        enable_model_summary=False,
    )


def _history_from_callback_metrics(metrics: dict[str, Any]) -> list[float]:
    train = metrics.get("train_loss")
    if train is None:
        return []
    return [float(train.detach().cpu().item() if hasattr(train, "detach") else train)]


def run_training_experiment(config: DictConfig) -> ExperimentResult:
    torch.set_float32_matmul_precision(config.trainer.matmul_precision)
    if config.trainer.suppress_accumulate_grad_stream_mismatch_warning:
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    pl.seed_everything(config.seed, workers=True)

    data_module = build_data_module(config)
    model = build_lightning_module(config)
    trainer = build_trainer(config)
    trainer.fit(model=model, datamodule=data_module)

    checkpoint_path = None
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            checkpoint_path = callback.best_model_path or None
            break

    history = model.history or _history_from_callback_metrics(trainer.callback_metrics)
    return ExperimentResult(
        history=history,
        is_global_zero=trainer.is_global_zero,
        checkpoint_path=checkpoint_path,
    )


def build_one_batch_overfit_config(base_config: DictConfig) -> DictConfig:
    return OmegaConf.merge(base_config, OmegaConf.create(ONE_BATCH_OVERFIT_OVERRIDES))


def run_one_batch_overfit(base_config: DictConfig) -> tuple[DictConfig, ExperimentResult]:
    config = build_one_batch_overfit_config(base_config)
    OmegaConf.resolve(config)
    result = run_training_experiment(config)
    return config, result
