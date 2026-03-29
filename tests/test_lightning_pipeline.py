from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

import jepa_sudoku.training.experiments as experiments
from jepa_sudoku.training.experiments import run_training_experiment


def test_lightning_pretraining_pipeline_runs_and_saves_checkpoint(tmp_path, monkeypatch) -> None:
    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    dataset_path = tmp_path / "pretrain_dataset.pt"
    torch.save(
        {"solution_boards": torch.randint(1, 10, (4, 81), dtype=torch.uint8)},
        dataset_path,
    )

    monkeypatch.setattr(experiments, "_build_mlflow_logger", lambda config: False)

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="default",
            overrides=[
                "training.max_epochs=1",
                "data.num_samples=4",
                "data.num_cells_to_mask=20",
                f"data.dataset_path={dataset_path}",
                "data.batch_size=2",
                "data.num_workers=0",
                "curriculum.enabled=false",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
                "trainer.fast_dev_run=false",
                "trainer.limit_train_batches=1",
                f"checkpoint.dirpath={tmp_path}",
                "checkpoint.filename=pretrain-test",
            ],
        )

    result = run_training_experiment(config)

    assert result.history
    assert result.checkpoint_path is not None
    assert Path(result.checkpoint_path).exists()
