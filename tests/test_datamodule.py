from __future__ import annotations

import torch

from jepa_sudoku.data.datamodule import (
    SudokuDataConfig,
    SudokuDataModule,
    SudokuPuzzleDataset,
    load_precomputed_solutions,
    select_solution_subset,
)


def test_dataset_returns_sparse_puzzle_targets_and_queries() -> None:
    dataset = SudokuPuzzleDataset(
        num_samples=2,
        num_cells_to_mask=10,
        seed=123,
        randomize_mask_per_access=False,
    )
    puzzle, target, queries = dataset[0]

    assert puzzle.shape == (71, 3)
    assert target.shape == (10, 3)
    assert queries.shape == (10, 2)

    assert torch.all(puzzle[:, 2] >= 1)
    assert torch.all(target[:, 2] >= 1)
    assert torch.equal(target[:, :2], queries)


def test_datamodule_batches_sparse_contract() -> None:
    module = SudokuDataModule(
        SudokuDataConfig(
            num_samples=8,
            num_cells_to_mask=6,
            seed=7,
            batch_size=4,
            num_workers=0,
            shuffle=False,
        )
    )
    batch_puzzle, batch_target, batch_queries = next(iter(module.train_dataloader()))

    assert batch_puzzle.shape == (4, 75, 3)
    assert batch_target.shape == (4, 6, 3)
    assert batch_queries.shape == (4, 6, 2)
    assert torch.equal(batch_target[..., :2], batch_queries)


def test_precomputed_dataset_respects_num_samples_limit(tmp_path) -> None:
    solution_boards = torch.randint(1, 10, (10, 81), dtype=torch.uint8)
    dataset_path = tmp_path / "dataset.pt"
    torch.save({"solution_boards": solution_boards}, dataset_path)

    loaded = load_precomputed_solutions(str(dataset_path))
    subset = select_solution_subset(loaded, 4)
    dataset = SudokuPuzzleDataset(
        num_samples=4,
        num_cells_to_mask=2,
        solution_boards=subset,
        seed=123,
        randomize_mask_per_access=False,
    )

    assert len(dataset) == 4
    assert torch.equal(dataset._solution_boards, loaded[:4].float())
