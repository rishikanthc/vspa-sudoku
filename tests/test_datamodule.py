from __future__ import annotations

import torch

from jepa_sudoku.data.datamodule import (
    SudokuDataConfig,
    SudokuDataModule,
    SudokuPuzzleDataset,
    load_precomputed_solutions,
    select_solution_subset,
)


def test_dataset_returns_full_board_target_and_mask() -> None:
    dataset = SudokuPuzzleDataset(
        num_samples=2,
        num_cells_to_mask=10,
        seed=123,
        randomize_mask_per_access=False,
    )
    puzzle, target, mask = dataset[0]

    assert puzzle.shape == (81, 3)
    assert target.shape == (81, 3)
    assert mask.shape == (81, 3)
    assert mask.dtype == torch.bool

    assert torch.equal(puzzle[:, :2], target[:, :2])
    assert torch.equal(mask[:, 0], mask[:, 1])
    assert torch.equal(mask[:, 1], mask[:, 2])
    assert int((~mask[:, 0]).sum()) == 10
    assert torch.all(puzzle[mask[:, 0], 2] >= 1)
    assert torch.all(puzzle[~mask[:, 0], 2] == 0)
    assert torch.all(target[:, 2] >= 1)


def test_datamodule_batches_full_board_contract() -> None:
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
    batch_puzzle, batch_target, batch_mask = next(iter(module.train_dataloader()))

    assert batch_puzzle.shape == (4, 81, 3)
    assert batch_target.shape == (4, 81, 3)
    assert batch_mask.shape == (4, 81, 3)
    assert int((~batch_mask[..., 0]).sum(dim=1).unique().item()) == 6


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
