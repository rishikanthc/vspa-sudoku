from __future__ import annotations

import torch

from jepa_sudoku.evaluation import find_constraint_violations, load_holdout_solutions


def test_load_holdout_solutions_uses_last_n_samples(tmp_path) -> None:
    solutions = torch.arange(5 * 81, dtype=torch.uint8).reshape(5, 81)
    dataset_path = tmp_path / "dataset.pt"
    torch.save({"solution_boards": solutions}, dataset_path)

    holdout = load_holdout_solutions(str(dataset_path), 2)

    assert torch.equal(holdout, solutions[-2:])


def test_find_constraint_violations_marks_duplicate_groups() -> None:
    board = torch.tensor(
        [
            [
                1, 1, 3, 4, 5, 6, 7, 8, 9,
                4, 5, 6, 7, 8, 9, 1, 2, 3,
                7, 8, 9, 1, 2, 3, 4, 5, 6,
                2, 3, 4, 5, 6, 7, 8, 9, 1,
                5, 6, 7, 8, 9, 1, 2, 3, 4,
                8, 9, 1, 2, 3, 4, 5, 6, 7,
                3, 4, 5, 6, 7, 8, 9, 1, 2,
                6, 7, 8, 9, 1, 2, 3, 4, 5,
                9, 2, 2, 3, 4, 5, 6, 7, 8,
            ]
        ],
        dtype=torch.long,
    )

    violations = find_constraint_violations(board)

    assert violations.shape == (1, 81)
    assert violations[0, 0]
    assert violations[0, 1]
    assert violations[0, 73]
    assert violations[0, 74]
