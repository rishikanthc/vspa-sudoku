from __future__ import annotations

import torch

from jepa_sudoku.evaluation import (
    find_constraint_violations,
    format_sudoku_board,
    jointly_decode_conflict_groups,
    load_holdout_solutions,
)


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


def test_format_sudoku_board_marks_cells_that_differ_from_solution() -> None:
    solved = torch.zeros((81, 3), dtype=torch.float32)
    board = torch.zeros((81, 3), dtype=torch.float32)
    xy = torch.cartesian_prod(torch.arange(1, 10), torch.arange(1, 10)).float()
    solved[:, :2] = xy
    board[:, :2] = xy
    solved[:, 2] = 1.0
    board[:, 2] = 1.0
    board[10, 2] = 2.0

    rendered = format_sudoku_board(board, solved)

    assert "[2]" in rendered
    assert "Legend: cells in [brackets] differ from the solution." in rendered


def test_jointly_decode_conflict_groups_fixes_two_cell_swap() -> None:
    board = torch.tensor(
        [
            5, 9, 6, 4, 8, 2, 7, 1, 3,
            1, 7, 8, 3, 9, 5, 6, 2, 4,
            4, 2, 3, 6, 1, 7, 5, 8, 9,
            2, 3, 7, 1, 5, 8, 9, 4, 6,
            9, 1, 4, 2, 6, 3, 8, 5, 7,
            6, 8, 5, 7, 4, 9, 1, 3, 2,
            3, 4, 9, 8, 7, 1, 2, 6, 8,
            8, 5, 2, 9, 3, 6, 4, 7, 1,
            7, 6, 1, 5, 2, 4, 3, 9, 5,
        ],
        dtype=torch.long,
    )
    logits = torch.full((81, 9), -10.0, dtype=torch.float32)
    logits[62, 4] = 4.9
    logits[62, 7] = 5.0
    logits[80, 4] = 5.0
    logits[80, 7] = 4.9
    original_clues = torch.ones(81, dtype=torch.bool)
    original_clues[62] = False
    original_clues[80] = False
    violations = find_constraint_violations(board.unsqueeze(0))[0]

    decoded = jointly_decode_conflict_groups(
        board,
        logits,
        original_clues,
        violations,
    )

    assert int(decoded[62].item()) == 5
    assert int(decoded[80].item()) == 8
