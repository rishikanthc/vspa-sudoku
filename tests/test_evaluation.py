from __future__ import annotations

import torch

from jepa_sudoku.evaluation import (
    find_constraint_violations,
    load_holdout_solutions,
    repair_conflict_clusters,
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


def test_repair_conflict_clusters_can_fix_small_connected_component() -> None:
    board = torch.tensor(
        [
            2, 4, 7, 5, 8, 6, 3, 9, 1,
            1, 6, 9, 7, 3, 4, 8, 2, 5,
            5, 8, 3, 9, 1, 2, 4, 6, 7,
            4, 7, 8, 1, 6, 5, 9, 3, 2,
            6, 2, 1, 4, 9, 3, 5, 5, 8,
            9, 3, 5, 8, 2, 7, 1, 4, 6,
            8, 5, 2, 3, 7, 9, 6, 1, 4,
            7, 9, 4, 6, 5, 1, 2, 8, 3,
            3, 1, 6, 2, 4, 8, 5, 5, 9,
        ],
        dtype=torch.long,
    )
    logits = torch.zeros((81, 9), dtype=torch.float32)
    logits[43, 6] = 5.0  # r5c8 -> 7
    logits[79, 4] = 5.0  # r9c8 -> 5
    original_clues = torch.zeros(81, dtype=torch.bool)
    violations = find_constraint_violations(board.unsqueeze(0))[0]

    repaired = repair_conflict_clusters(
        board_digits=board,
        logits=logits,
        original_clues=original_clues,
        violations=violations,
        max_component_size=6,
        max_candidates_per_cell=4,
    )

    assert int(repaired[43].item()) == 7
    assert int(repaired[79].item()) == 5
