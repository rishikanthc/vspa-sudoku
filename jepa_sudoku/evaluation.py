from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from jepa_sudoku.data.datamodule import SudokuPuzzleDataset, load_precomputed_solutions
from jepa_sudoku.model.models import Encoder, SudokuRepresentation
from jepa_sudoku.training.experiments import build_components


@dataclass(frozen=True)
class DifficultyMetrics:
    empty_cells: int
    avg_cell_accuracy: float
    board_solved_rate: float
    unsolved_board_count: int
    example_failed_board: Tensor | None = None
    example_failed_confidence: Tensor | None = None


def build_difficulty_levels(config: DictConfig) -> list[int]:
    start = int(config.evaluation.difficulty.start)
    stop = int(config.evaluation.difficulty.stop)
    step = int(config.evaluation.difficulty.step)
    if step <= 0:
        raise ValueError("evaluation.difficulty.step must be positive")
    if start < 0 or stop < start or stop > 81:
        raise ValueError("evaluation difficulty range must satisfy 0 <= start <= stop <= 81")
    return list(range(start, stop + 1, step))


def load_holdout_solutions(dataset_path: str, num_samples: int) -> Tensor:
    if num_samples <= 0:
        raise ValueError("evaluation.data.num_samples must be positive")
    solutions = load_precomputed_solutions(dataset_path)
    if num_samples > solutions.shape[0]:
        raise ValueError(
            f"Requested {num_samples} holdout samples, but dataset only has {solutions.shape[0]}."
        )
    return solutions[-num_samples:].contiguous()


def _checkpoint_state_dict(checkpoint_path: str) -> dict[str, Tensor]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a Lightning state_dict.")
    return state_dict


def load_encoder_from_checkpoint(
    config: DictConfig,
) -> tuple[SudokuRepresentation, Encoder]:
    representation, encoder = build_components(config)
    state_dict = _checkpoint_state_dict(config.evaluation.checkpoint_path)

    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError("Checkpoint does not contain encoder weights.")
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected encoder state load result. missing={missing}, unexpected={unexpected}"
        )

    repr_state = {
        key.removeprefix("representation."): value
        for key, value in state_dict.items()
        if key.startswith("representation.")
    }
    missing, unexpected = representation.load_state_dict(repr_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected representation state load result. missing={missing}, unexpected={unexpected}"
        )

    encoder.eval()
    representation.eval()
    return representation, encoder


def _group_indices() -> list[Tensor]:
    rows = [torch.arange(row * 9, (row + 1) * 9, dtype=torch.long) for row in range(9)]
    cols = [torch.arange(col, 81, 9, dtype=torch.long) for col in range(9)]
    boxes: list[Tensor] = []
    for box_row in range(3):
        for box_col in range(3):
            indices: list[int] = []
            for dr in range(3):
                row_start = (box_row * 3 + dr) * 9 + (box_col * 3)
                indices.extend(range(row_start, row_start + 3))
            boxes.append(torch.tensor(indices, dtype=torch.long))
    return rows + cols + boxes


GROUP_INDICES = _group_indices()
ROW_INDICES = GROUP_INDICES[:9]
COL_INDICES = GROUP_INDICES[9:18]
BOX_INDICES = GROUP_INDICES[18:]


def _cell_units(index: int) -> tuple[Tensor, Tensor, Tensor]:
    row = index // 9
    col = index % 9
    box = (row // 3) * 3 + (col // 3)
    return ROW_INDICES[row], COL_INDICES[col], BOX_INDICES[box]


CELL_NEIGHBORS: list[set[int]] = []
for idx in range(81):
    row_indices, col_indices, box_indices = _cell_units(idx)
    neighbors = set(row_indices.tolist()) | set(col_indices.tolist()) | set(box_indices.tolist())
    neighbors.discard(idx)
    CELL_NEIGHBORS.append(neighbors)


def find_constraint_violations(board_digits: Tensor) -> Tensor:
    if board_digits.ndim != 2 or board_digits.shape[1] != 81:
        raise ValueError(f"Expected board digits with shape (B, 81), got {tuple(board_digits.shape)}.")

    violations = torch.zeros_like(board_digits, dtype=torch.bool)
    for indices in GROUP_INDICES:
        group = board_digits[:, indices]
        one_hot = F.one_hot(group.clamp(min=0), num_classes=10)[..., 1:].to(dtype=torch.bool)
        duplicate_digits = one_hot.sum(dim=1) > 1
        group_violations = (one_hot & duplicate_digits.unsqueeze(1)).any(dim=-1)
        violations[:, indices] |= group_violations
    return violations


def _conflict_components(conflict_indices: list[int]) -> list[list[int]]:
    remaining = set(conflict_indices)
    components: list[list[int]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            connected = CELL_NEIGHBORS[current] & remaining
            if not connected:
                continue
            remaining.difference_update(connected)
            stack.extend(connected)
            component.extend(sorted(connected))
        components.append(sorted(component))
    return components


def _allowed_digits(board_digits: Tensor, index: int, component_set: set[int]) -> list[int]:
    used: set[int] = set()
    for unit in _cell_units(index):
        for cell_idx in unit.tolist():
            if cell_idx in component_set:
                continue
            value = int(board_digits[cell_idx].item())
            if value != 0:
                used.add(value)
    return [digit for digit in range(1, 10) if digit not in used]


def _search_component_assignment(
    board_digits: Tensor,
    logits: Tensor,
    component: list[int],
    max_candidates_per_cell: int,
) -> dict[int, int] | None:
    if max_candidates_per_cell <= 0:
        raise ValueError("evaluation.local_repair.max_candidates_per_cell must be positive")

    component_set = set(component)
    candidate_map: dict[int, list[int]] = {}
    score_map: dict[tuple[int, int], float] = {}
    for index in component:
        candidates = _allowed_digits(board_digits, index, component_set)
        if not candidates:
            return None
        ranked = sorted(
            candidates,
            key=lambda digit: float(logits[index, digit - 1].item()),
            reverse=True,
        )
        candidate_map[index] = ranked[:max_candidates_per_cell]
        for digit in candidate_map[index]:
            score_map[(index, digit)] = float(logits[index, digit - 1].item())

    ordered = sorted(component, key=lambda idx: (len(candidate_map[idx]), idx))
    best_assignment: dict[int, int] | None = None
    best_score = float("-inf")
    assignment: dict[int, int] = {}

    suffix_upper_bounds = [0.0 for _ in range(len(ordered) + 1)]
    for pos in range(len(ordered) - 1, -1, -1):
        index = ordered[pos]
        suffix_upper_bounds[pos] = suffix_upper_bounds[pos + 1] + max(
            score_map[(index, digit)] for digit in candidate_map[index]
        )

    def backtrack(position: int, current_score: float) -> None:
        nonlocal best_assignment, best_score
        if position == len(ordered):
            if current_score > best_score:
                best_score = current_score
                best_assignment = assignment.copy()
            return

        if current_score + suffix_upper_bounds[position] <= best_score:
            return

        index = ordered[position]
        for digit in candidate_map[index]:
            valid = True
            for neighbor in CELL_NEIGHBORS[index]:
                if assignment.get(neighbor) == digit:
                    valid = False
                    break
            if not valid:
                continue
            assignment[index] = digit
            backtrack(position + 1, current_score + score_map[(index, digit)])
            del assignment[index]

    backtrack(0, 0.0)
    return best_assignment


def repair_conflict_clusters(
    board_digits: Tensor,
    logits: Tensor,
    original_clues: Tensor,
    violations: Tensor,
    max_component_size: int,
    max_candidates_per_cell: int,
) -> Tensor:
    repaired = board_digits.clone()
    conflict_indices = torch.nonzero(violations & ~original_clues, as_tuple=False).reshape(-1).tolist()
    if not conflict_indices:
        return repaired

    for component in _conflict_components(conflict_indices):
        if len(component) > max_component_size:
            continue
        assignment = _search_component_assignment(
            repaired,
            logits,
            component,
            max_candidates_per_cell=max_candidates_per_cell,
        )
        if assignment is None:
            continue
        for index, digit in assignment.items():
            repaired[index] = digit
    return repaired


def infer_filled_board(
    encoder: Encoder,
    representation: SudokuRepresentation,
    puzzle: Tensor,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if temperature <= 0.0:
        raise ValueError("evaluation.inference.temperature must be positive")

    encoded = encoder(puzzle)
    coord_vecs = representation.encode_coordinates(puzzle)
    digit_estimate = torch.nn.functional.normalize(encoded * coord_vecs, dim=-1)
    prototypes = torch.nn.functional.normalize(
        representation.digit_prototypes().to(digit_estimate.device),
        dim=-1,
    )
    logits = torch.einsum("bsd,vd->bsv", digit_estimate, prototypes) / temperature
    probs = torch.softmax(logits, dim=-1)
    confidence = probs.amax(dim=-1)
    pred_digits = logits.argmax(dim=-1).to(dtype=puzzle.dtype) + 1.0
    filled = puzzle.clone()
    empty_mask = filled[..., 2].eq(0)
    filled[..., 2] = torch.where(empty_mask, pred_digits, filled[..., 2])
    prediction_confidence = torch.where(
        empty_mask,
        confidence.to(dtype=puzzle.dtype),
        torch.full_like(confidence, float("nan"), dtype=puzzle.dtype),
    )
    return filled, prediction_confidence, logits


def run_multi_pass_inference(
    encoder: Encoder,
    representation: SudokuRepresentation,
    puzzle: Tensor,
    clue_mask: Tensor,
    max_passes: int,
    temperature: float,
    repair_enabled: bool,
    repair_max_component_size: int,
    repair_max_candidates_per_cell: int,
    progress: tqdm | None = None,
) -> tuple[Tensor, Tensor, Tensor, list[Tensor], list[Tensor], list[Tensor]]:
    if max_passes <= 0:
        raise ValueError("evaluation.inference.max_passes must be positive")

    current_puzzle = puzzle.clone()
    original_clues = clue_mask[..., 0].to(dtype=torch.bool)
    final_board = current_puzzle.clone()
    final_confidence = torch.full_like(puzzle[..., 2], float("nan"))
    solved = torch.zeros(puzzle.shape[0], dtype=torch.bool, device=puzzle.device)
    board_history: list[Tensor] = []
    confidence_history: list[Tensor] = []
    solved_history: list[Tensor] = []

    for pass_idx in range(max_passes):
        final_board, final_confidence, logits = infer_filled_board(
            encoder,
            representation,
            current_puzzle,
            temperature,
        )
        digits = final_board[..., 2].to(dtype=torch.long)
        violations = find_constraint_violations(digits)
        if repair_enabled:
            repaired_digits = digits.clone()
            for board_idx in range(repaired_digits.shape[0]):
                repaired_digits[board_idx] = repair_conflict_clusters(
                    board_digits=repaired_digits[board_idx],
                    logits=logits[board_idx],
                    original_clues=original_clues[board_idx],
                    violations=violations[board_idx],
                    max_component_size=repair_max_component_size,
                    max_candidates_per_cell=repair_max_candidates_per_cell,
                )
            final_board = final_board.clone()
            final_board[..., 2] = repaired_digits.to(dtype=final_board.dtype)
            digits = final_board[..., 2].to(dtype=torch.long)
            violations = find_constraint_violations(digits)
        solved = digits.ne(0).all(dim=-1) & ~violations.any(dim=-1)
        board_history.append(final_board.clone())
        confidence_history.append(final_confidence.clone())
        solved_history.append(solved.clone())
        if progress is not None:
            progress.update(1)
        if solved.all():
            for _ in range(pass_idx + 1, max_passes):
                board_history.append(final_board.clone())
                confidence_history.append(final_confidence.clone())
                solved_history.append(solved.clone())
                if progress is not None:
                    progress.update(1)
            break

        clear_mask = violations & ~original_clues
        updated = final_board.clone()
        updated[..., 2] = updated[..., 2].masked_fill(clear_mask, 0.0)
        current_puzzle = torch.where(
            solved.view(-1, 1, 1),
            final_board,
            updated,
        )

    return final_board, final_confidence, solved, board_history, confidence_history, solved_history


def evaluate_difficulty(
    *,
    encoder: Encoder,
    representation: SudokuRepresentation,
    solutions: Tensor,
    empty_cells: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    max_passes: int,
    temperature: float,
    repair_enabled: bool,
    repair_max_component_size: int,
    repair_max_candidates_per_cell: int,
    device: torch.device,
) -> DifficultyMetrics:
    dataset = SudokuPuzzleDataset(
        num_samples=solutions.shape[0],
        num_cells_to_mask=empty_cells,
        solution_boards=solutions,
        seed=seed,
        randomize_mask_per_access=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    correct_sum = 0.0
    total_sum = 0.0
    solved_sum = 0.0
    example_failed_board: Tensor | None = None
    example_failed_confidence: Tensor | None = None
    pass_correct_sum = [0.0 for _ in range(max_passes)]
    pass_total_sum = [0.0 for _ in range(max_passes)]
    pass_solved_sum = [0.0 for _ in range(max_passes)]

    total_batches = len(loader)
    progress = tqdm(
        total=total_batches * max_passes,
        desc=f"eval empty={empty_cells}",
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for puzzle, target, clue_mask in loader:
            puzzle = puzzle.to(device)
            target = target.to(device)
            clue_mask = clue_mask.to(device)

            final_board, final_confidence, solved, board_history, confidence_history, solved_history = run_multi_pass_inference(
                encoder=encoder,
                representation=representation,
                puzzle=puzzle,
                clue_mask=clue_mask,
                max_passes=max_passes,
                temperature=temperature,
                repair_enabled=repair_enabled,
                repair_max_component_size=repair_max_component_size,
                repair_max_candidates_per_cell=repair_max_candidates_per_cell,
                progress=progress,
            )

            eval_mask = ~clue_mask[..., 0].to(dtype=torch.bool)
            pred_digits = final_board[..., 2].to(dtype=torch.long)
            target_digits = target[..., 2].to(dtype=torch.long)
            correct = pred_digits.eq(target_digits) & eval_mask

            correct_sum += float(correct.sum().item())
            total_sum += float(eval_mask.sum().item())
            solved_sum += float(solved.sum().item())

            if example_failed_board is None:
                failed_indices = torch.nonzero(~solved, as_tuple=False).reshape(-1)
                if failed_indices.numel() > 0:
                    failed_idx = int(failed_indices[0].item())
                    example_failed_board = final_board[failed_idx].detach().cpu()
                    # Show confidence from the first pass over the original puzzle,
                    # so every originally empty cell has a confidence value.
                    example_failed_confidence = confidence_history[0][failed_idx].detach().cpu()

            for pass_idx, (pass_board, pass_solved) in enumerate(
                zip(board_history, solved_history, strict=True)
            ):
                pass_pred_digits = pass_board[..., 2].to(dtype=torch.long)
                pass_correct = pass_pred_digits.eq(target_digits) & eval_mask
                pass_correct_sum[pass_idx] += float(pass_correct.sum().item())
                pass_total_sum[pass_idx] += float(eval_mask.sum().item())
                pass_solved_sum[pass_idx] += float(pass_solved.sum().item())

    progress.close()

    for pass_idx in range(max_passes):
        pass_accuracy = pass_correct_sum[pass_idx] / max(pass_total_sum[pass_idx], 1.0)
        pass_solved_rate = pass_solved_sum[pass_idx] / float(solutions.shape[0])
        pass_unsolved = int(solutions.shape[0] - pass_solved_sum[pass_idx])
        print(
            f"  pass={pass_idx + 1} "
            f"avg_cell_accuracy={pass_accuracy:.6f} "
            f"board_solved_rate={pass_solved_rate:.6f} "
            f"unsolved_board_count={pass_unsolved}"
        )

    avg_cell_accuracy = correct_sum / max(total_sum, 1.0)
    board_solved_rate = solved_sum / float(solutions.shape[0])
    unsolved_board_count = int(solutions.shape[0] - solved_sum)
    return DifficultyMetrics(
        empty_cells=empty_cells,
        avg_cell_accuracy=avg_cell_accuracy,
        board_solved_rate=board_solved_rate,
        unsolved_board_count=unsolved_board_count,
        example_failed_board=example_failed_board,
        example_failed_confidence=example_failed_confidence,
    )


def format_sudoku_board(board_xyz: Tensor) -> str:
    if board_xyz.shape != (81, 3):
        raise ValueError(f"Expected board with shape (81, 3), got {tuple(board_xyz.shape)}.")

    digits = board_xyz[:, 2].to(dtype=torch.long).reshape(9, 9)
    lines = ["+-------+-------+-------+"]
    for row_idx in range(9):
        row = [str(int(value.item())) if int(value.item()) != 0 else "." for value in digits[row_idx]]
        chunks = [" ".join(row[col:col + 3]) for col in range(0, 9, 3)]
        lines.append(f"| {' | '.join(chunks)} |")
        if (row_idx + 1) % 3 == 0:
            lines.append("+-------+-------+-------+")
    return "\n".join(lines)


def format_confidence_board(confidence: Tensor) -> str:
    if confidence.shape != (81,):
        raise ValueError(f"Expected confidence with shape (81,), got {tuple(confidence.shape)}.")

    values = confidence.reshape(9, 9)
    lines = ["+-------------------------+-------------------------+-------------------------+"]
    for row_idx in range(9):
        row: list[str] = []
        for value in values[row_idx]:
            scalar = float(value.item())
            row.append("   .  " if torch.isnan(value) else f"{scalar:0.3f}")
        chunks = [" ".join(row[col:col + 3]) for col in range(0, 9, 3)]
        lines.append(f"| {' | '.join(chunks)} |")
        if (row_idx + 1) % 3 == 0:
            lines.append("+-------------------------+-------------------------+-------------------------+")
    return "\n".join(lines)


def evaluate_checkpoint(config: DictConfig) -> list[DifficultyMetrics]:
    device = torch.device(config.evaluation.device)
    solutions = load_holdout_solutions(
        dataset_path=config.evaluation.data.dataset_path,
        num_samples=config.evaluation.data.num_samples,
    )
    representation, encoder = load_encoder_from_checkpoint(config)
    representation.to(device)
    encoder.to(device)

    metrics: list[DifficultyMetrics] = []
    difficulty_levels = build_difficulty_levels(config)
    for empty_cells in tqdm(difficulty_levels, desc="difficulty", dynamic_ncols=True):
        metrics.append(
            evaluate_difficulty(
                encoder=encoder,
                representation=representation,
                solutions=solutions,
                empty_cells=empty_cells,
                batch_size=config.evaluation.data.batch_size,
                num_workers=config.evaluation.data.num_workers,
                pin_memory=config.evaluation.data.pin_memory,
                seed=config.seed,
                max_passes=config.evaluation.inference.max_passes,
                temperature=config.evaluation.inference.temperature,
                repair_enabled=config.evaluation.local_repair.enabled,
                repair_max_component_size=config.evaluation.local_repair.max_component_size,
                repair_max_candidates_per_cell=config.evaluation.local_repair.max_candidates_per_cell,
                device=device,
            )
        )
    return metrics
