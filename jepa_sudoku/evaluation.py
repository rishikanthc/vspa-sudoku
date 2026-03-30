from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from jepa_sudoku.data.datamodule import SudokuPuzzleDataset, load_precomputed_solutions
from jepa_sudoku.model.models import Encoder, Predictor, SudokuRepresentation
from jepa_sudoku.training.experiments import build_components


@dataclass(frozen=True)
class DifficultyMetrics:
    empty_cells: int
    avg_cell_accuracy: float
    board_solved_rate: float
    unsolved_board_count: int
    example_failed_board: Tensor | None = None
    example_failed_solution: Tensor | None = None
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


def load_modules_from_checkpoint(
    config: DictConfig,
) -> tuple[SudokuRepresentation, Encoder, Predictor]:
    representation, encoder, predictor = build_components(config)
    state_dict = _checkpoint_state_dict(config.evaluation.checkpoint_path)

    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    predictor_state = {
        key.removeprefix("predictor."): value
        for key, value in state_dict.items()
        if key.startswith("predictor.")
    }
    repr_state = {
        key.removeprefix("representation."): value
        for key, value in state_dict.items()
        if key.startswith("representation.")
    }

    if not encoder_state:
        raise ValueError("Checkpoint does not contain encoder weights.")
    if not predictor_state:
        raise ValueError("Checkpoint does not contain predictor weights.")

    missing, unexpected = encoder.load_state_dict(encoder_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected encoder state load result. missing={missing}, unexpected={unexpected}"
        )
    missing, unexpected = predictor.load_state_dict(predictor_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected predictor state load result. missing={missing}, unexpected={unexpected}"
        )
    missing, unexpected = representation.load_state_dict(repr_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected representation state load result. missing={missing}, unexpected={unexpected}"
        )

    encoder.eval()
    predictor.eval()
    representation.eval()
    return representation, encoder, predictor


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
        for digit in range(1, 10):
            duplicate_mask = group.eq(digit)
            has_duplicate = duplicate_mask.sum(dim=1) > 1
            violations[:, indices] |= duplicate_mask & has_duplicate.unsqueeze(1)
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


def jointly_decode_conflict_groups(
    board_digits: Tensor,
    logits: Tensor,
    original_clues: Tensor,
    violations: Tensor,
    *,
    max_component_size: int = 6,
    top_k_per_cell: int = 4,
) -> Tensor:
    if max_component_size <= 0:
        raise ValueError("max_component_size must be positive.")
    if top_k_per_cell <= 0:
        raise ValueError("top_k_per_cell must be positive.")

    repaired = board_digits.clone()
    conflict_indices = torch.nonzero(violations & ~original_clues, as_tuple=False).reshape(-1).tolist()
    if not conflict_indices:
        return repaired

    for component in _conflict_components(conflict_indices):
        if len(component) > max_component_size:
            continue

        component_set = set(component)
        candidate_map: dict[int, list[int]] = {}
        score_map: dict[tuple[int, int], float] = {}
        for index in component:
            candidates = _allowed_digits(repaired, index, component_set)
            if not candidates:
                candidate_map = {}
                break
            ranked = sorted(
                candidates,
                key=lambda digit: float(logits[index, digit - 1].item()),
                reverse=True,
            )
            candidate_map[index] = ranked[:top_k_per_cell]
            for digit in candidate_map[index]:
                score_map[(index, digit)] = float(logits[index, digit - 1].item())
        if not candidate_map:
            continue

        ordered = sorted(component, key=lambda idx: (len(candidate_map[idx]), idx))
        suffix_upper_bounds = [0.0 for _ in range(len(ordered) + 1)]
        for pos in range(len(ordered) - 1, -1, -1):
            index = ordered[pos]
            suffix_upper_bounds[pos] = suffix_upper_bounds[pos + 1] + max(
                score_map[(index, digit)] for digit in candidate_map[index]
            )

        best_assignment: dict[int, int] | None = None
        best_score = float("-inf")
        assignment: dict[int, int] = {}

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
        if best_assignment is None:
            continue
        for index, digit in best_assignment.items():
            repaired[index] = digit

    return repaired


def _pad_context_from_board(board: Tensor) -> tuple[Tensor, Tensor]:
    context_lengths = board[..., 2].ne(0).sum(dim=1)
    max_context = int(context_lengths.max().item())
    batch_size = board.shape[0]
    device = board.device
    dtype = board.dtype

    padded = torch.zeros((batch_size, max_context, 3), device=device, dtype=dtype)
    mask = torch.zeros((batch_size, max_context), device=device, dtype=torch.bool)

    for batch_idx in range(batch_size):
        context = board[batch_idx, board[batch_idx, :, 2].ne(0)]
        length = context.shape[0]
        if length == 0:
            continue
        padded[batch_idx, :length] = context
        mask[batch_idx, :length] = True
    return padded, mask


def infer_filled_board(
    encoder: Encoder,
    predictor: Predictor,
    representation: SudokuRepresentation,
    current_board: Tensor,
    query_coordinates: Tensor,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if temperature <= 0.0:
        raise ValueError("evaluation.inference.temperature must be positive")

    context_xyz, context_mask = _pad_context_from_board(current_board)
    context_tokens = torch.zeros(
        (current_board.shape[0], context_xyz.shape[1], representation.d_model),
        device=current_board.device,
        dtype=current_board.dtype,
    )
    for batch_idx in range(current_board.shape[0]):
        length = int(context_mask[batch_idx].sum().item())
        if length == 0:
            continue
        context_tokens[batch_idx, :length] = representation.encode_board(
            context_xyz[batch_idx : batch_idx + 1, :length]
        )[0]
    encoded_context = encoder(context_tokens, attention_mask=context_mask)
    query_tokens = representation.encode_coordinates(query_coordinates)
    predicted_vectors = predictor(
        encoded_context,
        query_tokens,
        context_mask=context_mask,
    )

    logits = representation.logits_from_predictions(predicted_vectors, query_coordinates) / temperature
    probs = torch.softmax(logits, dim=-1)
    confidence = probs.amax(dim=-1)
    pred_digits = logits.argmax(dim=-1).to(dtype=current_board.dtype) + 1.0

    filled = current_board.clone()
    filled[..., 2] = filled[..., 2].scatter(1, _query_indices(query_coordinates, current_board.device), pred_digits)

    full_confidence = torch.full_like(current_board[..., 2], float("nan"))
    full_confidence = full_confidence.scatter(
        1,
        _query_indices(query_coordinates, current_board.device),
        confidence.to(dtype=full_confidence.dtype),
    )
    return filled, full_confidence, logits


def _query_indices(query_coordinates: Tensor, device: torch.device) -> Tensor:
    x = query_coordinates[..., 0].to(dtype=torch.long) - 1
    y = query_coordinates[..., 1].to(dtype=torch.long) - 1
    return (x * 9 + y).to(device=device)


def _scatter_query_logits_to_board(logits: Tensor, query_coordinates: Tensor) -> Tensor:
    board_logits = torch.full(
        (logits.shape[0], 81, logits.shape[-1]),
        float("-inf"),
        device=logits.device,
        dtype=logits.dtype,
    )
    query_indices = _query_indices(query_coordinates, logits.device)
    board_logits.scatter_(
        1,
        query_indices.unsqueeze(-1).expand(-1, -1, logits.shape[-1]),
        logits,
    )
    return board_logits


def run_multi_pass_inference(
    encoder: Encoder,
    predictor: Predictor,
    representation: SudokuRepresentation,
    puzzle: Tensor,
    query_coordinates: Tensor,
    max_passes: int,
    temperature: float,
    joint_decode_max_component_size: int = 10,
    joint_decode_top_k_per_cell: int = 5,
    progress: tqdm | None = None,
) -> tuple[Tensor, Tensor, Tensor, list[Tensor], list[Tensor], list[Tensor]]:
    if max_passes <= 0:
        raise ValueError("evaluation.inference.max_passes must be positive")

    current_puzzle = puzzle.clone()
    original_clues = puzzle[..., 2].ne(0)
    final_board = current_puzzle.clone()
    final_confidence = torch.full_like(puzzle[..., 2], float("nan"))
    solved = torch.zeros(puzzle.shape[0], dtype=torch.bool, device=puzzle.device)
    board_history: list[Tensor] = []
    confidence_history: list[Tensor] = []
    solved_history: list[Tensor] = []

    for pass_idx in range(max_passes):
        final_board, final_confidence, logits = infer_filled_board(
            encoder,
            predictor,
            representation,
            current_puzzle,
            query_coordinates,
            temperature,
        )
        board_logits = _scatter_query_logits_to_board(logits, query_coordinates)
        digits = final_board[..., 2].to(dtype=torch.long)
        violations = find_constraint_violations(digits)
        if violations.any():
            repaired_digits = digits.clone()
            for board_idx in range(repaired_digits.shape[0]):
                repaired_digits[board_idx] = jointly_decode_conflict_groups(
                    repaired_digits[board_idx],
                    board_logits[board_idx],
                    original_clues[board_idx],
                    violations[board_idx],
                    max_component_size=joint_decode_max_component_size,
                    top_k_per_cell=joint_decode_top_k_per_cell,
                )
            final_board = final_board.clone()
            final_board[..., 2] = repaired_digits.to(dtype=final_board.dtype)
            digits = repaired_digits
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
    predictor: Predictor,
    representation: SudokuRepresentation,
    solutions: Tensor,
    empty_cells: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    max_passes: int,
    temperature: float,
    joint_decode_max_component_size: int,
    joint_decode_top_k_per_cell: int,
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
    example_failed_solution: Tensor | None = None
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
        for puzzle, target, queries in loader:
            puzzle = puzzle.to(device)
            target = target.to(device)
            queries = queries.to(device)

            full_puzzle = torch.zeros((puzzle.shape[0], 81, 3), device=device, dtype=puzzle.dtype)
            xy = torch.cartesian_prod(
                torch.arange(1, 10, device=device),
                torch.arange(1, 10, device=device),
            ).to(dtype=puzzle.dtype)
            full_puzzle[:, :, :2] = xy
            puzzle_indices = _query_indices(puzzle[..., :2], device) if puzzle.shape[1] > 0 else torch.empty((puzzle.shape[0], 0), dtype=torch.long, device=device)
            target_indices = _query_indices(target[..., :2], device)
            if puzzle.shape[1] > 0:
                full_puzzle[..., 2].scatter_(1, puzzle_indices, puzzle[..., 2])

            final_board, final_confidence, solved, board_history, confidence_history, solved_history = run_multi_pass_inference(
                encoder=encoder,
                predictor=predictor,
                representation=representation,
                puzzle=full_puzzle,
                query_coordinates=queries,
                max_passes=max_passes,
                temperature=temperature,
                joint_decode_max_component_size=joint_decode_max_component_size,
                joint_decode_top_k_per_cell=joint_decode_top_k_per_cell,
                progress=progress,
            )

            pred_digits = final_board[..., 2].gather(1, target_indices).to(dtype=torch.long)
            target_digits = target[..., 2].to(dtype=torch.long)
            correct = pred_digits.eq(target_digits)

            correct_sum += float(correct.sum().item())
            total_sum += float(target_digits.numel())
            solved_sum += float(solved.sum().item())

            if example_failed_board is None:
                failed_indices = torch.nonzero(~solved, as_tuple=False).reshape(-1)
                if failed_indices.numel() > 0:
                    failed_idx = int(failed_indices[0].item())
                    example_failed_board = final_board[failed_idx].detach().cpu()
                    solved_board = torch.zeros_like(final_board[failed_idx])
                    solved_board[:, :2] = final_board[failed_idx, :, :2]
                    solved_board[:, 2] = final_board[failed_idx, :, 2]
                    solved_board[target_indices[failed_idx], 2] = target[failed_idx, :, 2]
                    example_failed_solution = solved_board.detach().cpu()
                    example_failed_confidence = confidence_history[0][failed_idx].detach().cpu()

            for pass_idx, (pass_board, pass_solved) in enumerate(
                zip(board_history, solved_history, strict=True)
            ):
                pass_pred_digits = pass_board[..., 2].gather(1, target_indices).to(dtype=torch.long)
                pass_correct = pass_pred_digits.eq(target_digits)
                pass_correct_sum[pass_idx] += float(pass_correct.sum().item())
                pass_total_sum[pass_idx] += float(target_digits.numel())
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
        example_failed_solution=example_failed_solution,
        example_failed_confidence=example_failed_confidence,
    )


def format_sudoku_board(board_xyz: Tensor, solution_xyz: Tensor | None = None) -> str:
    if board_xyz.shape != (81, 3):
        raise ValueError(f"Expected board with shape (81, 3), got {tuple(board_xyz.shape)}.")
    if solution_xyz is not None and solution_xyz.shape != (81, 3):
        raise ValueError(
            f"Expected solution with shape (81, 3), got {tuple(solution_xyz.shape)}."
        )

    digits = board_xyz[:, 2].to(dtype=torch.long).reshape(9, 9)
    solution_digits = (
        solution_xyz[:, 2].to(dtype=torch.long).reshape(9, 9) if solution_xyz is not None else None
    )
    lines = ["+-------+-------+-------+"]
    for row_idx in range(9):
        row: list[str] = []
        for col_idx, value in enumerate(digits[row_idx]):
            digit = int(value.item())
            cell = str(digit) if digit != 0 else "."
            if solution_digits is not None:
                solution_digit = int(solution_digits[row_idx, col_idx].item())
                if digit != solution_digit:
                    cell = f"[{cell}]"
            row.append(cell)
        chunks = [" ".join(row[col:col + 3]) for col in range(0, 9, 3)]
        lines.append(f"| {' | '.join(chunks)} |")
        if (row_idx + 1) % 3 == 0:
            lines.append("+-------+-------+-------+")
    if solution_digits is not None:
        lines.append("Legend: cells in [brackets] differ from the solution.")
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
    representation, encoder, predictor = load_modules_from_checkpoint(config)
    representation.to(device)
    encoder.to(device)
    predictor.to(device)

    metrics: list[DifficultyMetrics] = []
    difficulty_levels = build_difficulty_levels(config)
    for empty_cells in tqdm(difficulty_levels, desc="difficulty", dynamic_ncols=True):
        metrics.append(
            evaluate_difficulty(
                encoder=encoder,
                predictor=predictor,
                representation=representation,
                solutions=solutions,
                empty_cells=empty_cells,
                batch_size=config.evaluation.data.batch_size,
                num_workers=config.evaluation.data.num_workers,
                pin_memory=config.evaluation.data.pin_memory,
                seed=config.seed,
                max_passes=config.evaluation.inference.max_passes,
                temperature=config.evaluation.inference.temperature,
                joint_decode_max_component_size=config.evaluation.joint_decoding.max_component_size,
                joint_decode_top_k_per_cell=config.evaluation.joint_decoding.top_k_per_cell,
                device=device,
            )
        )
    return metrics
