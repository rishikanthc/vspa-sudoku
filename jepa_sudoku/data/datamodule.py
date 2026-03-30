from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .sudoku_generator import SudokuBoardGenerator

MaskSchedule = Callable[[int], int]


@dataclass(frozen=True)
class LinearMaskCurriculum:
    start: int
    max_mask: int
    num_steps: int = 1

    def __post_init__(self) -> None:
        if self.num_steps < 0:
            raise ValueError("num_steps must be >= 0")
        if not 0 <= self.start <= 81:
            raise ValueError("start must be between 0 and 81")
        if not 0 <= self.max_mask <= 81:
            raise ValueError("max_mask must be between 0 and 81")
        if self.max_mask < self.start:
            raise ValueError("max_mask must be >= start")

    def __call__(self, step: int) -> int:
        if self.num_steps == 0:
            return self.max_mask

        clamped_step = max(0, step)
        if clamped_step >= self.num_steps:
            return self.max_mask

        ratio = clamped_step / self.num_steps
        return int(round(self.start + (self.max_mask - self.start) * ratio))


@dataclass(frozen=True)
class SudokuDataConfig:
    num_samples: int
    num_cells_to_mask: int
    dataset_path: str | None = None
    seed: int = 0
    unique_solution: bool = False
    randomize_mask_per_access: bool = True
    mask_cells_curriculum: MaskSchedule | None = None
    batch_size: int = 32
    num_workers: int = 0
    shuffle: bool = True
    pin_memory: bool = False
    drop_last: bool = False


def _board_to_value_tensor(board: list[list[int]]) -> Tensor:
    values = [cell for row in board for cell in row]
    return torch.tensor(values, dtype=torch.float32)


def load_precomputed_solutions(dataset_path: str) -> Tensor:
    path = Path(dataset_path)
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dataset payload dict in {path}, got {type(payload)!r}")
    solutions = payload.get("solution_boards")
    if not isinstance(solutions, torch.Tensor):
        raise ValueError(f"Dataset {path} is missing a tensor field named 'solution_boards'.")
    if solutions.ndim != 2 or solutions.shape[1] != 81:
        raise ValueError(
            f"Expected solution_boards with shape (N, 81), got {tuple(solutions.shape)}."
        )
    return solutions.contiguous().to(device="cpu")


def select_solution_subset(solution_boards: Tensor, num_samples: int) -> Tensor:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    available = int(solution_boards.shape[0])
    if num_samples > available:
        raise ValueError(
            f"Requested num_samples={num_samples}, but dataset only has {available} solution boards."
        )
    return solution_boards[:num_samples].contiguous()


class SudokuPuzzleDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """
    Dataset yielding:
      - puzzle:   (N, 3) given cells [x, y, z]
      - target:   (81 - N, 3) solved values for empty cells [x, y, z]
      - queries:  (81 - N, 2) empty-cell coordinates [x, y]
    """

    def __init__(
        self,
        num_samples: int,
        num_cells_to_mask: int,
        solution_boards: Tensor | None = None,
        seed: int = 0,
        unique_solution: bool = False,
        randomize_mask_per_access: bool = True,
        mask_cells_curriculum: MaskSchedule | None = None,
    ) -> None:
        if num_samples < 0:
            raise ValueError("num_samples must be >= 0")
        if not 0 <= num_cells_to_mask <= 81:
            raise ValueError("num_cells_to_mask must be between 0 and 81 inclusive")

        self._solution_boards = (
            select_solution_subset(solution_boards.to(device="cpu"), num_samples)
            if solution_boards is not None
            else None
        )
        self.num_samples = num_samples
        self.num_cells_to_mask = num_cells_to_mask
        self.seed = int(seed)
        self.unique_solution = unique_solution
        self.randomize_mask_per_access = randomize_mask_per_access
        self.mask_cells_curriculum = mask_cells_curriculum
        self.current_epoch = 0
        self._manual_num_cells_to_mask: int | None = None
        self._access_counts: dict[int, int] = {}
        self._solution_cache: dict[int, Tensor] = {}

        xs = torch.arange(1, 10)
        ys = torch.arange(1, 10)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        self._xy = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1).float()

    def _sample_seed(self, index: int, *, epoch: int = 0) -> int:
        return (self.seed * 1_000_003) + index + (epoch * 10_000_019)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def set_num_cells_to_mask(self, num_cells_to_mask: int) -> None:
        num_cells = int(num_cells_to_mask)
        if not 0 <= num_cells <= 81:
            raise ValueError("num_cells_to_mask must be between 0 and 81 inclusive")
        self._manual_num_cells_to_mask = num_cells

    def _num_cells_for_epoch(self) -> int:
        if self._manual_num_cells_to_mask is not None:
            return self._manual_num_cells_to_mask
        if self.mask_cells_curriculum is None:
            return self.num_cells_to_mask
        return max(0, min(81, int(self.mask_cells_curriculum(self.current_epoch))))

    def _get_or_build_template(self, index: int) -> Tensor:
        if self._solution_boards is not None:
            return self._solution_boards[index].to(dtype=torch.float32)
        if index not in self._solution_cache:
            sample_seed = self._sample_seed(index)
            generator = SudokuBoardGenerator(seed=sample_seed)
            solution_board = generator.generate_full_board()
            self._solution_cache[index] = _board_to_value_tensor(solution_board)
        return self._solution_cache[index]

    def _mask_seed(self, index: int) -> int:
        base_seed = self.seed + (index * 1_000_003) + self._num_cells_for_epoch()
        if not self.randomize_mask_per_access:
            return base_seed
        access_count = self._access_counts.get(index, 0)
        return base_seed + (self.current_epoch * 10_007) + (access_count * 1_009)

    def _masked_indices(self, index: int, num_cells_to_mask: int) -> Tensor:
        if num_cells_to_mask <= 0:
            return torch.empty((0,), dtype=torch.long)

        generator = torch.Generator()
        generator.manual_seed(self._mask_seed(index))
        return torch.randperm(81, generator=generator)[:num_cells_to_mask]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        if not 0 <= index < self.num_samples:
            raise IndexError(
                f"Index {index} out of range for dataset of size {self.num_samples}"
            )

        if self.randomize_mask_per_access:
            self._access_counts[index] = self._access_counts.get(index, 0) + 1

        num_cells_to_mask = self._num_cells_for_epoch()
        if self.unique_solution:
            sample_seed = self._sample_seed(index, epoch=self.current_epoch)
            puzzle_data = SudokuBoardGenerator(seed=sample_seed).generate_puzzle(
                removed_cells=min(num_cells_to_mask, 80),
                unique_solution=True,
            )
            solution_values = _board_to_value_tensor(puzzle_data.solution)
            puzzle_values = _board_to_value_tensor(puzzle_data.puzzle)
            if int((puzzle_values == 0).sum()) != num_cells_to_mask:
                solution_values = self._get_or_build_template(index)
                puzzle_values = solution_values.clone()
        else:
            solution_values = self._get_or_build_template(index)
            puzzle_values = solution_values.clone()

        n = min(num_cells_to_mask, 81)
        if n > 0 and not self.unique_solution:
            masked_indices = self._masked_indices(index, n)
            puzzle_values[masked_indices] = 0.0

        clue_mask = puzzle_values.ne(0.0)
        empty_mask = ~clue_mask

        puzzle = torch.empty((int(clue_mask.sum().item()), 3), dtype=torch.float32)
        puzzle[:, :2] = self._xy[clue_mask]
        puzzle[:, 2] = puzzle_values[clue_mask]

        target = torch.empty((int(empty_mask.sum().item()), 3), dtype=torch.float32)
        target[:, :2] = self._xy[empty_mask]
        target[:, 2] = solution_values[empty_mask]

        queries = self._xy[empty_mask].clone()
        return puzzle, target, queries


class SudokuDataModule:
    def __init__(self, config: SudokuDataConfig) -> None:
        self.config = config
        self.dataset = SudokuPuzzleDataset(
            num_samples=config.num_samples,
            num_cells_to_mask=config.num_cells_to_mask,
            solution_boards=(
                load_precomputed_solutions(config.dataset_path)
                if config.dataset_path is not None
                else None
            ),
            seed=config.seed,
            unique_solution=config.unique_solution,
            randomize_mask_per_access=config.randomize_mask_per_access,
            mask_cells_curriculum=config.mask_cells_curriculum,
        )
        self._dataloader_generator = torch.Generator().manual_seed(config.seed)

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)

    def set_num_cells_to_mask(self, num_cells_to_mask: int) -> None:
        self.dataset.set_num_cells_to_mask(num_cells_to_mask)

    @property
    def current_num_cells_to_mask(self) -> int:
        return self.dataset._num_cells_for_epoch()

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            generator=self._dataloader_generator,
        )

    def get_single(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)
