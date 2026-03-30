from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

from jepa_sudoku.evaluation import (
    evaluate_checkpoint,
    format_confidence_board,
    format_sudoku_board,
)


@hydra_main(config_path="configs", config_name="eval", version_base=None)
def main(config: DictConfig) -> None:
    OmegaConf.resolve(config)
    metrics = evaluate_checkpoint(config)
    print("Evaluation complete.")
    for metric in metrics:
        print(
            f"empty_cells={metric.empty_cells} "
            f"avg_cell_accuracy={metric.avg_cell_accuracy:.6f} "
            f"board_solved_rate={metric.board_solved_rate:.6f} "
            f"unsolved_board_count={metric.unsolved_board_count}"
        )
        if metric.example_failed_board is not None:
            print(f"example_failed_board empty_cells={metric.empty_cells}")
            print(format_sudoku_board(metric.example_failed_board, metric.example_failed_solution))
        if metric.example_failed_confidence is not None:
            print(f"example_failed_confidence empty_cells={metric.empty_cells}")
            print(format_confidence_board(metric.example_failed_confidence))


if __name__ == "__main__":
    main()
