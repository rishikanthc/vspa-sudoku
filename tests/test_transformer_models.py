from __future__ import annotations

import torch

from jepa_sudoku.model.models import Encoder, Predictor, SudokuRepresentation, TransformerConfig


def _build_components() -> tuple[SudokuRepresentation, Encoder, Predictor]:
    config = TransformerConfig(
        context_size=81,
        n_heads=2,
        head_dim=16,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    )
    representation = SudokuRepresentation(d_model=config.d_model, seed=123)
    encoder = Encoder(config)
    predictor = Predictor(config)
    return representation, encoder, predictor


def test_representation_encodes_bound_cell_vectors() -> None:
    representation, _, _ = _build_components()
    board = torch.tensor([[[1.0, 1.0, 4.0], [1.0, 2.0, 7.0]]])

    encoded = representation.encode_board(board)
    coords = representation.encode_coordinates(board)
    targets = representation.encode_targets(board)

    assert encoded.shape == (1, 2, representation.d_model)
    assert not torch.allclose(encoded, coords, atol=1e-6)
    assert torch.allclose(encoded, targets, atol=1e-6)


def test_encoder_and_predictor_match_sparse_contract() -> None:
    representation, encoder, predictor = _build_components()
    puzzle = torch.tensor(
        [[[1.0, 1.0, 5.0], [1.0, 2.0, 3.0], [1.0, 3.0, 7.0]]],
        dtype=torch.float32,
    )
    queries = torch.tensor(
        [[[1.0, 4.0], [1.0, 5.0]]],
        dtype=torch.float32,
    )

    puzzle_vectors = representation.encode_board(puzzle)
    query_vectors = representation.encode_coordinates(queries)
    encoded = encoder(puzzle_vectors)
    predicted = predictor(encoded, query_vectors)
    logits = representation.logits_from_predictions(predicted, queries)

    assert encoded.shape == (1, 3, representation.d_model)
    assert predicted.shape == (1, 2, representation.d_model)
    assert logits.shape == (1, 2, 9)
