from __future__ import annotations

import torch

from jepa_sudoku.model.models import Encoder, SudokuRepresentation, TransformerConfig


def _build_components() -> tuple[SudokuRepresentation, Encoder]:
    config = TransformerConfig(
        context_size=81,
        n_heads=2,
        head_dim=16,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    )
    representation = SudokuRepresentation(d_model=config.d_model, seed=123)
    encoder = Encoder(config, representation=representation)
    return representation, encoder


def test_representation_encodes_empty_cells_as_coordinates_only() -> None:
    representation, _ = _build_components()
    board = torch.tensor([[[1.0, 1.0, 0.0], [1.0, 2.0, 7.0]]])

    encoded = representation.encode_board(board)
    coords = representation.encode_coordinates(board)
    targets = representation.encode_targets(board)

    assert torch.allclose(encoded[:, :1], coords[:, :1], atol=1e-6)
    assert torch.allclose(encoded[:, 1:], targets[:, 1:], atol=1e-6)


def test_encoder_outputs_full_board_vectors_and_decodes_digits() -> None:
    representation, encoder = _build_components()
    board = torch.zeros(2, 81, 3)
    xy = torch.cartesian_prod(torch.arange(1, 10), torch.arange(1, 10)).float()
    board[:, :, :2] = xy
    board[:, :, 2] = 0

    encoded = encoder(board)
    logits = representation.logits_from_predictions(representation.encode_targets(board + torch.tensor([0.0, 0.0, 1.0])), board)

    assert encoded.shape == (2, 81, representation.d_model)
    assert logits.shape == (2, 81, 9)
