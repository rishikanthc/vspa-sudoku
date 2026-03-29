from __future__ import annotations

import torch

from jepa_sudoku.model.losses import cosine_loss, masked_cosine_loss


def test_cosine_loss_matches_expected_values() -> None:
    a = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    b = torch.tensor([[[1.0, 0.0], [0.0, -1.0]]])
    loss = cosine_loss(a, b)
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)


def test_masked_cosine_loss_only_scores_selected_cells() -> None:
    pred = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
    target = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]], dtype=torch.float32)
    include_mask = torch.tensor([[True, False]])

    masked = masked_cosine_loss(pred, target, include_mask)
    unmasked = cosine_loss(pred, target)

    assert torch.isclose(masked, torch.tensor(0.0), atol=1e-6)
    assert unmasked > masked
