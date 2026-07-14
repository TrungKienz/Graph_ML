"""Loss functions for pairwise recommendation training.

The Bayesian Personalized Ranking (BPR) loss is shared by BPR-MF and LightGCN.
Both models are trained on ``(user, positive_item, negative_item)`` triplets and
optimize the pairwise ranking objective

    L_BPR = - mean over triplets of  log(sigmoid(score_ui - score_uj))

where ``i`` is a positive (interacted) item and ``j`` is a sampled negative item.

L2 regularization on the model embeddings is applied separately in the training
loop (see :func:`src.losses.l2_regularization`) rather than baked into this loss,
so the same ``bpr_loss`` can be reused unchanged by later variants such as
Item Loss Equalization (ILE), which reweights the *per-interaction* BPR terms.
For that reason ``bpr_loss`` supports ``reduction="none"`` to expose the loss of
every individual triplet.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Bayesian Personalized Ranking loss for a batch of triplets.

    Args:
        pos_scores: Predicted scores ``score_ui`` for the positive items,
            shape ``[batch_size]``.
        neg_scores: Predicted scores ``score_uj`` for the sampled negative items,
            shape ``[batch_size]`` (must broadcast against ``pos_scores``).
        reduction: ``"mean"`` (default) returns the scalar mean loss, ``"sum"``
            returns the summed loss, and ``"none"`` returns the per-triplet loss
            with shape ``[batch_size]``. ``"none"`` is what ILE needs so it can
            reweight each interaction by its item's popularity.

    Returns:
        The BPR loss, either a scalar (``mean``/``sum``) or a per-triplet tensor
        (``none``).

    Notes:
        Implemented with ``logsigmoid`` for numerical stability, which is
        equivalent to ``-log(sigmoid(pos - neg))`` but avoids overflow when the
        score gap is large in magnitude.
    """
    if pos_scores.shape != neg_scores.shape:
        raise ValueError(
            f"pos_scores and neg_scores must have the same shape, "
            f"got {tuple(pos_scores.shape)} and {tuple(neg_scores.shape)}"
        )

    per_triplet = -F.logsigmoid(pos_scores - neg_scores)

    if reduction == "none":
        return per_triplet
    if reduction == "sum":
        return per_triplet.sum()
    if reduction == "mean":
        return per_triplet.mean()
    raise ValueError(f"reduction must be one of 'mean', 'sum', 'none', got {reduction!r}")


def l2_regularization(*embeddings: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
    """Sum of squared L2 norms of the given embedding tensors.

    This is the regularization term added to the BPR loss. It is applied to the
    *initial* (ego) embeddings of the users/items in the batch, following the
    LightGCN convention, and is scaled by ``config.WEIGHT_DECAY`` in the training
    loop.

    Args:
        *embeddings: Any number of embedding tensors (e.g. the batch's user,
            positive-item, and negative-item ego embeddings).
        batch_size: If given, the total squared norm is divided by this value so
            the penalty is per-interaction rather than per-batch. Pass the number
            of triplets in the batch to keep the regularization strength
            independent of batch size.

    Returns:
        A scalar tensor with the (optionally averaged) squared L2 norm.
    """
    reg = pos = None
    for emb in embeddings:
        term = emb.pow(2).sum()
        reg = term if reg is None else reg + term
    if reg is None:
        raise ValueError("l2_regularization requires at least one embedding tensor")
    if batch_size is not None:
        reg = reg / batch_size
    return reg
