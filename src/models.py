"""Recommendation models: BPR-MF and LightGCN.

Both models are trained with the shared BPR loss (see ``src.losses``) and expose
a ``full_sort_scores`` method that returns a dense score matrix of shape
``[num_users, num_items]``. That matrix is the single hand-off point to the
evaluation pipeline: pass it straight to ``src.metrics.evaluate_full_ranking``,
which masks training items and computes all top-K / long-tail metrics. Models do
**not** mask training interactions themselves.

ID conventions (must match ``src.data`` / the preprocessing notebook):
    * users are indexed ``0 .. num_users - 1``.
    * items are indexed ``0 .. num_items - 1``.
    * in the LightGCN graph, item ``i`` is node ``i + num_users`` and
      ``edge_index_train`` is a bidirectional ``[2, num_edges]`` edge list already
      built with that offset.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import EMBEDDING_DIM, NUM_LAYERS


class BPRMF(nn.Module):
    """Matrix-factorization recommender trained with BPR loss.

    A personalized, non-graph baseline: each user and item has a free embedding
    vector, and the score of a ``(user, item)`` pair is their dot product.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Small random init keeps early dot products near zero (standard for BPR-MF).
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Score aligned ``(user, item)`` pairs; returns shape ``[batch_size]``."""
        u = self.user_embedding(users)
        i = self.item_embedding(items)
        return (u * i).sum(dim=-1)

    def bpr_forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute positive/negative scores plus ego embeddings for a triplet batch.

        Returns:
            ``(pos_scores, neg_scores, (user_emb, pos_emb, neg_emb))``. The three
            embedding tensors are returned so the training loop can apply L2
            regularization to exactly the parameters used in this step.
        """
        u = self.user_embedding(users)
        p = self.item_embedding(pos_items)
        n = self.item_embedding(neg_items)
        pos_scores = (u * p).sum(dim=-1)
        neg_scores = (u * n).sum(dim=-1)
        return pos_scores, neg_scores, (u, p, n)

    @torch.no_grad()
    def full_sort_scores(self) -> torch.Tensor:
        """Return the full score matrix ``[num_users, num_items]``.

        ``scores[u, i]`` is the dot product of user ``u`` and item ``i``. Feed the
        result directly to ``evaluate_full_ranking`` (training-item masking is
        handled there, not here).
        """
        return self.user_embedding.weight @ self.item_embedding.weight.t()


class LightGCNRecommender(nn.Module):
    """LightGCN: graph collaborative filtering on the user-item bipartite graph.

    A single embedding table holds one vector per *node* (users first, then
    items). Final embeddings are the layer-wise average of repeated symmetric-
    normalized neighborhood aggregation (no feature transform, no non-linearity),
    exactly as in the LightGCN paper. Scores are dot products of the propagated
    user and item embeddings.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = EMBEDDING_DIM,
        num_layers: int = NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # One embedding per node: rows [0, num_users) are users, the rest items.
        self.embedding = nn.Embedding(self.num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

        # Cache for the normalized adjacency, keyed on the edge_index identity so
        # we recompute only when the graph actually changes (e.g. edge dropout).
        self._adj_cache: torch.Tensor | None = None
        self._adj_key: tuple[int, int] | None = None

    def _normalized_adjacency(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Build (and cache) the symmetric-normalized sparse adjacency D^-1/2 A D^-1/2."""
        key = (edge_index.data_ptr(), int(edge_index.size(1)))
        if self._adj_cache is not None and self._adj_key == key \
                and self._adj_cache.device == self.embedding.weight.device:
            return self._adj_cache

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must have shape [2, num_edges], got {tuple(edge_index.shape)}"
            )

        device = self.embedding.weight.device
        edge_index = edge_index.to(device)
        row, col = edge_index[0], edge_index[1]

        deg = torch.bincount(row, minlength=self.num_nodes).to(torch.float32)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        adj = torch.sparse_coo_tensor(
            edge_index, norm, (self.num_nodes, self.num_nodes)
        ).coalesce()

        self._adj_cache = adj
        self._adj_key = key
        return adj

    def propagate(self, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run LightGCN message passing and split node embeddings.

        Args:
            edge_index: Bidirectional graph, shape ``[2, num_edges]``, with items
                offset by ``num_users`` (i.e. ``edge_index_train``).

        Returns:
            ``(user_embeddings, item_embeddings)`` with shapes
            ``[num_users, dim]`` and ``[num_items, dim]``, each the layer-average
            of the propagated embeddings.
        """
        adj = self._normalized_adjacency(edge_index)

        e = self.embedding.weight                 # E^(0)
        out = e                                    # accumulate E^(0) + ... + E^(K)
        for _ in range(self.num_layers):
            e = torch.sparse.mm(adj, e)            # E^(k+1) = A_norm @ E^(k)
            out = out + e
        out = out / (self.num_layers + 1)          # layer-wise mean

        user_emb = out[: self.num_users]
        item_emb = out[self.num_users:]
        return user_emb, item_emb

    def bpr_forward(
        self,
        edge_index: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Propagate once, then score a triplet batch.

        Returns:
            ``(pos_scores, neg_scores, (user_ego, pos_ego, neg_ego))`` where the
            three ego tensors are the *initial* (layer-0) embeddings of the batch
            nodes, used for L2 regularization (LightGCN regularizes E^(0) only).
        """
        user_emb, item_emb = self.propagate(edge_index)

        u = user_emb[users]
        p = item_emb[pos_items]
        n = item_emb[neg_items]
        pos_scores = (u * p).sum(dim=-1)
        neg_scores = (u * n).sum(dim=-1)

        u0 = self.embedding(users)
        p0 = self.embedding(pos_items + self.num_users)
        n0 = self.embedding(neg_items + self.num_users)
        return pos_scores, neg_scores, (u0, p0, n0)

    @torch.no_grad()
    def full_sort_scores(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Return the full score matrix ``[num_users, num_items]``.

        Propagates on ``edge_index`` and returns ``user_emb @ item_emb.T``. Feed
        the result directly to ``evaluate_full_ranking`` (masking handled there).
        """
        user_emb, item_emb = self.propagate(edge_index)
        return user_emb @ item_emb.t()
