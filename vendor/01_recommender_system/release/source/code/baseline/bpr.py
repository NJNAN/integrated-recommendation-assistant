"""Time-aware Bayesian Personalized Ranking for the competition pipeline.

The implementation deliberately persists only NumPy arrays and string-to-index
maps.  PyTorch is used during fitting, but submission generation does not need a
live torch model and remains a light-weight CPU operation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BPRState:
    user_pos: dict[str, int]
    item_pos: dict[str, int]
    user_factors: np.ndarray
    item_factors: np.ndarray
    item_bias: np.ndarray
    epochs: int
    final_loss: float
    device: str


def train_time_decay_bpr(
    history: pd.DataFrame,
    candidate_item_weights: dict[str, float] | None = None,
    *,
    seed: int = 42,
    factors: int = 48,
    epochs: int = 6,
    batch_size: int = 4096,
    learning_rate: float = 0.025,
    regularization: float = 2e-4,
    half_life_days: float = 120.0,
) -> BPRState:
    """Fit BPR on implicit purchases with recency-weighted positive sampling.

    Negatives are sampled from a blend of training popularity and the unlabeled
    competition candidate distribution.  Candidate rows never become labels;
    their frequencies only make BPR practise the distinctions it will face at
    submission time.
    """
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("v8 的时间衰减 BPR 需要安装 PyTorch") from exc

    users = sorted(history.user_id.astype(str).unique())
    items = sorted(history.item_id.astype(str).unique())
    user_pos = {value: index for index, value in enumerate(users)}
    item_pos = {value: index for index, value in enumerate(items)}

    pairs = history.groupby(["user_id", "item_id"], sort=True).agg(
        last_time=("transaction_time", "max"),
        orders=("invoice_no", "nunique"),
    ).reset_index()
    pair_users = pairs.user_id.astype(str).map(user_pos).to_numpy(np.int64)
    pair_items = pairs.item_id.astype(str).map(item_pos).to_numpy(np.int64)
    reference = pd.Timestamp(history.transaction_time.max()) + pd.Timedelta(seconds=1)
    age_days = (reference - pairs.last_time).dt.total_seconds().to_numpy(float) / 86400.0
    recency = np.exp(-np.maximum(age_days, 0.0) / half_life_days)
    repeat = 1.0 + 0.25 * np.log1p(pairs.orders.to_numpy(float))
    positive_probability = recency * repeat
    positive_probability /= positive_probability.sum()

    popularity = pairs.groupby("item_id").user_id.nunique().reindex(items).fillna(0).to_numpy(float)
    popularity = np.power(popularity + 0.25, 0.75)
    popularity /= popularity.sum()
    candidate = np.asarray([
        float((candidate_item_weights or {}).get(item, 0.0)) for item in items
    ], dtype=float)
    candidate = np.power(candidate + 0.05, 0.75)
    candidate /= candidate.sum()
    negative_probability = 0.35 * popularity + 0.65 * candidate
    negative_probability /= negative_probability.sum()

    seen_by_user: list[set[int]] = [set() for _ in users]
    for user_index, item_index in zip(pair_users, pair_items):
        seen_by_user[int(user_index)].add(int(item_index))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"BPR device: {device}", flush=True)
    generator = np.random.default_rng(seed)
    user_factors = torch.nn.Embedding(len(users), factors).to(device)
    item_factors = torch.nn.Embedding(len(items), factors).to(device)
    item_bias = torch.nn.Embedding(len(items), 1).to(device)
    torch.nn.init.normal_(user_factors.weight, std=0.08)
    torch.nn.init.normal_(item_factors.weight, std=0.08)
    torch.nn.init.zeros_(item_bias.weight)
    optimizer = torch.optim.Adam(
        [*user_factors.parameters(), *item_factors.parameters(), *item_bias.parameters()],
        lr=learning_rate,
    )

    steps_per_epoch = max(24, int(np.ceil(len(pairs) / batch_size)))
    final_loss = float("nan")
    for epoch in range(epochs):
        running = 0.0
        for _ in range(steps_per_epoch):
            selected = generator.choice(
                len(pairs), size=batch_size, replace=True, p=positive_probability
            )
            batch_users = pair_users[selected]
            batch_positive = pair_items[selected]
            batch_negative = generator.choice(
                len(items), size=batch_size, replace=True, p=negative_probability
            ).astype(np.int64, copy=False)
            collision = np.fromiter(
                (int(item) in seen_by_user[int(user)] for user, item in zip(batch_users, batch_negative)),
                dtype=bool,
                count=batch_size,
            )
            attempts = 0
            while collision.any() and attempts < 12:
                batch_negative[collision] = generator.choice(
                    len(items), size=int(collision.sum()), replace=True, p=negative_probability
                )
                collision = np.fromiter(
                    (int(item) in seen_by_user[int(user)] for user, item in zip(batch_users, batch_negative)),
                    dtype=bool,
                    count=batch_size,
                )
                attempts += 1
            # Extremely dense users are rare; a remaining collision is harmless
            # but is excluded so it cannot teach a false preference.
            keep = ~collision
            u = torch.from_numpy(batch_users[keep]).to(device)
            i = torch.from_numpy(batch_positive[keep]).to(device)
            j = torch.from_numpy(batch_negative[keep]).to(device)
            u_vector = user_factors(u)
            i_vector = item_factors(i)
            j_vector = item_factors(j)
            positive_score = (u_vector * i_vector).sum(1) + item_bias(i).squeeze(1)
            negative_score = (u_vector * j_vector).sum(1) + item_bias(j).squeeze(1)
            ranking_loss = functional.softplus(-(positive_score - negative_score)).mean()
            penalty = regularization * (
                u_vector.square().mean() + i_vector.square().mean() +
                j_vector.square().mean() + item_bias(i).square().mean() +
                item_bias(j).square().mean()
            )
            loss = ranking_loss + penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
        final_loss = running / steps_per_epoch
        print(f"BPR epoch {epoch + 1}/{epochs}: loss={final_loss:.6f}", flush=True)

    return BPRState(
        user_pos=user_pos,
        item_pos=item_pos,
        user_factors=user_factors.weight.detach().cpu().numpy().astype(np.float32),
        item_factors=item_factors.weight.detach().cpu().numpy().astype(np.float32),
        item_bias=item_bias.weight.detach().cpu().numpy().ravel().astype(np.float32),
        epochs=epochs,
        final_loss=final_loss,
        device=str(device),
    )


def attach_bpr_state(engine, state: BPRState) -> None:
    """Attach portable BPR arrays to a BaselineEngine for feature generation."""
    engine.competition_bpr_user_pos = state.user_pos
    engine.competition_bpr_item_pos = state.item_pos
    engine.competition_bpr_user_factors = state.user_factors
    engine.competition_bpr_item_factors = state.item_factors
    engine.competition_bpr_item_bias = state.item_bias
    engine.competition_bpr_epochs = state.epochs
    engine.competition_bpr_final_loss = state.final_loss
    engine.competition_bpr_device = state.device
