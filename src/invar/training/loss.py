"""InVAR hybrid loss.

Components and weights (per spec section "Loss"):

  - huber          weight 1.0   on (y_hat, y_cs)
  - listwise IC    weight 0.5   listwise_lambdarank_ic surrogate; we ship
                                a differentiable Pearson-IC variant
  - pairwise       weight 0.3   RSR-style margin loss on within-day
                                ticker pairs
  - regime CE      weight 0.1   on (regime_logits, GMM label)
  - vol MSE        weight 0.1   on (vol_hat, fwd_vol_20d)
  - entropy reg    weight 0.01  on the regime cross-attention weights
                                (negative entropy added to the loss so
                                a higher-entropy attention is preferred)
  - sinkhorn       weight 0.05  on retrieval-bank usage frequency

Listwise choice: we use a differentiable Pearson-IC surrogate (one minus
the within-day Pearson correlation between y_hat and y_cs), rather than
the original LambdaRank-NDCG formulation. The IC surrogate optimises
the headline metric directly and has stable gradients on the small
N_t cross-sections (N approximately 400 to 500). LambdaRank's pairwise
weights would also work here; the choice is documented per spec.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class LossWeights:
    """Loss component weights.

    Defaults are ranking-aware: Huber regression on z-scored returns,
    listwise IC surrogate, and pairwise margin (the three "primary"
    ranking objectives). Auxiliary heads (regime_ce, vol_mse, entropy,
    sinkhorn) default to zero and must be opted into via
    ``loss_weights_for("full")``.

    The 2026-05-07 audit caught a regression where the v3 commit
    incorrectly defaulted listwise and pairwise to zero, which silently
    neutered any baseline call that used the bare LossWeights() default
    (MASTER, post-commit StockMixer F2 seeds 45, 46). Defaults are now
    restored.
    """

    huber: float = 1.0
    listwise: float = 0.5
    pairwise: float = 0.3
    regime_ce: float = 0.0
    vol_mse: float = 0.0
    entropy: float = 0.0
    sinkhorn: float = 0.0
    listmle: float = 0.0
    # Option C (2026-05-26): differentiable Sharpe surrogate weight.
    # Default 0.0 keeps every existing preset byte-identical. The
    # 'diff_sharpe' preset and the run_stage2_finetune wiring set this
    # to 0.2 (small additive regulariser, NOT a replacement for the
    # ranking-aware Huber + listwise + pairwise core).
    diff_sharpe: float = 0.0


def loss_weights_for(config: str) -> LossWeights:
    """Return preset LossWeights for ``config in {'minimal', 'ranking', 'full'}``.

    minimal: Huber 1.0 only (regression on z-scored returns; not
             ranking-aware). Used to A/B-test the ranking-loss
             contribution; not recommended as a baseline loss.
    ranking: Huber 1.0 + listwise IC 0.5 + pairwise margin 0.3.
             No auxiliary heads (regime_ce, vol_mse, entropy, sinkhorn
             all zero). Default for InVAR architecture comparisons
             against iTransformer.
    full:    v1 / v2 weights (huber 1, listwise 0.5, pairwise 0.3,
             regime_ce 0.1, vol_mse 0.1, entropy 0.01, sinkhorn 0.05).
    """
    if config == "minimal":
        return LossWeights(huber=1.0, listwise=0.0, pairwise=0.0,
                            regime_ce=0.0, vol_mse=0.0, entropy=0.0,
                            sinkhorn=0.0, listmle=0.0)
    if config == "ranking":
        return LossWeights()  # uses ranking-aware defaults
    if config == "full":
        return LossWeights(
            huber=1.0, listwise=0.5, pairwise=0.3,
            regime_ce=0.1, vol_mse=0.1, entropy=0.01, sinkhorn=0.05,
            listmle=0.0,
        )
    if config == "listmle":
        # F2 fundamental L1 listwise rank loss upgrade (2026-05-26).
        # Drops the Pearson-IC surrogate, swaps in the Plackett-Luce
        # listwise rank likelihood. Keeps Huber on the cross-section
        # (low weight) for numerical anchoring and the pairwise margin
        # as a complementary local-order regulariser.
        return LossWeights(
            huber=0.3, listwise=0.0, pairwise=0.3,
            regime_ce=0.0, vol_mse=0.0, entropy=0.0, sinkhorn=0.0,
            listmle=1.0,
        )
    if config == "diff_sharpe":
        # Option C (2026-05-26): differentiable Sharpe surrogate added
        # as a SMALL additive regulariser on top of the canonical
        # ranking-aware Huber + listwise IC + pairwise margin core.
        # Weight 0.2 by design (the F2 ListMLE failure showed that
        # replacing the hybrid core entirely breaks generalisation;
        # this preset ADDS a Sharpe-aligned ranking signal alongside).
        # Note: on the canonical clpretrain pipeline (which uses
        # cs_mse_loss instead of the legacy hybrid_loss), this preset
        # is consumed by run_stage2_finetune which adds the
        # differentiable Sharpe term to cs_loss; see the wiring in
        # src/baselines/train_invar_clpretrain_v2.py.
        return LossWeights(
            huber=1.0, listwise=0.5, pairwise=0.3,
            regime_ce=0.0, vol_mse=0.0, entropy=0.0, sinkhorn=0.0,
            listmle=0.0, diff_sharpe=0.2,
        )
    if config == "listmle_soft":
        # Option C compose (2026-05-26): softened ListMLE weight (0.5)
        # to combine with F3 cross-stock attention. The audit found that
        # F2 ListMLE at weight 1.0 induces a strong F2-vs-F5 trade-off
        # (pool +0.460 vs canonical +0.899; F2 +0.976 but F5 -0.800).
        # The compose hypothesis is that lowering listmle to 0.5 while
        # raising the anchoring regression weight to 0.7 preserves the
        # F2 stress-fold lift but recovers F5 momentum-tailwind. Huber
        # plays the cross-section regression anchor role on the legacy
        # pipeline; on the canonical clpretrain pipeline, cs_mse_weight
        # plays the same role (set via cfg.cs_mse_weight).
        return LossWeights(
            huber=0.7, listwise=0.0, pairwise=0.3,
            regime_ce=0.0, vol_mse=0.0, entropy=0.0, sinkhorn=0.0,
            listmle=0.5,
        )
    raise ValueError(f"unknown loss config: {config!r}")


def huber_loss(y_hat: Tensor, y_cs: Tensor, mask: Tensor,
                delta: float = 1.0) -> Tensor:
    """Standard Huber loss masked to the active subset."""
    if not mask.any():
        return y_hat.sum() * 0.0
    y_hat_a = y_hat[mask]
    y_cs_a = y_cs[mask]
    return F.huber_loss(y_hat_a, y_cs_a, delta=delta, reduction="mean")


def listwise_ic_loss(y_hat: Tensor, y_cs: Tensor, mask: Tensor,
                       eps: float = 1e-8) -> Tensor:
    """Differentiable Pearson-IC surrogate: 1 - corr(y_hat, y_cs).

    Computed on the active subset only. This is the listwise choice
    documented in the module docstring; the pairwise margin loss covers
    the LambdaRank-flavored objective separately.
    """
    if not mask.any():
        return y_hat.sum() * 0.0
    a = y_hat[mask]
    b = y_cs[mask]
    a = a - a.mean()
    b = b - b.mean()
    num = (a * b).sum()
    denom = torch.sqrt((a ** 2).sum() * (b ** 2).sum() + eps)
    rho = num / denom
    return 1.0 - rho


def listmle_loss(y_hat: Tensor, y_cs: Tensor, mask: Tensor,
                  eps: float = 1e-8) -> Tensor:
    """Plackett-Luce listwise rank likelihood (ListMLE).

    Reference: Xia et al. 2008, "Listwise Approach to Learning to Rank:
    Theory and Algorithm" (ICML). The Plackett-Luce model places the
    probability of an observed permutation pi over K active items at::

        P(pi | y_hat) = prod_{k=1..K} exp(y_hat[pi(k)])
                         / sum_{j>=k} exp(y_hat[pi(j)])

    where the permutation pi is the descending sort of the targets.
    The negative log-likelihood, averaged over rank steps, is::

        L = mean_k (logsumexp(y_hat[pi(k:K)]) - y_hat[pi(k)])

    Lower is better. Implemented with ``torch.logcumsumexp`` on the
    flipped (ascending) order so the rightward cumulative log-sum-exp
    is numerically stable. Masked to the active subset (mirrors the
    existing ``huber_loss`` / ``listwise_ic_loss`` masking convention).
    The temperature is fixed at 1.0; an optional scale on ``y_hat``
    can be implemented at the call site if a sweep is needed.
    """
    if not mask.any():
        return y_hat.sum() * 0.0
    a = y_hat[mask]
    b = y_cs[mask]
    if a.numel() < 2:
        return y_hat.sum() * 0.0
    # Upcast to float32: torch.logcumsumexp has no fp16 backward and
    # the canonical InVAR finetune runs under autocast(fp16). The cost
    # is a single per-day (N,) tensor cast.
    orig_dtype = a.dtype
    a32 = a.float()
    b32 = b.float()
    # Sort by descending target -> observed Plackett-Luce permutation.
    sorted_idx = torch.argsort(b32, descending=True)
    a_sorted = a32[sorted_idx]
    # Cumulative logsumexp from the END: cum_lse[k] = logsumexp(a_sorted[k:])
    a_flipped = a_sorted.flip(0)
    cum_lse_flipped = torch.logcumsumexp(a_flipped, dim=0)
    cum_lse = cum_lse_flipped.flip(0)
    # Per-rank NLL contribution; mean for scale stability across
    # variable-K cross-sections.
    out = (cum_lse - a_sorted).mean()
    # Cast back to caller dtype so downstream sums stay homogeneous.
    return out.to(orig_dtype)


def soft_topk_relaxation(
    scores: Tensor,
    K: int,
    temperature: float = 0.1,
) -> Tensor:
    """Differentiable soft top-K relaxation over a 1-D score vector.

    Returns a soft mask of shape ``(N,)`` in [0, 1] that approximates
    the indicator of the top-K positions when ranked by descending
    ``scores``. Implementation uses a score-thresholded sigmoid:

        threshold = detached K-th-largest value of ``scores`` (the
                    decision boundary between top-K and bottom-(N-K)).
        soft_w_i  = sigmoid((scores_i - threshold) / temperature)

    The threshold is detached so the gradient flows only through the
    relative position of ``scores_i`` vs the threshold (not through
    the threshold definition itself). As ``temperature -> 0``, the
    sigmoid becomes a step function and ``soft_w`` converges to the
    hard top-K indicator. The gradient w.r.t. ``scores`` is strictly
    positive within the soft band around the threshold, which is
    exactly the "swing positions" where ranking matters for the L/S
    portfolio composition.

    After the sigmoid, weights are renormalised so they sum to ``K``,
    matching the hard top-K equal-weight portfolio mass and keeping
    the surrogate insensitive to N and temperature.

    This is the standard score-thresholded soft top-K used in soft
    ranking / soft k-NN literature (e.g., SoftSort variants); a
    simpler, fully-differentiable alternative to the Cuturi 2019
    Sinkhorn ranking formulation while preserving the same boundary
    behaviour.

    The function is a building block for ``differentiable_sharpe_loss``
    below; it is not exported as a loss component on its own.

    Args:
        scores: ``(N,)`` 1-D float tensor of per-asset scores.
        K: Top-K size; must satisfy ``1 <= K <= N``.
        temperature: Sigmoid temperature in score units. Smaller is
            sharper. Default 0.1; tune to the typical std of
            ``scores`` (InVAR L1 scores have std approximately 1 after
            cross-sectional standardisation).

    Returns:
        ``(N,)`` float tensor of soft weights, summing to ``K``.
    """
    if scores.dim() != 1:
        raise ValueError(
            f"soft_topk_relaxation expects 1-D scores; got {tuple(scores.shape)}"
        )
    N = int(scores.shape[0])
    if N == 0:
        return scores.new_zeros(0)
    K = max(1, min(int(K), N))
    tau = max(float(temperature), 1.0e-4)
    # K-th-largest as the decision boundary. Detached so the gradient
    # flows only through scores_i in the soft band, not through the
    # threshold definition.
    sorted_desc, _ = torch.sort(scores.detach(), descending=True)
    # Boundary halfway between the K-th and (K+1)-th value when K < N,
    # else just the K-th value. The halfway choice centres the soft
    # band on the rank-K boundary.
    if K < N:
        threshold = 0.5 * (sorted_desc[K - 1] + sorted_desc[K])
    else:
        threshold = sorted_desc[K - 1]
    soft_w = torch.sigmoid((scores - threshold) / tau)
    soft_w = soft_w / (soft_w.sum() + 1.0e-8) * float(K)
    return soft_w


def differentiable_sharpe_loss(
    scores_batch: list[Tensor],
    returns_batch: list[Tensor],
    mask_batch: list[Tensor],
    K: int,
    temperature: float = 0.1,
    eps: float = 1.0e-6,
) -> Tensor:
    """Differentiable surrogate Sharpe of the soft top-K L/S portfolio.

    For each day t in the batch:
        - long_w_t  = soft_topk_relaxation(scores_t, K, tau) over the
                      active subset.
        - short_w_t = soft_topk_relaxation(-scores_t, K, tau) over the
                      active subset.
        - pr_t = (long_w_t * returns_t).sum() / K
                 - (short_w_t * returns_t).sum() / K
    Then the batch-level Sharpe surrogate is::

        sharpe = mean(pr) / (std(pr) + eps)

    The loss returned is ``-sharpe`` (to be minimised). Sharpe needs
    multiple days for the std to be well-defined; the caller is
    responsible for batching at least ``diff_sharpe_batch_days``
    days before calling this function.

    Args:
        scores_batch: list of ``(N_t,)`` 1-D tensors, one per day.
        returns_batch: list of ``(N_t,)`` raw next-day return tensors
            (NOT z-scored; the Sharpe surrogate is a return-scale
            objective).
        mask_batch: list of ``(N_t,)`` bool tensors marking the active
            subset (loss-mask, mirrors the existing huber/listwise
            convention).
        K: Per-side top-K wrapper size (SP500 K=50, NDX K=20, NBI K=25).
        temperature: Soft-topk sigmoid temperature; default 0.1.
        eps: std denominator floor for numerical stability.

    Returns:
        Scalar tensor: ``-sharpe`` over the batch of days.
    """
    if len(scores_batch) == 0:
        # No days collected yet; return a zero with the device of the
        # caller's first parameter (caller passes empty only at the
        # very start; safe to anchor to the loss-graph elsewhere).
        return torch.zeros((), requires_grad=True)
    portfolio_returns = []
    for s, r, m in zip(scores_batch, returns_batch, mask_batch):
        if not m.any():
            continue
        s_a = s[m]
        r_a = r[m]
        if s_a.numel() < 2:
            continue
        K_eff = max(1, min(int(K), int(s_a.numel())))
        long_w = soft_topk_relaxation(s_a, K_eff, temperature)
        short_w = soft_topk_relaxation(-s_a, K_eff, temperature)
        pr = ((long_w * r_a).sum() - (short_w * r_a).sum()) / float(K_eff)
        portfolio_returns.append(pr)
    if len(portfolio_returns) < 2:
        # Single-day batch: cannot compute std. Return zero, propagates
        # no gradient, harmless (the cs_loss + ranking terms carry the
        # backward pass on this step).
        first = scores_batch[0]
        return first.sum() * 0.0
    pr_tensor = torch.stack(portfolio_returns)
    mean_pr = pr_tensor.mean()
    std_pr = pr_tensor.std(unbiased=False)
    sharpe = mean_pr / (std_pr + eps)
    return -sharpe


def pairwise_margin_loss(
    y_hat: Tensor, y_cs: Tensor, mask: Tensor,
    margin: float = 0.0, max_pairs: int = 4096,
) -> Tensor:
    """RSR-style pairwise margin loss within the day's active set.

    Pairs are sampled by the magnitude of the target return difference;
    the model is penalised whenever the predicted ordering disagrees
    with the target ordering by more than ``-margin``. Cap the number
    of pairs for tractability on N approximately 500 cross-sections.
    """
    if not mask.any():
        return y_hat.sum() * 0.0
    a_idx = mask.nonzero(as_tuple=True)[0]
    n = a_idx.numel()
    if n < 2:
        return y_hat.sum() * 0.0
    n_pairs = min(max_pairs, n * (n - 1) // 2)
    rng = torch.randint(low=0, high=n, size=(n_pairs * 2,), device=y_hat.device)
    i = a_idx[rng[: n_pairs]]
    j = a_idx[rng[n_pairs:]]
    keep = (i != j)
    if not keep.any():
        return y_hat.sum() * 0.0
    i = i[keep]; j = j[keep]
    diff_target = y_cs[i] - y_cs[j]
    diff_pred = y_hat[i] - y_hat[j]
    sign = torch.sign(diff_target)
    losses = F.relu(margin - sign * diff_pred)
    weight = diff_target.abs()
    return (losses * weight).sum() / (weight.sum() + 1e-8)


def regime_ce_loss(regime_logits: Tensor, regime_label: int) -> Tensor:
    """Cross-entropy of regime_logits against the day's GMM cluster id."""
    target = torch.tensor([regime_label], device=regime_logits.device,
                            dtype=torch.long)
    return F.cross_entropy(regime_logits.unsqueeze(0), target)


def vol_mse_loss(vol_hat: Tensor, vol_target: Tensor, mask: Tensor) -> Tensor:
    """MSE on per-ticker 20-day forward realised vol, masked by has_fwd_vol."""
    if not mask.any():
        return vol_hat.sum() * 0.0
    a = vol_hat[mask]
    b = vol_target[mask]
    return F.mse_loss(a, b, reduction="mean")


def regime_attn_entropy(attn_weights: list[dict] | None) -> Tensor:
    """Negative mean entropy of regime cross-attention weights across blocks.

    A higher-entropy attention distribution is preferred (more diverse
    use of regime tokens), so we return ``-entropy`` and let the trainer
    add it with a positive weight (which becomes a soft pull toward
    high entropy).
    """
    if not attn_weights:
        return torch.zeros((), requires_grad=True)
    entropies = []
    for blk in attn_weights:
        if blk is None:
            continue
        ca = blk.get("ca")
        if ca is None:
            continue
        # ca shape: (batch, num_heads, query_len, key_len) or simpler
        # depending on need_weights setting; flatten over all but last.
        p = ca.float()
        p = p.reshape(-1, p.shape[-1])
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-9)
        ent = -(p * (p + 1e-9).log()).sum(dim=-1).mean()
        entropies.append(ent)
    if not entropies:
        first = next(
            (b["ca"] for b in attn_weights if b is not None and b.get("ca") is not None),
            None,
        )
        device = first.device if first is not None else "cpu"
        return torch.zeros((), device=device, requires_grad=True)
    return -torch.stack(entropies).mean()


def sinkhorn_balance_loss(
    usage_counts: Tensor, target: Tensor | None = None, eps: float = 0.05,
    n_iter: int = 5,
) -> Tensor:
    """Soft Sinkhorn-style balance penalty on bank-entry usage frequency.

    Args:
        usage_counts: ``(bank_size,)`` tensor of (soft) usage scores; we
            convert to a probability distribution and penalise its
            divergence from uniform via a Sinkhorn-regularised objective.
        target: optional target distribution; defaults to uniform.
        eps: Sinkhorn regulariser.
        n_iter: number of Sinkhorn iterations (currently unused; we use a
            cheap KL-to-uniform proxy that the spec calls "Sinkhorn-style".
            The proper OT-balanced formulation is an upgrade target.)
    """
    if usage_counts.numel() == 0:
        return torch.zeros((), requires_grad=True)
    p = usage_counts.float()
    p = p / (p.sum() + 1e-9)
    if target is None:
        target = torch.full_like(p, 1.0 / p.numel())
    return F.kl_div((p + 1e-9).log(), target, reduction="sum")


@dataclass
class LossOutput:
    total: Tensor
    components: dict[str, float]


def hybrid_loss(
    y_hat: Tensor,
    y_cs: Tensor,
    mask: Tensor,
    regime_logits: Tensor,
    regime_label: int,
    vol_hat: Tensor,
    vol_target: Tensor,
    has_vol_mask: Tensor,
    weights: LossWeights | None = None,
    attn_weights: list[dict] | None = None,
    bank_usage_counts: Tensor | None = None,
    diff_sharpe_term: Tensor | None = None,
) -> LossOutput:
    """Combine the per-component losses with weights.

    ``diff_sharpe_term`` is an optional pre-computed scalar from
    :func:`differentiable_sharpe_loss`. It is OPTIONAL because Sharpe
    needs a batch of days (this loss is single-day), so the trainer
    typically computes it externally on an accumulating buffer and
    passes it in here. When None, the diff_sharpe contribution is
    zero and the canonical multi-day-free hybrid loss path is
    byte-identical.
    """
    w = weights or LossWeights()
    h = huber_loss(y_hat, y_cs, mask)
    lst = listwise_ic_loss(y_hat, y_cs, mask)
    pw = pairwise_margin_loss(y_hat, y_cs, mask)
    rce = regime_ce_loss(regime_logits, regime_label)
    vmse = vol_mse_loss(vol_hat, vol_target, has_vol_mask)
    ent = regime_attn_entropy(attn_weights)
    snk = (sinkhorn_balance_loss(bank_usage_counts)
            if bank_usage_counts is not None else torch.zeros((), device=h.device))
    lmle = (listmle_loss(y_hat, y_cs, mask)
             if w.listmle > 0 else y_hat.sum() * 0.0)
    dsh = (diff_sharpe_term if diff_sharpe_term is not None
            else y_hat.sum() * 0.0)

    total = (
        w.huber * h
        + w.listwise * lst
        + w.pairwise * pw
        + w.regime_ce * rce
        + w.vol_mse * vmse
        + w.entropy * ent
        + w.sinkhorn * snk
        + w.listmle * lmle
        + w.diff_sharpe * dsh
    )
    return LossOutput(
        total=total,
        components={
            "huber": float(h.detach().item()),
            "listwise": float(lst.detach().item()),
            "pairwise": float(pw.detach().item()),
            "regime_ce": float(rce.detach().item()),
            "vol_mse": float(vmse.detach().item()),
            "entropy": float(ent.detach().item()) if ent.requires_grad or ent.is_floating_point() else 0.0,
            "sinkhorn": float(snk.detach().item()),
            "listmle": float(lmle.detach().item()),
            "diff_sharpe": float(dsh.detach().item()),
        },
    )


__all__ = [
    "LossWeights", "LossOutput", "loss_weights_for",
    "huber_loss", "listwise_ic_loss", "listmle_loss",
    "pairwise_margin_loss",
    "regime_ce_loss", "vol_mse_loss", "regime_attn_entropy",
    "sinkhorn_balance_loss", "hybrid_loss",
    "soft_topk_relaxation", "differentiable_sharpe_loss",
]
