"""Zero-retrain decoding upgrades for linear-chain CRF models (v1 & v2).

Provides:
- `forward_backward_marginals`: Exact log-space Forward-Backward computing log P(y_t=i | x).
- `mbr_decode`: Minimum Bayes Risk / Posterior Marginal decoding maximizing expected token accuracy.
- `constrained_viterbi`: Viterbi with hard clamping (`locked_mask`) for user-specified words.
"""
from __future__ import annotations

import torch

NEG = -1e9


@torch.no_grad()
def forward_backward_marginals(
    emit: torch.Tensor,
    trans: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact posterior log-marginals for a linear-chain CRF.

    Args:
        emit: (B, L, K) local emission potentials (masked slots <= -1e8).
        trans: (B, L - 1, K, K) transition potentials from t to t+1.
        mask: (B, L, K) boolean tensor of valid candidate slots at real tokens.

    Returns:
        log_marginals: (B, L, K) log P(y_t = i | x), with invalid slots set to NEG.
        log_z: (B,) log partition function.
    """
    B, L, K = emit.shape
    emit_m = emit.masked_fill(~mask, NEG)
    real_pos = mask.any(dim=-1)  # (B, L)

    # Forward pass: alpha[b, t, k] = log sum over prefixes ending at k at step t
    alpha_list = []
    alpha = emit_m[:, 0]
    alpha_list.append(alpha)
    for t in range(1, L):
        nxt = torch.logsumexp(alpha.unsqueeze(2) + trans[:, t - 1], dim=1) + emit_m[:, t]
        real_t = real_pos[:, t].unsqueeze(1)
        alpha = torch.where(real_t, nxt, alpha)
        alpha_list.append(alpha)
    log_z = torch.logsumexp(alpha, dim=1)  # (B,)

    # Backward pass: beta[b, t, i] = log sum over suffixes starting after step t from i
    beta_list = [None] * L
    beta = torch.zeros(B, K, dtype=emit.dtype, device=emit.device)
    beta_list[L - 1] = beta
    for t in range(L - 2, -1, -1):
        # msg[b, i] = log sum_j (trans[b, t, i, j] + emit_m[b, t+1, j] + beta[b, j])
        nxt_beta = torch.logsumexp(
            trans[:, t] + (emit_m[:, t + 1] + beta).unsqueeze(1), dim=2
        )
        # Only step backward across transitions into a real position t+1
        real_next = real_pos[:, t + 1].unsqueeze(1)
        beta = torch.where(real_next, nxt_beta, beta)
        beta_list[t] = beta

    alpha_all = torch.stack(alpha_list, dim=1)  # (B, L, K)
    beta_all = torch.stack(beta_list, dim=1)    # (B, L, K)
    log_marg = alpha_all + beta_all - log_z.view(B, 1, 1)
    log_marg = log_marg.masked_fill(~mask, NEG)
    return log_marg, log_z


@torch.no_grad()
def constrained_viterbi(
    emit: torch.Tensor,
    trans: torch.Tensor,
    mask: torch.Tensor,
    locked_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact Viterbi decoding with optional user-constraint clamping.

    `locked_mask`: optional boolean (B, L, K). Where `locked_mask[b, t].any()` is True,
    only slots with `locked_mask[b, t, k] == True` are permitted.
    """
    eff_mask = mask if locked_mask is None else _apply_lock(mask, locked_mask)
    eff_emit = emit.masked_fill(~eff_mask, NEG)
    B, L, K = eff_emit.shape
    score = eff_emit[:, 0].clone()
    ptr = torch.zeros(B, L, K, dtype=torch.long, device=eff_emit.device)
    real_pos = eff_mask.any(dim=-1)
    for t in range(1, L):
        best, idx = (score.unsqueeze(2) + trans[:, t - 1]).max(dim=1)
        real_t = real_pos[:, t].unsqueeze(1)
        score = torch.where(real_t, best + eff_emit[:, t], score)
        ptr[:, t] = idx
    cur = score.argmax(1)
    out = torch.zeros(B, L, dtype=torch.long, device=eff_emit.device)
    for t in range(L - 1, -1, -1):
        w = real_pos[:, t]
        out[:, t] = torch.where(w, cur, out[:, t])
        cur = torch.where(w, ptr[:, t].gather(1, cur.unsqueeze(1)).squeeze(1), cur)
    return out


@torch.no_grad()
def mbr_decode(
    emit: torch.Tensor,
    trans: torch.Tensor,
    mask: torch.Tensor,
    locked_mask: torch.Tensor | None = None,
    marginal_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimum Bayes Risk / Posterior Marginal decoding.

    Args:
        marginal_weight:
            1.0 -> pure argmax_i log P(y_t = i | x) (Bayes optimal for token accuracy).
            0.0 < w < 1.0 -> Marginal-informed Viterbi blending local marginals and transitions
                             to retain bigram smoothness while optimizing expected token accuracy.
    Returns:
        preds: (B, L) chosen candidate index per position.
        log_marginals: (B, L, K) posterior log-probabilities.
    """
    eff_mask = mask if locked_mask is None else _apply_lock(mask, locked_mask)
    eff_emit = emit.masked_fill(~eff_mask, NEG)
    log_marg, _ = forward_backward_marginals(eff_emit, trans, eff_mask)
    if marginal_weight >= 1.0 - 1e-6:
        preds = log_marg.argmax(dim=-1)
        preds = torch.where(eff_mask.any(dim=-1), preds, torch.zeros_like(preds))
        return preds, log_marg
    blended_emit = (1.0 - marginal_weight) * eff_emit + marginal_weight * log_marg
    preds = constrained_viterbi(blended_emit, trans, eff_mask, locked_mask=None)
    return preds, log_marg


def _apply_lock(mask: torch.Tensor, locked_mask: torch.Tensor) -> torch.Tensor:
    has_lock = locked_mask.any(dim=-1, keepdim=True)
    combined = mask & locked_mask
    # Guard against empty intersection (safety fallback to original mask)
    valid_lock = has_lock & combined.any(dim=-1, keepdim=True)
    return torch.where(valid_lock, combined, mask)
