#!/usr/bin/env python3
"""Small, dependency-free Muon optimizer helpers for Tiny Leela experiments.

Muon is used here as an opt-in experimental optimizer for matrix-like weights.
Biases, norms, embeddings, and scalar/vector parameters should remain on AdamW.
This implementation intentionally favors readability and safety over maximum speed.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import torch


def _zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonal factor of a 2D matrix with Newton-Schulz steps."""
    if g.ndim != 2:
        raise ValueError(f"Muon orthogonalization expects a 2D tensor, got {tuple(g.shape)}")
    x = g.float()
    if x.numel() == 0:
        return g
    x = x / (x.norm() + eps)
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    # Coefficients commonly used by public Muon reference implementations.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(max(1, int(steps))):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x
    if transposed:
        x = x.T
    return x.to(dtype=g.dtype)


class Muon(torch.optim.Optimizer):
    """Experimental Muon optimizer for 2D/flattened matrix parameters.

    For Conv kernels, callers can pass the raw parameter; it is flattened to
    [out_channels, -1] for orthogonalized updates and reshaped back.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3,
                 momentum: float = 0.95, weight_decay: float = 0.0,
                 ns_steps: int = 5, nesterov: bool = True):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group['lr']); momentum = float(group['momentum'])
            wd = float(group['weight_decay']); ns_steps = int(group['ns_steps'])
            nesterov = bool(group['nesterov'])
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    raise ValueError('Muon should only receive matrix-like parameters; route vectors/scalars to AdamW')
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                shape = update.shape
                update2 = update.reshape(shape[0], -1)
                ortho = _zeropower_via_newtonschulz5(update2, steps=ns_steps).reshape(shape)
                # Decoupled weight decay. The sqrt fan ratio keeps update scale
                # sane for non-square matrices without trying to be exact.
                if wd:
                    p.mul_(1.0 - lr * wd)
                fan_ratio = max(1.0, shape[0] / max(1, update2.shape[1])) ** 0.5
                p.add_(ortho, alpha=-lr * fan_ratio)
        return loss


def split_muon_adamw_params(module: torch.nn.Module) -> Tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Return (matrix_like, other) params for Muon+AdamW hybrid training."""
    matrix_like = []
    other = []
    seen: set[int] = set()
    for name, p in module.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        lname = name.lower()
        # Keep embeddings, normalization params, explicit biases, and output/aux heads on AdamW.
        # For this first Tiny Leela experiment, Muon is applied to trunk/stem matrices only;
        # very wide policy/AV heads are both task-sensitive and expensive to orthogonalize.
        is_head = any(tag in lname for tag in ('policy', 'wdl', 'av.', 'rank.', 'regret.'))
        if p.ndim >= 2 and 'emb' not in lname and 'norm' not in lname and not lname.endswith('bias') and not is_head:
            matrix_like.append(p)
        else:
            other.append(p)
    return matrix_like, other


class MultiOptimizer:
    """Tiny adapter that behaves like one optimizer around several optimizers."""

    def __init__(self, *optimizers: torch.optim.Optimizer):
        self.optimizers = [o for o in optimizers if o is not None]
        self.param_groups = [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self, *args, **kwargs):
        for opt in self.optimizers:
            opt.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        out = None
        for opt in self.optimizers:
            out = opt.step(*args, **kwargs)
        return out

    def state_dict(self):
        return {'optimizers': [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state_dict):
        states = state_dict.get('optimizers') if isinstance(state_dict, dict) else None
        if not states:
            return
        for opt, state in zip(self.optimizers, states):
            opt.load_state_dict(state)
