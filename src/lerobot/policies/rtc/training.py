# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training-time action-prefix conditioning shared by pi05 and FineART-VLA.

Actions must already be in normalized model coordinates. For relative EE policies,
the entire chunk (including its clean prefix) uses the current observation's EE
frame. Inference reanchors canonical leftovers to that same frame before inpainting.
"""

import torch
from torch import Tensor


def sample_training_rtc_prefix_mask(
    batch_size: int,
    action_horizon: int,
    max_delay: int,
    device: torch.device,
) -> Tensor | None:
    """Sample an inclusive delay in ``[0, max_delay]`` independently per flow draw.

    Returns ``None`` for a zero ``max_delay`` without drawing random numbers, so callers
    keep scalar flow time. The policy config guarantees ``max_delay < action_horizon``,
    which leaves at least one action to predict after every sampled prefix.
    """
    if max_delay == 0:
        return None
    delays = torch.randint(0, max_delay + 1, (batch_size,), device=device)
    positions = torch.arange(action_horizon, device=device)
    return positions.unsqueeze(0) < delays.unsqueeze(1)


def reduce_training_rtc_loss(
    losses: Tensor,
    prefix_mask: Tensor | None,
    reduction: str,
) -> Tensor:
    """Average flow loss over predicted postfix actions, excluding the clean prefix.

    ``none`` returns one mean per flow draw, for routing and repeat averaging in
    FineART-VLA. ``mean`` averages over every unconditioned action value, as pi05 does.
    Callers remove padded action dimensions before applying this reduction.
    """
    if reduction not in {"mean", "none"}:
        raise ValueError(f"Unsupported loss reduction: {reduction!r}")
    if prefix_mask is None:
        return losses.mean() if reduction == "mean" else losses.mean(dim=(1, 2))

    postfix_mask = (~prefix_mask).unsqueeze(-1).expand_as(losses)
    if reduction == "none":
        numerator = (losses * postfix_mask).sum(dim=(1, 2))
        denominator = postfix_mask.sum(dim=(1, 2))
        return numerator / denominator.clamp(min=1)
    return (losses * postfix_mask).sum() / postfix_mask.sum().clamp(min=1)
