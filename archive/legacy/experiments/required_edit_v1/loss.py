"""Training-only alternatives for required edits; inference is unchanged.

set_mass rewards any valid next edit. coverage distributes supervision across
all remaining required edits. weakest tests the more aggressive bottleneck
hypothesis. None of these treats unknown or alternative patches as required.
The caller must supply complete, compatible, remaining-edit labels.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F

VARIANTS = ("set_mass", "coverage", "weakest")


def required_edit_loss(scores, bits, rows, none_index, variant="set_mass"):
    """Keep byte supervision and STOP unchanged; alter position aggregation only."""
    if variant not in VARIANTS:
        raise ValueError("unknown supervision rule")
    if scores.ndim != 2 or scores.shape != (len(rows), none_index + 1) or not rows:
        raise ValueError("nonempty score/label batches must align")
    if bits.shape != (len(rows), none_index, 8):
        raise ValueError("byte logits must have eight bits per position")
    allowed = torch.zeros_like(scores, dtype=torch.bool)
    batch_ids, offsets, values = [], [], []
    for j, row in enumerate(rows):
        edits = row["edits"]
        if not edits:
            allowed[j, none_index] = True
        seen = set()
        for edit in edits:
            p, v = edit["offset"], edit["value"]
            if type(p) is not int or not 0 <= p < none_index or p in seen:
                raise ValueError("invalid or duplicate edit position")
            if type(v) is not int or not 0 <= v < 256:
                raise ValueError("invalid replacement byte")
            seen.add(p)
            allowed[j, p] = True
            batch_ids.append(j)
            offsets.append(p)
            values.append([(v >> k) & 1 for k in range(8)])
    logp = scores.log_softmax(-1)
    if variant == "set_mass":
        per_row = -logp.masked_fill(~allowed, -torch.inf).logsumexp(-1)
    elif variant == "coverage":
        per_row = -logp.masked_fill(~allowed, 0).sum(-1) / allowed.sum(-1)
    else:
        per_row = -logp.masked_fill(~allowed, torch.inf).min(-1).values
    position_loss = per_row.mean()
    if values:
        targets = torch.tensor(values, dtype=bits.dtype, device=bits.device)
        byte_loss = F.binary_cross_entropy_with_logits(bits[batch_ids, offsets], targets)
    else:
        byte_loss = bits.sum() * 0
    return position_loss + 2 * byte_loss, position_loss, byte_loss
