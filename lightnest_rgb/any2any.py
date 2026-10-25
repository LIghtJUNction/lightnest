"""Any2Any artifact model primitives shared by generation and editing.

The artifact is a typed byte stream. Format-specific validators stay at the
boundary; the neural core receives bytes, kind, metadata and observable
feedback rather than an ELF-specific assumption.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from .context import NeuralNotebook

KINDS = ("raw", "text", "json", "csv", "elf", "pe", "macho", "archive", "compressed", "image", "audio", "pdf", "sqlite", "bytecode")
DEFAULT_RED_DEPTHS = (2, 4, 8, 16)
DEFAULT_MAX_BYTES = 1 << 20
DEFAULT_SEQUENCE_LIMIT = 4096

@dataclass(frozen=True)
class CopyPasteAction:
    source_start: int
    source_end: int
    target_start: int
    operation: str = "copy_replace"

    def __post_init__(self):
        if self.operation not in {"copy_replace","copy_insert"} or min(self.source_start,self.source_end,self.target_start) < 0 or self.source_end <= self.source_start:
            raise ValueError("invalid copy/paste action")


def apply_copy_paste(raw: bytes, action: CopyPasteAction, expected_sha256: str) -> bytes:
    if hashlib.sha256(raw).hexdigest() != expected_sha256: raise ValueError("artifact identity changed")
    if action.source_end > len(raw) or action.target_start > len(raw): raise ValueError("copy/paste range out of bounds")
    payload=raw[action.source_start:action.source_end]
    if action.operation == "copy_insert": return raw[:action.target_start]+payload+raw[action.target_start:]
    if action.target_start + len(payload) > len(raw): raise ValueError("replacement range out of bounds")
    result=bytearray(raw); result[action.target_start:action.target_start+len(payload)]=payload; return bytes(result)


def kind_index(kind: str) -> int:
    if not isinstance(kind, str) or not 1 <= len(kind) <= 128:
        raise ValueError("artifact kind must be a short nonempty string")
    if kind in KINDS: return KINDS.index(kind)
    return 32 + int.from_bytes(hashlib.sha256(kind.encode()).digest()[:2], "little") % 224


@dataclass(frozen=True)
class ArtifactRequest:
    kind: str
    operation: str
    specification: str = ""

    def __post_init__(self):
        if (not isinstance(self.kind, str) or not 1 <= len(self.kind) <= 128 or not self.operation
                or len(self.operation.encode()) > 512 or not isinstance(self.specification, str)
                or len(self.specification.encode()) > 2048):
            raise ValueError("invalid Any2Any artifact request")


class ArtifactCore(nn.Module):
    """Format-agnostic shared encoder with green observation and red/blue heads."""
    def __init__(self, width=128, heads=4, max_bytes=DEFAULT_MAX_BYTES, red_depths=DEFAULT_RED_DEPTHS,
                 sequence_limit=DEFAULT_SEQUENCE_LIMIT, note_slots=32, request_encoding="gru",
                 semantic_chunk_size=0, semantic_bpe=None):
        super().__init__()
        if width % heads or width < 32 or max_bytes < 256 or sequence_limit < 256: raise ValueError("invalid Any2Any geometry")
        if not red_depths or tuple(sorted(set(red_depths))) != tuple(red_depths) or red_depths[0] <= 1:
            raise ValueError("invalid Any2Any thinking schedule")
        self.width, self.max_bytes, self.sequence_limit = width, max_bytes, sequence_limit
        if request_encoding not in {"mean", "gru"}:
            raise ValueError("unknown request encoding")
        self.request_encoding = request_encoding
        if type(semantic_chunk_size) is not int or semantic_chunk_size not in (0, 4, 8, 16):
            raise ValueError("invalid semantic chunk size")
        self.semantic_chunk_size = semantic_chunk_size
        self.semantic_bpe = semantic_bpe
        if semantic_bpe is not None and semantic_chunk_size:
            raise ValueError("choose byte chunks or lossless BPE")
        self.blue_depth, self.red_depths = 1, tuple(red_depths)
        self.byte = nn.Embedding(257, width)
        self.kind = nn.Embedding(256, width)
        self.position = nn.Embedding(sequence_limit, width)
        if request_encoding == "gru":
            self.request_encoder = nn.GRU(width, width, batch_first=True)
        if semantic_chunk_size:
            self.semantic_chunk = nn.Sequential(nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
                                                nn.Conv1d(width, width, 5, padding=2), nn.GELU())
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(width, heads, 4 * width, batch_first=True, norm_first=True), 4)
        self.red = nn.GRUCell(width, width)
        self.abstract = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.LayerNorm(4 * width))
        self.to_blue = nn.Linear(4 * width, width)
        self.kind_head = nn.Linear(width, 256)
        self.summary_head = nn.Linear(width, 5)
        self.budget = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.LayerNorm(width),
                                    nn.Linear(width, len(self.red_depths) + 1))
        self.notebook = NeuralNotebook(width, note_slots)
        self.copy_query = nn.Linear(width, width, bias=False)
        self.blue_byte = nn.Linear(width, 257)
        self.red_byte = nn.Linear(width, 257)
        self.stop = nn.Linear(width, 1)
        if semantic_bpe is not None:
            from .byte_tokenizer import ByteBPE
            self.semantic_tokenizer = ByteBPE(semantic_bpe)
            vocabulary = self.semantic_tokenizer.vocab_size + 2
            self.semantic_token = nn.Embedding(vocabulary, width)
            self.semantic_pool = nn.Linear(width, 1)
            self.semantic_absolute_position = nn.Linear(8, width, bias=False)
            self.semantic_norm = nn.LayerNorm(width)
            self.semantic_context = nn.Linear(width, width)
            nn.init.zeros_(self.semantic_context.weight)
            nn.init.constant_(self.semantic_context.bias, -4.)
            self.language = nn.GRU(width, width, 2, batch_first=True)
            self.language_attention = nn.MultiheadAttention(width, heads, batch_first=True)
            self.language_norm = nn.LayerNorm(width)
            self.language_copy = nn.Linear(width, 1)
            nn.init.constant_(self.language_copy.bias, -2.)

    def semantic_tokens(self, artifacts):
        if self.semantic_bpe is None:
            raise ValueError("token-level semantics requires a BPE checkpoint")
        if not artifacts or any(not 1 <= len(raw) <= self.max_bytes for raw in artifacts):
            raise ValueError("artifact exceeds semantic byte capacity")
        ids = [self.semantic_tokenizer.encode(raw) for raw in artifacts]
        device = self.byte.weight.device
        tokens = torch.zeros(len(ids), max(map(len, ids)), dtype=torch.long, device=device)
        byte_offsets = torch.zeros_like(tokens, dtype=self.byte.weight.dtype)
        lengths = torch.tensor([len(row) for row in ids], device=device)
        for index, row in enumerate(ids):
            tokens[index, :len(row)] = torch.tensor(row, device=device)
            offset = 0; starts = []
            for token_id in row:
                starts.append(offset); offset += len(self.semantic_tokenizer.pieces[token_id])
            byte_offsets[index, :len(row)] = torch.tensor(starts, device=device)
        positions = torch.arange(tokens.shape[1], device=device)[None]
        valid = positions < lengths[:, None]
        requests = [ArtifactRequest("raw", "represent content")] * len(ids)
        scales = byte_offsets.new_tensor((1., 64., 4096., 262144.))
        angle = byte_offsets[..., None] / scales
        absolute = self.semantic_absolute_position(torch.cat((angle.sin(), angle.cos()), -1))
        inputs = (self.semantic_token(tokens) + self.position(positions % self.sequence_limit)
                  + absolute + self.encode_requests(requests)[:, None])
        chunks = []
        for start in range(0, tokens.shape[1], self.sequence_limit):
            end = min(start + self.sequence_limit, tokens.shape[1])
            active = (lengths > start).nonzero().flatten()
            chunk = self.encoder(inputs[active, start:end], src_key_padding_mask=~valid[active, start:end])
            full = inputs.new_zeros(len(ids), end - start, self.width).index_copy(0, active, chunk)
            chunks.append(full)
        encoded = torch.cat(chunks, 1)
        pooled = (encoded * valid[..., None]).sum(1) / lengths[:, None]
        notes = self.notebook.compress(encoded, pooled, hard=not self.training, valid=valid)
        encoded = encoded + self.semantic_context(pooled).sigmoid()[:, None] * notes['context'][:, None]
        pooled = (encoded * valid[..., None]).sum(1) / lengths[:, None]
        _, guidance = self.think(pooled[:, None], depth=4)
        features = self.semantic_norm(encoded + guidance[:, None])
        weights = self.semantic_pool(features).squeeze(-1).masked_fill(~valid, -torch.inf).softmax(-1)
        return {"features": features, "valid": valid, "weights": weights, "tokens": tokens,
                "context_budget": notes['budget_logits'].argmax(-1)}

    def language_forward(self, memory, previous, hidden=None):
        if hidden is None:
            pooled = (memory["features"] * memory["weights"][..., None]).sum(1)
            hidden = pooled[None].expand(2, -1, -1).contiguous()
        decoded, hidden = self.language(self.semantic_token(previous), hidden)
        context, attention = self.language_attention(decoded, memory["features"], memory["features"],
                                                     key_padding_mask=~memory["valid"])
        value = self.language_norm(decoded + context)
        vocabulary = F.linear(value, self.semantic_token.weight).div(self.width**.5).softmax(-1)
        copied = torch.zeros_like(vocabulary)
        indices = memory["tokens"][:, None].expand(-1, previous.shape[1], -1)
        copied.scatter_add_(2, indices, attention)
        gate = self.language_copy(value).sigmoid()
        return ((1 - gate) * vocabulary + gate * copied).clamp_min(1e-9).log(), hidden

    @torch.inference_mode()
    def explain_bytes(self, raw, max_tokens=96):
        if self.training:
            raise RuntimeError("eval() required")
        memory = self.semantic_tokens([raw])
        token = torch.tensor([[self.semantic_tokenizer.bos]], device=self.byte.weight.device)
        ids = []; hidden = None; stopped = False; invalid = False
        for _ in range(max_tokens):
            output, hidden = self.language_forward(memory, token, hidden)
            value = int(output[0, -1].argmax())
            if value == self.semantic_tokenizer.eos:
                stopped = True; break
            if value == self.semantic_tokenizer.bos:
                invalid = True; break
            ids.append(value); token = torch.tensor([[value]], device=token.device)
        result = self.semantic_tokenizer.decode(ids)
        try:
            text = result.decode('utf-8'); utf8_valid = True
        except UnicodeDecodeError:
            text = result.decode('utf-8', errors='replace'); utf8_valid = False
        return {"text": text, "raw_hex": result.hex(), "stopped": stopped,
                "utf8_valid": utf8_valid, "invalid_special_token": invalid, "tokens": len(ids),
                "context_budget": int(memory['context_budget'][0])}

    def encode_batch(self, artifacts, requests):
        """Encode complete local windows, masking padding in both modalities."""
        if not artifacts or len(artifacts) != len(requests):
            raise ValueError("artifact/request batch mismatch")
        if any(not 1 <= len(raw) <= min(self.max_bytes, self.sequence_limit) for raw in artifacts):
            raise ValueError("artifact exceeds local Any2Any capacity")
        device = self.byte.weight.device
        lengths = torch.tensor([len(raw) for raw in artifacts], device=device)
        values = torch.zeros(len(artifacts), int(lengths.max()), dtype=torch.long, device=device)
        for row, raw in enumerate(artifacts):
            values[row, :len(raw)] = torch.tensor(list(raw), device=device)
        positions = torch.arange(values.shape[1], device=device)[None]
        valid = positions < lengths[:, None]
        kinds = torch.tensor([kind_index(request.kind) for request in requests], device=device)
        context = self.encode_requests(requests)
        inputs = self.byte(values) + self.position(positions) + self.kind(kinds)[:, None] + context[:, None]
        encoded = self.encoder(inputs, src_key_padding_mask=~valid)
        return encoded, valid, kinds

    def encode_requests(self, requests):
        device = self.byte.weight.device
        request_bytes = [(request.operation + "\0" + request.specification).encode() for request in requests]
        request_tokens = torch.zeros(len(requests), max(map(len, request_bytes)), dtype=torch.long, device=device)
        for row, raw in enumerate(request_bytes):
            request_tokens[row, :len(raw)] = torch.tensor([value + 1 for value in raw], device=device)
        embeddings = self.byte(request_tokens)
        if self.request_encoding == "gru":
            packed = nn.utils.rnn.pack_padded_sequence(embeddings, [len(raw) for raw in request_bytes],
                                                      batch_first=True, enforce_sorted=False)
            _, hidden = self.request_encoder(packed)
            context = hidden[-1]
        else:
            request_valid = torch.arange(request_tokens.shape[1], device=device)[None] < torch.tensor(
                [len(raw) for raw in request_bytes], device=device)[:, None]
            context = (embeddings * request_valid[..., None]).sum(1) / request_valid.sum(1)[:, None]
        return context

    def encode(self, raw: bytes, request: ArtifactRequest):
        encoded, _, kinds = self.encode_batch([raw], [request])
        return encoded, kinds

    def semantic_embedding(self, artifacts):
        """Shared learned representation; no parser, filenames or target text."""
        if self.semantic_bpe is not None:
            state = self.semantic_tokens(artifacts)
            return F.normalize((state["features"] * state["weights"][..., None]).sum(1), dim=-1)
        requests = [ArtifactRequest("text", "represent content")] * len(artifacts)
        if not self.semantic_chunk_size:
            encoded, valid, _ = self.encode_batch(artifacts, requests)
        else:
            if not artifacts or any(not 1 <= len(raw) <= min(self.max_bytes, self.sequence_limit) for raw in artifacts):
                raise ValueError("artifact exceeds local semantic capacity")
            size = self.semantic_chunk_size
            device = self.byte.weight.device
            length = (max(map(len, artifacts)) + size - 1) // size * size
            values = torch.zeros(len(artifacts), length, dtype=torch.long, device=device)
            lengths = torch.tensor([len(raw) for raw in artifacts], device=device)
            for row, raw in enumerate(artifacts):
                values[row, :len(raw)] = torch.tensor(list(raw), device=device)
            byte_valid = torch.arange(length, device=device)[None] < lengths[:, None]
            local = self.byte(values).transpose(1, 2)
            for layer in self.semantic_chunk:
                local = layer(local) * byte_valid[:, None]
            local = local.reshape(len(artifacts), self.width, -1, size).sum(-1).transpose(1, 2)
            counts = byte_valid.reshape(len(artifacts), -1, size).sum(-1).clamp_min(1)
            local = local / counts[..., None]
            positions = torch.arange(0, length, size, device=device)[None]
            valid = positions < lengths[:, None]
            inputs = (local + self.position(positions)
                      + self.kind.weight[kind_index("text")] + self.encode_requests(requests)[:, None])
            encoded = self.encoder(inputs, src_key_padding_mask=~valid)
        pooled = (encoded * valid[..., None]).sum(1) / valid.sum(1)[:, None]
        _, guidance = self.think(pooled[:, None], depth=4)
        return F.normalize(pooled + guidance, dim=-1)

    def encode_chunks(self, raw: bytes, request: ArtifactRequest, chunk_size=256, max_chunks=16):
        if not 32 <= chunk_size <= self.sequence_limit or not raw or len(raw) > self.max_bytes: raise ValueError("invalid hierarchical artifact")
        starts=list(range(0, len(raw), chunk_size))
        if len(starts)>max_chunks:
            starts=[starts[int(i*(len(starts)-1)/(max_chunks-1))] for i in range(max_chunks)]
        chunks=[]
        for start in starts:
            encoded, _ = self.encode(raw[start:start+chunk_size], request)
            chunks.append(encoded.mean(1))
        return torch.stack(chunks, 1)

    def contextualize(self, encoded, *, hard=None):
        if hard is None:
            hard = not self.training
        notes = self.notebook.compress(encoded, hard=hard)
        return encoded + notes["context"][:, None], notes

    @torch.inference_mode()
    def summarize(self, raw: bytes, request: ArtifactRequest, depth=None, chunk_size=256):
        encoded=self.encode_chunks(raw, request, chunk_size)
        encoded, notes=self.contextualize(encoded)
        if depth is None:
            abstract,guidance,budget=self.think_budgeted(encoded)
            return abstract, self.heads(encoded, guidance[:, None], budget_logits=budget["logits"]) | {"budget": budget, "context": notes}
        abstract,guidance=self.think(encoded, depth)
        return abstract, self.heads(encoded, guidance[:, None])

    def think(self, encoded, depth=8):
        if type(depth) is not int or depth < 1:
            raise ValueError("thinking depth must be positive")
        pooled = encoded.mean(1); hidden = pooled
        for _ in range(depth): hidden = self.red(pooled, hidden)
        abstract = self.abstract(hidden)
        return abstract, self.to_blue(abstract)

    def think_budgeted(self, encoded):
        """Run blue, let its learned budget head choose red compute, then continue."""
        pooled = encoded.mean(1); hidden = pooled
        for _ in range(self.blue_depth): hidden = self.red(pooled, hidden)
        abstract = self.abstract(hidden); guidance = self.to_blue(abstract)
        budget_logits = self.budget(torch.cat((hidden, guidance), -1))
        index = int(budget_logits.argmax(-1).item())
        selected = None if index == 0 else self.red_depths[index - 1]
        if selected is not None:
            for _ in range(self.blue_depth, selected): hidden = self.red(pooled, hidden)
            abstract = self.abstract(hidden); guidance = self.to_blue(abstract)
        return abstract, guidance, {"index": index, "depth": selected, "logits": budget_logits}

    def heads(self, encoded, blue_guidance, budget_logits=None):
        guidance = blue_guidance if blue_guidance.ndim == 3 else blue_guidance[:, None]
        blue = self.blue_byte(encoded + guidance)
        red = self.red_byte(encoded + guidance)
        kind_context = guidance.mean(1)
        if budget_logits is None:
            budget_logits = self.budget(torch.cat((encoded.mean(1), kind_context), -1))
        return {"blue_logits": blue, "red_logits": red, "stop_logits": self.stop(encoded).squeeze(-1),
                "kind_logits": self.kind_head(kind_context), "summary": self.summary_head(kind_context),
                "budget_logits": budget_logits, "copy_paste": self.copy_paste_scores(encoded, kind_context)}

    def fingerprint(self, request: ArtifactRequest):
        return hashlib.sha256(f"{request.kind}\0{request.operation}\0{request.specification}".encode()).hexdigest()

    def copy_paste_scores(self, encoded, guidance):
        """Native pointer scores for source interval and destination offset."""
        tokens=encoded.shape[1]; query=self.copy_query(encoded + guidance[:,None]); scores=query @ query.transpose(-1,-2) / self.width**.5
        return {"source_start": scores.mean(1), "source_end": scores.mean(1), "target_start": scores.mean(1), "length": scores.mean(1)}

    @torch.inference_mode()
    def propose_copy_paste(self, raw: bytes, request: ArtifactRequest):
        encoded,_=self.encode(raw,request); encoded,notes=self.contextualize(encoded); abstract,guidance,budget=self.think_budgeted(encoded); scores=self.copy_paste_scores(encoded,guidance)
        matrix=scores["source_start"][0]; flat=matrix.flatten()
        for index in flat.argsort(descending=True).tolist():
            start=index//matrix.shape[-1]; target=index%matrix.shape[-1]
            length=min(16,len(raw)-start,len(raw)-target)
            if length > 0: break
        end=start+length
        return CopyPasteAction(start,end,target,"copy_replace"), {"abstract":abstract,"scores":scores,"budget":budget,"context":notes}


def semantic_similarity(left, right):
    """Learned token matching; padding never participates in either direction."""
    a, b = F.normalize(left['features'], dim=-1), F.normalize(right['features'], dim=-1)
    scores = torch.einsum('bnd,cmd->bcnm', a, b)
    valid = left['valid'][:, None, :, None] & right['valid'][None, :, None, :]
    scores = scores.masked_fill(~valid, -2.)
    forward = (scores.max(-1).values * left['weights'][:, None]).sum(-1)
    backward = (scores.max(-2).values * right['weights'][None]).sum(-1)
    return (forward + backward) / 2


def artifact_budget_target(candidates, target, kind_target=None, summary_target=None):
    """Choose a Pareto-style compute point from observed task losses.

    Each available task head contributes a rank, so byte scale cannot drown
    out kind or structure supervision.  The budget head learns this choice;
    this helper only constructs training labels from task outcomes.
    """
    if not candidates or target.ndim != 2 or target.shape[0] != 1:
        raise ValueError("invalid Any2Any budget supervision")
    metrics = [[] for _ in range(3)]
    for candidate in candidates:
        metrics[0].append(F.cross_entropy(candidate["red_logits"].reshape(-1, 257), target.reshape(-1)).detach().item())
        metrics[1].append(F.cross_entropy(candidate["kind_logits"], kind_target).detach().item() if kind_target is not None else 0.)
        metrics[2].append(F.mse_loss(candidate["summary"], summary_target).detach().item() if summary_target is not None else 0.)
    ranks = [[0] * len(candidates) for _ in metrics]
    for values, rank in zip(metrics, ranks, strict=True):
        for position, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
            rank[index] = position
    scores = [sum(rank[index] for rank in ranks) for index in range(len(candidates))]
    return min(range(len(scores)), key=lambda index: (scores[index], index))
