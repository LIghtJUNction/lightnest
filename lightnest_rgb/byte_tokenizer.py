"""Lossless byte BPE with an alphabet that covers every possible artifact byte."""
from __future__ import annotations

import json

from tokenizers import Tokenizer, models, trainers


def byte_text(raw):
    return ''.join(chr(256 + value) for value in raw)


class ByteBPE:
    def __init__(self, serialized):
        config = json.loads(serialized)
        if (config['model']['type'] != 'BPE' or config.get('normalizer') is not None
                or config.get('pre_tokenizer') is not None or config.get('post_processor') is not None
                or config.get('truncation') is not None or config.get('padding') is not None
                or config.get('added_tokens') or config['model'].get('dropout') is not None):
            raise ValueError('byte BPE cannot normalize, truncate or inject tokens')
        self.serialized = serialized
        self.impl = Tokenizer.from_str(serialized)
        self.vocab_size = self.impl.get_vocab_size()
        self.pieces = []
        for index in range(self.vocab_size):
            try:
                piece = bytes(ord(char) - 256 for char in self.impl.id_to_token(index))
            except (ValueError, TypeError):
                raise ValueError('token outside the byte alphabet') from None
            if not piece:
                raise ValueError('empty byte token')
            self.pieces.append(piece)
        if any(self.impl.token_to_id(chr(256 + value)) is None for value in range(256)):
            raise ValueError('byte alphabet is incomplete')
        self.bos, self.eos = self.vocab_size, self.vocab_size + 1

    @classmethod
    def train(cls, values, *, vocab_size=2048):
        if not 256 <= vocab_size <= 16384:
            raise ValueError('invalid byte vocabulary size')
        tokenizer = Tokenizer(models.BPE())
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=4, max_token_length=24,
                                      initial_alphabet=[chr(256 + value) for value in range(256)],
                                      show_progress=False)
        tokenizer.train_from_iterator((byte_text(raw) for raw in values), trainer=trainer)
        return cls(tokenizer.to_str())

    def encode(self, raw):
        ids = self.impl.encode(byte_text(raw), add_special_tokens=False).ids
        if self.decode(ids) != raw:
            raise ValueError('byte tokenization changed the artifact')
        return ids

    def decode(self, ids):
        if any(type(index) is not int or not 0 <= index < self.vocab_size for index in ids):
            raise ValueError('invalid artifact token')
        return b''.join(self.pieces[index] for index in ids)
