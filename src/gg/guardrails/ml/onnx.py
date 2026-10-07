"""tokenizers + onnxruntime sequence classifiers; imported only when a model loads (needs the `ml` extra)"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from tokenizers import Tokenizer

from gg.guardrails.ml.artefacts import FileRole, ModelSpec

type FloatArray = npt.NDArray[np.float32]

BATCH = 8
_WARMUP_TOKENS = (16, 128, 512)


def session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    # one thread per run: parallelism comes from the shared cpu executor, never from ort itself
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def special_frame(tokenizer: Tokenizer) -> tuple[list[int], list[int]]:
    """special ids the post-processor puts around one sequence ([CLS] .. [SEP], <s> .. </s>)"""
    enc = tokenizer.encode("a")
    content = [i for i, special in enumerate(enc.special_tokens_mask) if not special]
    first, last = content[0], content[-1]
    return enc.ids[:first], enc.ids[last + 1 :]


def softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def feed(
    sess: ort.InferenceSession, rows: Sequence[Sequence[int]], pad_id: int
) -> dict[str, npt.NDArray[np.int64]]:
    """right-padded int64 inputs for whichever of input_ids/attention_mask/token_type_ids the graph takes"""
    width = max(len(r) for r in rows)
    ids = np.full((len(rows), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=np.int64)
    for i, row in enumerate(rows):
        ids[i, : len(row)] = row
        mask[i, : len(row)] = 1
    available = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
    return {i.name: available[i.name] for i in sess.get_inputs()}


class OnnxTextClassifier:
    """probability of the positive class; long text is split into overlapping windows, max over windows"""

    def __init__(
        self,
        model_id: str,
        sess: ort.InferenceSession,
        tokenizer: Tokenizer,
        *,
        pad_id: int,
        positive: int,
        max_length: int = 512,
        stride: int = 64,
    ) -> None:
        self.model_id = model_id
        self._session = sess
        self._tokenizer = tokenizer
        tokenizer.no_padding()
        tokenizer.no_truncation()
        self._prefix, self._suffix = special_frame(tokenizer)
        self._body = max_length - len(self._prefix) - len(self._suffix)
        if not 0 <= stride < self._body:
            raise ValueError(f"{model_id}: stride must be smaller than the window body ({self._body} tokens)")
        self._step = self._body - stride
        self._pad_id = pad_id
        self._positive = positive

    @classmethod
    def load(cls, spec: ModelSpec, files: Mapping[FileRole, Path]) -> "OnnxTextClassifier":
        tokenizer = Tokenizer.from_file(str(files["tokenizer"]))
        config: dict[str, Any] = json.loads(files["config"].read_text()) if "config" in files else {}
        pad_id = config.get("pad_token_id")
        model = cls(
            spec.id,
            session(files["onnx"]),
            tokenizer,
            pad_id=int(pad_id) if isinstance(pad_id, int) else 0,
            positive=spec.positive,
            max_length=spec.max_length,
            stride=spec.stride,
        )
        model.warmup()
        return model

    def warmup(self) -> None:
        # first runs pay for graph optimisation and allocator growth; do that before traffic
        for n in _WARMUP_TOKENS:
            self.scores(["warm " * n])

    def windows(self, text: str, max_windows: int) -> list[list[int]]:
        """token windows of at most max_length, overlapping by `stride` tokens, each framed by special tokens.

        past max_windows the head and the tail are kept: injected instructions tend to open or close a text.
        """
        ids = self._tokenizer.encode(text, add_special_tokens=False).ids
        starts = list(range(0, max(len(ids) - (self._body - self._step), 1), self._step))
        if len(starts) > max_windows:
            head = (max_windows + 1) // 2
            starts = starts[:head] + starts[len(starts) - (max_windows - head) :]
        return [[*self._prefix, *ids[i : i + self._body], *self._suffix] for i in starts]

    def _run(self, rows: Sequence[Sequence[int]]) -> FloatArray:
        out: list[FloatArray] = []
        for start in range(0, len(rows), BATCH):
            batch = rows[start : start + BATCH]
            logits = np.asarray(self._session.run(None, feed(self._session, batch, self._pad_id))[0])
            if logits.shape[-1] == 1:
                probs = 1.0 / (1.0 + np.exp(-logits[:, 0]))
            else:
                probs = softmax(logits)[:, self._positive]
            out.append(probs.astype(np.float32))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def scores(self, texts: Sequence[str], *, max_windows: int = 16) -> list[float]:
        """blocking; call through the cpu executor"""
        owners: list[int] = []
        rows: list[list[int]] = []
        for i, text in enumerate(texts):
            for row in self.windows(text, max_windows):
                owners.append(i)
                rows.append(row)
        best = [0.0] * len(texts)
        for owner, p in zip(owners, self._run(rows).tolist(), strict=True):
            best[owner] = max(best[owner], float(p))
        return best
