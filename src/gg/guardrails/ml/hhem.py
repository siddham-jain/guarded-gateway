"""vectara hhem-2.1-open (flan-t5 encoder + token head) as onnx; imported only when the model loads"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from gg.guardrails.ml.artefacts import FileRole, ModelSpec
from gg.guardrails.ml.onnx import feed, session, softmax

# upstream prompt; the literal <pad> is part of the template the model was trained with
TEMPLATE = (
    "<pad> Determine if the hypothesis is true given the premise?\n\n"
    "Premise: {premise}\n\nHypothesis: {hypothesis}"
)
_BATCH = 4


class HhemScorer:
    """consistency in [0, 1] of each hypothesis against one premise; 1 = supported"""

    def __init__(self, spec: ModelSpec, files: Mapping[FileRole, Path]) -> None:
        self.model_id = spec.id
        self._tokenizer = Tokenizer.from_file(str(files["tokenizer"]))
        self._tokenizer.no_padding()
        self._tokenizer.enable_truncation(spec.max_length)
        self._session = session(files["onnx"])
        self.scores("warm up", ["warm up"])

    def scores(self, premise: str, hypotheses: Sequence[str]) -> list[float]:
        """blocking; call through the cpu executor"""
        prompts = [TEMPLATE.format(premise=premise, hypothesis=h) for h in hypotheses]
        out: list[float] = []
        for start in range(0, len(prompts), _BATCH):
            rows = [e.ids for e in self._tokenizer.encode_batch(prompts[start : start + _BATCH])]
            logits = np.asarray(self._session.run(None, feed(self._session, rows, 0))[0])
            out.extend(float(p) for p in softmax(logits[:, 0, :])[:, 1])
        return out
