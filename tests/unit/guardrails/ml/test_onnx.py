"""OnnxTextClassifier windowing, batching and padding with a word-level tokenizer and a scripted session"""

import itertools
from dataclasses import dataclass, field
from typing import Any

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("onnxruntime")
tokenizers = pytest.importorskip("tokenizers")

from gg.guardrails.ml.onnx import BATCH, OnnxTextClassifier  # noqa: E402

VOCAB = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "bad": 4, **{f"w{i}": 10 + i for i in range(100)}}
BAD = VOCAB["bad"]
MAX_LEN = 8
STRIDE = 2


def tokenizer() -> Any:
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    tok = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    return tok


@dataclass
class Input:
    name: str


@dataclass
class FakeSession:
    """two-class logits: unsafe when the row holds the 'bad' token; one-logit mode for sigmoid heads"""

    inputs: tuple[str, ...] = ("input_ids", "attention_mask")
    one_logit: bool = False
    batches: list[dict[str, Any]] = field(default_factory=lambda: [])

    def get_inputs(self) -> list[Input]:
        return [Input(n) for n in self.inputs]

    def run(self, outputs: Any, feeds: dict[str, Any]) -> list[Any]:
        self.batches.append(feeds)
        bad = (feeds["input_ids"] == BAD).any(axis=1)
        if self.one_logit:
            return [np.where(bad, 4.0, -4.0)[:, None].astype(np.float32)]
        return [np.stack([np.where(bad, 0.0, 6.0), np.where(bad, 6.0, 0.0)], axis=1).astype(np.float32)]


def classifier(session: FakeSession | None = None) -> tuple[OnnxTextClassifier, FakeSession]:
    sess = session or FakeSession()
    clf = OnnxTextClassifier(
        "fake", sess, tokenizer(), pad_id=0, positive=1, max_length=MAX_LEN, stride=STRIDE
    )  # type: ignore[arg-type]
    return clf, sess


def words(n: int, *, bad_at: int | None = None) -> str:
    return " ".join("bad" if i == bad_at else f"w{i}" for i in range(n))


def test_long_text_is_split_into_overlapping_windows() -> None:
    clf, _ = classifier()
    windows = clf.windows(words(20), max_windows=16)
    assert len(windows) > 1
    assert all(w[0] == 1 and w[-1] == 2 and len(w) <= MAX_LEN for w in windows)
    bodies = [w[1:-1] for w in windows]
    for prev, nxt in itertools.pairwise(bodies):
        assert prev[-STRIDE:] == nxt[:STRIDE]
    assert sorted({t for b in bodies for t in b}) == [10 + i for i in range(20)]


def test_score_is_the_max_over_windows_so_a_late_attack_is_found() -> None:
    clf, _ = classifier()
    [late, clean] = clf.scores([words(30, bad_at=28), words(30)])
    assert late > 0.99
    assert clean < 0.01


def test_max_windows_keeps_the_head_and_the_tail() -> None:
    clf, _ = classifier()
    windows = clf.windows(words(60), max_windows=3)
    assert len(windows) == 3
    assert windows[0][1] == 10
    assert windows[-1][-2] == 10 + 59
    assert clf.scores([words(60, bad_at=59)], max_windows=2) == [pytest.approx(1.0, abs=0.01)]
    assert clf.scores([words(60, bad_at=30)], max_windows=2) == [pytest.approx(0.0, abs=0.01)]


def test_windows_of_all_texts_are_batched_and_right_padded() -> None:
    clf, sess = classifier(FakeSession(inputs=("input_ids", "attention_mask", "token_type_ids")))
    texts = [words(3)] * (BATCH + 3)
    texts[-1] = words(10, bad_at=9)
    scores = clf.scores(texts)
    assert scores[-1] > 0.99
    assert max(scores[:-1]) < 0.01
    assert [len(b["input_ids"]) for b in sess.batches] == [BATCH, 4]
    last = sess.batches[-1]
    assert set(last) == {"input_ids", "attention_mask", "token_type_ids"}
    short_row = last["input_ids"][0]
    assert short_row.tolist()[:5] == [1, 10, 11, 12, 2]
    assert last["attention_mask"][0].tolist() == [1] * 5 + [0] * (len(short_row) - 5)
    assert (short_row[5:] == 0).all()


def test_single_logit_heads_use_a_sigmoid() -> None:
    clf, _ = classifier(FakeSession(one_logit=True))
    [bad, ok] = clf.scores(["bad", "w1"])
    assert bad == pytest.approx(1 / (1 + np.exp(-4.0)))
    assert ok == pytest.approx(1 / (1 + np.exp(4.0)))
