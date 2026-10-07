"""presidio analyzer on spacy en_core_web_sm plus gg's own recognizers; imported only when it loads"""

from collections.abc import Sequence
from dataclasses import dataclass

import spacy
from presidio_analyzer import AnalyzerEngine, EntityRecognizer, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NerModelConfiguration, SpacyNlpEngine
from presidio_analyzer.predefined_recognizers import (
    UkDrivingLicenceRecognizer,
    UkNinoRecognizer,
    UkPassportRecognizer,
)

# email and url recognizers fetch the public suffix list over the network (tldextract); pii_regex covers email
_DROPPED = ("EmailRecognizer", "UrlRecognizer")


@dataclass(frozen=True, slots=True)
class EntityHit:
    entity: str
    start: int
    end: int
    score: float


def custom_recognizers() -> list[EntityRecognizer]:
    # base scores sit below min_score so dates and sort codes only count next to their context words
    months = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
    return [
        PatternRecognizer(
            supported_entity="DATE_OF_BIRTH",
            patterns=[
                Pattern(
                    "dob_numeric",
                    r"\b(?:0?[1-9]|[12]\d|3[01])[/.-](?:0?[1-9]|1[0-2])[/.-](?:19|20)\d{2}\b",
                    0.3,
                ),
                Pattern("dob_iso", r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b", 0.3),
                Pattern(
                    "dob_text",
                    rf"(?i)\b(?:[0-3]?\d\s+{months}|{months}\s+[0-3]?\d,?)\s+(?:19|20)\d{{2}}\b",
                    0.3,
                ),
            ],
            # context is matched on lemmas and spacy lemmatises "born" to "bear"
            context=["born", "bear", "birth", "birthday", "dob"],
        ),
        PatternRecognizer(
            supported_entity="STREET_ADDRESS",
            patterns=[
                Pattern(
                    "us_street",
                    r"\b\d{1,6}\s+(?:[A-Z][a-z]+\s+){1,4}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln"
                    r"|Drive|Dr|Court|Ct|Way|Place|Pl|Terrace|Close)\b\.?",
                    0.6,
                )
            ],
            context=["address", "live", "lives", "street", "ship"],
        ),
        PatternRecognizer(
            supported_entity="UK_SORT_CODE",
            patterns=[Pattern("sort_code", r"\b\d{2}-\d{2}-\d{2}\b", 0.3)],
            context=["sort", "code", "bank", "account"],
        ),
        UkNinoRecognizer(),
        UkPassportRecognizer(),
        UkDrivingLicenceRecognizer(),
    ]


_UNMAPPED_SPACY_LABELS = (
    "CARDINAL",
    "ORDINAL",
    "QUANTITY",
    "PERCENT",
    "MONEY",
    "FAC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
)


def build_analyzer(spacy_model: str) -> AnalyzerEngine:
    # presidio would try to pip-download a missing spacy model at runtime; fail instead
    if not spacy.util.is_package(spacy_model):
        raise RuntimeError(f"spacy model '{spacy_model}' is not installed (it ships with the 'ml' extra)")
    # spacy labels with no presidio entity (numbers, places, works of art) only produce a warning per request
    ner = NerModelConfiguration(labels_to_ignore=list(_UNMAPPED_SPACY_LABELS))
    nlp_engine = SpacyNlpEngine(
        models=[{"lang_code": "en", "model_name": spacy_model}], ner_model_configuration=ner
    )
    # the dependency parser is a quarter of the cost and presidio only needs tokens, lemmas and entities
    nlp_engine.nlp = {"en": spacy.load(spacy_model, exclude=["parser"])}
    engine = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    for name in _DROPPED:
        engine.registry.remove_recognizer(name)
    for recognizer in custom_recognizers():
        engine.registry.add_recognizer(recognizer)
    engine.analyze("warm up: Jane Doe was born on 4 May 1990", language="en")
    return engine


class PresidioAnalyzer:
    def __init__(self, spacy_model: str) -> None:
        self._engine = build_analyzer(spacy_model)

    def analyze(self, text: str, entities: Sequence[str], min_score: float) -> list[EntityHit]:
        """blocking; call through the cpu executor"""
        results = self._engine.analyze(
            text, language="en", entities=list(entities), score_threshold=min_score
        )
        return [EntityHit(r.entity_type, r.start, r.end, float(r.score)) for r in results]
