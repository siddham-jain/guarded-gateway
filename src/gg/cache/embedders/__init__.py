from pathlib import Path

from gg.cache.base import Embedder
from gg.cache.config import EmbedderConfig
from gg.cache.embedders.fastembed import FastEmbedEmbedder
from gg.cache.embedders.hashing import HashingEmbedder
from gg.core.aio import CpuExecutor


def build_embedder(cfg: EmbedderConfig, cpu: CpuExecutor, *, cache_dir: Path | None = None) -> Embedder:
    """the composition root builds one and shares it with the topic guard"""
    if cfg.provider == "hashing":
        return HashingEmbedder(cfg.dim)
    return FastEmbedEmbedder(cpu, model=cfg.name, dim=cfg.dim, cache_dir=cache_dir)
