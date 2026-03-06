"""Central configuration for HyP-DLM pipeline. All hyperparameters from paper."""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # --- General ---
    device: str = "cpu"  # "cuda", "cpu", "mps"
    seed: int = 42
    storage_dir: str = "storage"
    embedding_model: str = "all-mpnet-base-v2"

    # --- Coreference ---
    coref_model: str = "LingMessCoref"
    coref_batch_size: int = 32

    # --- Chunking ---
    chunk_breakpoint_percentile: float = 90.0

    # --- NER ---
    ner_backend: str = "gliner"  # "gliner" or "spacy"
    gliner_model: str = "fastino/gliner2-base-v1"
    spacy_model: str = "en_core_web_sm"
    ner_min_entity_len: int = 3
    ner_entity_types: list = field(default_factory=lambda: [
        "person", "organization", "location", "event",
        "date", "concept", "product", "technology",
    ])

    # --- Synonym ---
    synonym_rrf_k: int = 60
    synonym_char_ngram_min: int = 2
    synonym_char_ngram_max: int = 4
    synonym_context_gate: str = "AND"
    synonym_cosine_agg: str = "MAX"
    synonym_rrf_threshold: float = 0.01
    synonym_ctx_threshold: float = 0.3

    # --- Masking ---
    hdbscan_min_cluster_size: int = 15
    multi_label_threshold: float = 0.5  # tau_multi

    # --- Propagation ---
    ppr_alpha: float = 0.85
    ppr_epsilon_conv: float = 0.01
    ppr_max_iterations: int = 6  # T
    ppr_top_p_clusters: int = 5
    ppr_delta_pruning: float = 0.05
    ppr_tau_init: float = 0.5
    ppr_alpha_parent: float = 0.5

    # --- Scoring ---
    score_weight_hyper: float = 1.0  # w_h
    score_weight_entity: float = 0.3  # w_e
    score_top_k_per_subq: int = 30
    score_top_n_final: int = 10

    # --- LLM ---
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # --- Workers ---
    max_workers: int = 4
    index_batch_size: int = 10

    # --- Logging ---
    log_level: str = "INFO"
    log_dir: str = "logs"
    metrics_dir: str = "metrics"

    # --- Output ---
    output_dir: str = "output"

    # --- Dataset ---
    dataset_name: str = "default"

    def storage_path(self) -> Path:
        """Returns storage path with embedding model name for ablation."""
        # Replace / in model name with underscore
        safe_name = self.embedding_model.replace("/", "_")
        return Path(self.storage_dir) / safe_name

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        """Load config from YAML, merge with defaults."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    @classmethod
    def from_args(cls, yaml_path: Optional[str] = None, **overrides) -> Config:
        """Load from YAML then apply CLI overrides."""
        if yaml_path and os.path.exists(yaml_path):
            cfg = cls.from_yaml(yaml_path)
        else:
            cfg = cls()
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
