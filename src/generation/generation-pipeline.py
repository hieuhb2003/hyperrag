"""Generation pipeline: orchestrates answer generation and output saving."""

from __future__ import annotations

import importlib

from tqdm import tqdm

from src.config import Config

_logging = importlib.import_module("src.logging-setup")
_tt = importlib.import_module("src.token-tracker")
_timer = importlib.import_module("src.time-tracker")
_ag = importlib.import_module("src.generation.answer-generator")
_os = importlib.import_module("src.evaluation.output-saver")

logger = _logging.get_logger("generation")


def run_generation(config: Config, queries_path: str, storage_path=None):
    """Entry point for generation command."""
    _dl = importlib.import_module("src.data-loader")
    _rp = importlib.import_module("src.retrieval.retrieval-pipeline")

    queries = _dl.load_queries(queries_path)
    index_data = _rp._load_index(config, storage_path)

    token_tracker = _tt.TokenTracker(config.metrics_dir)
    time_tracker = _timer.TimeTracker(config.metrics_dir)

    # Run retrieval first
    ret_pipeline = _rp.RetrievalPipeline(config, token_tracker, time_tracker)
    ret_pipeline.setup(index_data)
    retrieval_results = ret_pipeline.run_all(queries, index_data)

    # Run generation
    pipeline = GenerationPipeline(config, token_tracker, time_tracker)
    gen_results = pipeline.run(queries, retrieval_results, index_data["chunks"], config.dataset_name)

    # Save metrics
    token_tracker.save(config.dataset_name)
    time_tracker.save(config.dataset_name)

    logger.info(f"Generation complete: {len(gen_results)} answers generated")
    return gen_results


class GenerationPipeline:
    """Orchestrate answer generation for all queries."""

    def __init__(self, config: Config, token_tracker: _tt.TokenTracker,
                 time_tracker: _timer.TimeTracker):
        self.config = config
        self.generator = _ag.AnswerGenerator(config, token_tracker)
        self.saver = _os.OutputSaver(config)
        self.time_tracker = time_tracker

    def run(self, queries: list, retrieval_results: list[dict],
            chunks: list, dataset_name: str) -> list[dict]:
        """Generate answers for all queries and save outputs."""
        gen_results = []
        for query, ret_result in tqdm(zip(queries, retrieval_results),
                                       total=len(queries), desc="Generation"):
            with self.time_tracker.track("generation", query.id):
                result = self.generator.generate(query, ret_result, chunks)
                gen_results.append(result)

        # Save outputs
        self.saver.save_retrieval_results(retrieval_results, dataset_name)
        self.saver.save_generation_results(gen_results, dataset_name)

        logger.info(f"Generated answers for {len(gen_results)} queries")
        return gen_results
