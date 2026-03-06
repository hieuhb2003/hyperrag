"""Trace-augmented answer generation via LLM."""

from __future__ import annotations

import importlib

from src.config import Config

_logging = importlib.import_module("src.logging-setup")
_tt = importlib.import_module("src.token-tracker")
logger = _logging.get_logger("generator")


class AnswerGenerator:
    """Generate final answers using trace-augmented prompting."""

    def __init__(self, config: Config, token_tracker: _tt.TokenTracker):
        self.config = config
        self.token_tracker = token_tracker

    def generate(self, query, retrieval_result: dict, chunks: list) -> dict:
        """Generate final answer for a query using retrieval results."""
        import litellm

        # Build reasoning trace from DAG
        trace = self._build_trace(retrieval_result)

        # Build evidence from top chunks
        evidence = self._build_evidence(retrieval_result["top_results"], chunks)

        # Build prompt
        prompt = self._build_prompt(query.query, trace, evidence)
        logger.debug(f"Prompt length: {len(prompt)} chars")

        # LLM call
        response = litellm.completion(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )

        # Track tokens
        self.token_tracker.record(
            phase="generation",
            doc_or_query_id=query.id,
            model=self.config.llm_model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

        answer = response.choices[0].message.content.strip()

        # Extract answer from <answer> tags if present
        clean_answer = self._extract_answer(answer)

        logger.info(f"Query '{query.id}': answer length {len(clean_answer)} chars")

        return {
            "query_id": query.id,
            "query": query.query,
            "generated_answer": clean_answer,
            "raw_answer": answer,
            "ground_truth": query.answer,
            "evidence_doc_ids": query.evidence,
            "reasoning_trace": trace,
            "top_results": retrieval_result["top_results"],
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

    def _build_trace(self, retrieval_result: dict) -> str:
        """Build reasoning trace from DAG traversal."""
        dag = retrieval_result["dag"]
        intermediate = retrieval_result.get("intermediate_answers", {})
        lines = ["--- Reasoning Path ---"]
        for sq_id in dag.topological_order:
            sq = next(s for s in dag.sub_questions if s.id == sq_id)
            ans = intermediate.get(sq_id, "[to be determined]")
            lines.append(f"Sub-question {sq.id}: {sq.text}")
            lines.append(f"  Answer: {ans}")
        return "\n".join(lines)

    def _build_evidence(self, top_results: list[tuple[str, float]], chunks: list) -> str:
        """Build evidence text from top-ranked chunks."""
        chunk_map = {c.id: c for c in chunks}
        parts = []
        for rank, (chunk_id, score) in enumerate(top_results[:10]):
            chunk = chunk_map.get(chunk_id)
            if chunk:
                parts.append(f"[{rank + 1}] (score: {score:.3f}) {chunk.text}")
        return "\n\n".join(parts)

    def _build_prompt(self, query: str, trace: str, evidence: str) -> str:
        return f"""{trace}

--- Evidence (ranked by relevance) ---
{evidence}

--- Question ---
{query}

--- Instruction ---
Based on the reasoning path and evidence above, answer the question.
Reason inside <think>...</think>, answer inside <answer>...</answer>."""

    def _extract_answer(self, text: str) -> str:
        """Extract answer from <answer>...</answer> tags if present."""
        import re
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
