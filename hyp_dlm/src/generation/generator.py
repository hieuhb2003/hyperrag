"""
Hybrid RAG Generator — combines graph-retrieved and chunk-retrieved knowledge
to generate final answers.
"""

from typing import Optional

from src.utils.logger import get_logger
from src.utils.llm_client import LLMClient

logger = get_logger(__name__)

GENERATION_PROMPT = """---Role--- You are a helpful assistant responding to questions based on given knowledge.
---Knowledge---
{retrieved_knowledge}
---Goal--- Answer the given question.
You must first conduct reasoning inside <think>...</think>.
When you have the final answer, output it inside <answer>...</answer>.
---Question---
{question}"""


class HybridRAGGenerator:
    """Generate answers using combined graph + dense retrieval results."""

    def __init__(self, config: dict, llm_client: Optional[LLMClient] = None):
        self.llm_model = config.get("llm_model", "gpt-4o-mini")
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.0)
        self.fusion_weight_graph = config.get("fusion_weight_graph", 0.7)
        self.fusion_weight_chunk = config.get("fusion_weight_chunk", 0.3)
        self._llm = llm_client

        logger.step(
            "Generator",
            "Initialized",
            model=self.llm_model,
            fusion_graph=self.fusion_weight_graph,
            fusion_chunk=self.fusion_weight_chunk,
        )

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(
                model=self.llm_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return self._llm

    def set_llm(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def generate(
        self,
        query: str,
        graph_passages: list[dict],
        chunk_passages: list[dict],
        chunks: list,
    ) -> dict:
        """
        Generate final answer from combined retrieval results.

        Args:
            query: original question
            graph_passages: [{"hyperedge_id": int, "score": float}]
            chunk_passages: [{"hyperedge_id": int, "score": float}]
            chunks: list of Chunk objects for text lookup

        Returns:
            {
                "answer": str,
                "raw_response": str,
                "retrieved_passages": list[dict],
            }
        """
        logger.start_timer("generation")
        logger.debug(
            f"Generating answer: {len(graph_passages)} graph passages, "
            f"{len(chunk_passages)} chunk passages"
        )

        # 1. Merge and deduplicate
        merged = self._merge_passages(graph_passages, chunk_passages)
        logger.debug(f"  Merged: {len(merged)} unique passages")

        # 2. Build context string
        retrieved_texts = []
        for passage in merged:
            he_id = passage["hyperedge_id"]
            if he_id < len(chunks):
                retrieved_texts.append({
                    "hyperedge_id": he_id,
                    "score": passage["score"],
                    "text": chunks[he_id].text,
                })

        # Sort by score descending
        retrieved_texts.sort(key=lambda x: x["score"], reverse=True)

        # Build knowledge string
        knowledge_parts = []
        for i, rt in enumerate(retrieved_texts):
            knowledge_parts.append(
                f"[Passage {i+1} (score: {rt['score']:.4f})]\n{rt['text']}"
            )
        knowledge_str = "\n\n".join(knowledge_parts)

        logger.debug(f"  Context length: {len(knowledge_str)} chars")

        # 3. Generate
        prompt = GENERATION_PROMPT.format(
            retrieved_knowledge=knowledge_str,
            question=query,
        )

        raw_response = self.llm.generate(prompt)

        # 4. Extract answer from <answer> tags
        answer = self._extract_answer(raw_response)

        elapsed = logger.stop_timer("generation")

        logger.step(
            "Generator",
            f"Generated answer ({len(answer)} chars)",
            passages_used=len(retrieved_texts),
            time=elapsed,
        )
        logger.debug(f"  Answer: '{answer[:200]}...'")

        return {
            "answer": answer,
            "raw_response": raw_response,
            "retrieved_passages": retrieved_texts,
        }

    def _merge_passages(
        self,
        graph_passages: list[dict],
        chunk_passages: list[dict],
    ) -> list[dict]:
        """Merge and deduplicate passages with weighted scores."""
        score_map: dict[int, float] = {}

        for p in graph_passages:
            he_id = p["hyperedge_id"]
            score_map[he_id] = score_map.get(he_id, 0) + self.fusion_weight_graph * p["score"]

        for p in chunk_passages:
            he_id = p["hyperedge_id"]
            score_map[he_id] = score_map.get(he_id, 0) + self.fusion_weight_chunk * p["score"]

        # Sort by combined score
        merged = sorted(
            [{"hyperedge_id": k, "score": v} for k, v in score_map.items()],
            key=lambda x: x["score"],
            reverse=True,
        )

        return merged

    @staticmethod
    def _extract_answer(response: str) -> str:
        """Extract answer from <answer>...</answer> tags."""
        import re
        match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: return the whole response
        return response.strip()
