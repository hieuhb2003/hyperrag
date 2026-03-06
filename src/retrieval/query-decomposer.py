"""Query decomposition into DAG of sub-questions via LLM."""

from __future__ import annotations

import importlib
import json
import re
from collections import deque
from dataclasses import dataclass, field

from src.config import Config

_logging = importlib.import_module("src.logging-setup")
_tt = importlib.import_module("src.token-tracker")
logger = _logging.get_logger("query-decomposer")


@dataclass
class SubQuestion:
    id: str  # "sq_0", "sq_1", ...
    text: str  # sub-question text with possible <ANS-sq_X> placeholders
    depends_on: list[str] = field(default_factory=list)


@dataclass
class QueryDAG:
    root_query: str
    sub_questions: list[SubQuestion]
    topological_order: list[str]  # IDs in execution order


class QueryDecomposer:
    """Decompose complex queries into DAG of sub-questions."""

    def __init__(self, config: Config, token_tracker: _tt.TokenTracker):
        self.config = config
        self.token_tracker = token_tracker

    def decompose(self, query: str, query_id: str) -> QueryDAG:
        """Decompose query into DAG via single LLM call."""
        import litellm

        prompt = self._build_prompt(query)

        response = litellm.completion(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.llm_temperature,
            response_format={"type": "json_object"},
        )

        # Track tokens
        self.token_tracker.record(
            phase="retrieval",
            doc_or_query_id=query_id,
            model=self.config.llm_model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

        # Parse response
        content = response.choices[0].message.content
        sub_questions = self._parse_response(content)

        # Fallback: if parsing fails or returns empty, treat as single question
        if not sub_questions:
            logger.warning(f"Query '{query_id}': decomposition failed, using single node")
            sub_questions = [SubQuestion(id="sq_0", text=query, depends_on=[])]

        topo_order = self._topological_sort(sub_questions)
        dag = QueryDAG(root_query=query, sub_questions=sub_questions, topological_order=topo_order)
        logger.info(f"Query '{query_id}': {len(sub_questions)} sub-questions, "
                    f"order={topo_order}")
        return dag

    def resolve_placeholders(self, sq: SubQuestion, answers: dict[str, str]) -> str:
        """Replace <ANS-sq_X> placeholders with actual answers."""
        text = sq.text
        for dep_id in sq.depends_on:
            if dep_id in answers:
                text = text.replace(f"<ANS-{dep_id}>", answers[dep_id])
        return text

    def _build_prompt(self, query: str) -> str:
        return f"""You are a question decomposition expert. Given a complex question, break it down into simple single-hop sub-questions and their dependencies.

Output format (JSON):
{{
  "nodes": [
    {{"id": "sq_0", "question": "...", "depends_on": []}},
    {{"id": "sq_1", "question": "...", "depends_on": ["sq_0"]}}
  ]
}}

Rules:
- Each sub-question should be answerable from a single passage.
- Use <ANS-sq_N> to reference the answer of sub-question sq_N.
- The DAG must be acyclic.
- For comparison questions, create parallel branches.
- For simple questions, return a single node.

Question: {query}"""

    def _parse_response(self, text: str) -> list[SubQuestion]:
        """Parse LLM JSON output into SubQuestion objects."""
        try:
            # Try direct JSON parse
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code block
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    return []
            else:
                return []

        nodes = data.get("nodes", [])
        sub_questions = []
        for node in nodes:
            sq = SubQuestion(
                id=node.get("id", f"sq_{len(sub_questions)}"),
                text=node.get("question", ""),
                depends_on=node.get("depends_on", []),
            )
            sub_questions.append(sq)
        return sub_questions

    def _topological_sort(self, sub_questions: list[SubQuestion]) -> list[str]:
        """Kahn's algorithm for topological sort."""
        # Build adjacency + in-degree
        nodes = {sq.id for sq in sub_questions}
        in_degree = {sq.id: 0 for sq in sub_questions}
        children: dict[str, list[str]] = {sq.id: [] for sq in sub_questions}

        for sq in sub_questions:
            for dep in sq.depends_on:
                if dep in nodes:
                    in_degree[sq.id] += 1
                    children[dep].append(sq.id)

        # Process zero in-degree nodes
        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for child in children.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        # If not all nodes visited, there's a cycle — fall back to original order
        if len(order) != len(sub_questions):
            logger.warning("Cycle detected in DAG, using original order")
            return [sq.id for sq in sub_questions]

        return order
