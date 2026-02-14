"""
Query Decomposer — breaks multi-hop questions into a DAG of sub-questions.

This is the ONE place where LLM is called during retrieval.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from src.utils.logger import get_logger
from src.utils.llm_client import LLMClient

logger = get_logger(__name__)

DECOMPOSITION_PROMPT = """You are a question decomposition expert. Given a complex question, break it down into simple single-hop sub-questions and their dependencies.

Output format (JSON):
{
  "nodes": [
    {"id": "q1", "question": "...", "depends_on": []},
    {"id": "q2", "question": "...", "depends_on": ["q1"]},
    {"id": "q3", "question": "...", "depends_on": ["q1", "q2"]}
  ]
}

Rules:
- Each sub-question should be answerable from a single passage.
- Use <ANS-qN> to reference the answer of sub-question qN.
- The DAG must be acyclic.
- For comparison questions, create parallel branches.

Examples:
Q: "Who is older, the founder of Microsoft or the founder of Apple?"
→ {
    "nodes": [
      {"id": "q1", "question": "Who founded Microsoft?", "depends_on": []},
      {"id": "q2", "question": "Who founded Apple?", "depends_on": []},
      {"id": "q3", "question": "When was <ANS-q1> born?", "depends_on": ["q1"]},
      {"id": "q4", "question": "When was <ANS-q2> born?", "depends_on": ["q2"]},
      {"id": "q5", "question": "Who is older, <ANS-q1> (born <ANS-q3>) or <ANS-q2> (born <ANS-q4>)?", "depends_on": ["q3", "q4"]}
    ]
  }

Question: {query}"""


@dataclass
class SubQuestion:
    """A node in the query DAG."""
    id: str
    question: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class QueryDAG:
    """Directed Acyclic Graph of sub-questions."""
    nodes: list[SubQuestion]
    original_query: str

    def topological_order(self) -> list[SubQuestion]:
        """Return nodes in topological order (respecting dependencies)."""
        # Kahn's algorithm
        in_degree: dict[str, int] = {n.id: 0 for n in self.nodes}
        adj: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        node_map = {n.id: n for n in self.nodes}

        for n in self.nodes:
            for dep in n.depends_on:
                if dep in adj:
                    adj[dep].append(n.id)
                    in_degree[n.id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            nid = queue.pop(0)
            order.append(node_map[nid])
            for child in adj.get(nid, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            logger.warning("DAG has cycles! Falling back to original order.")
            return self.nodes

        return order

    @property
    def is_single_hop(self) -> bool:
        return len(self.nodes) <= 1


class QueryDecomposer:
    """Decompose multi-hop queries into DAG of sub-questions."""

    def __init__(self, config: dict, llm_client: Optional[LLMClient] = None):
        self.llm_model = config.get("llm_model", "gpt-4o-mini")
        self.max_sub_questions = config.get("max_sub_questions", 6)
        self._llm = llm_client

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(model=self.llm_model)
        return self._llm

    def set_llm(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def decompose(self, query: str) -> QueryDAG:
        """
        Decompose a query into a DAG of sub-questions.

        For simple single-hop queries, returns a single-node DAG.
        """
        logger.start_timer("decomposition")
        logger.debug(f"Decomposing query: '{query}'")

        prompt = DECOMPOSITION_PROMPT.replace("{query}", query)

        try:
            result = self.llm.generate_json(prompt)

            nodes = []
            for node_data in result.get("nodes", []):
                nodes.append(
                    SubQuestion(
                        id=node_data["id"],
                        question=node_data["question"],
                        depends_on=node_data.get("depends_on", []),
                    )
                )

            # Limit sub-questions
            if len(nodes) > self.max_sub_questions:
                logger.warning(
                    f"DAG has {len(nodes)} nodes, truncating to {self.max_sub_questions}"
                )
                nodes = nodes[: self.max_sub_questions]

        except Exception as e:
            logger.warning(f"Decomposition failed: {e}. Using query as single hop.")
            nodes = [SubQuestion(id="q1", question=query, depends_on=[])]

        dag = QueryDAG(nodes=nodes, original_query=query)

        # Validate DAG
        topo = dag.topological_order()

        elapsed = logger.stop_timer("decomposition")
        logger.step(
            "QueryDecomposer",
            f"Decomposed into {len(dag.nodes)} sub-questions "
            f"(single_hop={dag.is_single_hop})",
            time=elapsed,
        )

        # Log sub-questions
        for i, node in enumerate(topo):
            logger.debug(
                f"  {node.id}: '{node.question}' "
                f"(depends_on={node.depends_on})"
            )

        return dag

    @staticmethod
    def resolve_references(question: str, answers: dict[str, str]) -> str:
        """Replace <ANS-qN> placeholders with actual answers."""
        resolved = question
        for ans_id, answer in answers.items():
            placeholder = f"<ANS-{ans_id}>"
            resolved = resolved.replace(placeholder, answer)

        if resolved != question:
            logger.debug(f"  Resolved: '{question}' → '{resolved}'")

        return resolved
