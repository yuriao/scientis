"""Agent runner — executes the LangGraph scientific discovery workflow.

Supports:
  - run()    — await a single result dict
  - stream() — async-iterate over per-node events
  - resume_with_review() — resume after human review via interrupt/Command
"""

import logging
import uuid
from typing import Any, AsyncIterator, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from scientis.agent.graph import build_workflow
from scientis.agent.state import AgentState, WorkflowConfig

logger = logging.getLogger(__name__)


class DiscoveryRunner:
    """Runs the scientific discovery workflow."""

    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self._memory = MemorySaver()
        self._workflow = build_workflow().compile(checkpointer=self._memory)

    def _make_initial_state(self, question: str, session_id: str) -> AgentState:
        return {
            "question": question,
            "session_id": session_id,
            "expanded_queries": [],
            "disease_synonyms": [],
            "mechanism_synonyms": [],
            "retrieved_chunks": [],
            "retrieved_claims": [],
            "evidence_count": 0,
            "comparison_set": [],
            "supporting_papers": [],
            "conflicting_papers": [],
            "hypotheses": [],
            "ranked_hypotheses": [],
            "counterevidence_findings": [],
            "weakened_hypotheses": [],
            "review_required": False,
            "review_decisions": [],
            "reviewed_by": "",
            "status": "running",
            "final_report": None,
            "error_message": "",
            "iteration_count": 0,
            "model_tier_used": self.config.model_tier,
        }

    async def run(
        self, question: str, session_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Run the workflow to completion (or until human review is required).

        Returns the final AgentState as a plain dict. If the workflow pauses
        at the human_review node, status will be 'awaiting_review' and the
        caller should call resume_with_review() to continue.
        """
        if session_id is None:
            session_id = f"sess-{uuid.uuid4().hex[:12]}"

        initial_state = self._make_initial_state(question, session_id)
        thread_config = {"configurable": {"thread_id": session_id}}

        try:
            result = await self._workflow.ainvoke(initial_state, thread_config)
            return dict(result) if result else initial_state
        except Exception as e:
            logger.exception("Workflow failed for session %s", session_id)
            return {**initial_state, "status": "error", "error_message": str(e)}

    async def stream(
        self, question: str, session_id: Optional[str] = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Run the workflow and yield a state snapshot after each node completes."""
        if session_id is None:
            session_id = f"sess-{uuid.uuid4().hex[:12]}"

        initial_state = self._make_initial_state(question, session_id)
        thread_config = {"configurable": {"thread_id": session_id}}
        accumulated = dict(initial_state)

        async for event in self._workflow.astream(initial_state, thread_config):
            for node_name, node_output in event.items():
                accumulated.update(node_output)
                yield {"node": node_name, "state": dict(accumulated)}

    async def resume_with_review(
        self,
        session_id: str,
        review_decisions: list[dict],
        reviewer: str = "",
    ) -> dict[str, Any]:
        """Resume a workflow that is paused at the human_review interrupt.

        review_decisions: list of {hypothesis_id, action: accept|reject|revise, comment}
        """
        thread_config = {"configurable": {"thread_id": session_id}}
        try:
            result = await self._workflow.ainvoke(
                Command(resume={"decisions": review_decisions, "reviewer": reviewer}),
                thread_config,
            )
            return dict(result) if result else {}
        except Exception as e:
            logger.exception("Workflow resume failed for session %s", session_id)
            return {"status": "error", "error_message": str(e)}


_runner: Optional[DiscoveryRunner] = None


def get_runner() -> DiscoveryRunner:
    global _runner
    if _runner is None:
        _runner = DiscoveryRunner()
    return _runner
