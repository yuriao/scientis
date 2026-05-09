"""Agent runner — executes the LangGraph workflow.

Supports:
  - Synchronous streaming execution
  - Human-in-the-loop checkpoints (pause/resume)
  - Error recovery
"""

import asyncio
import logging
import uuid
from typing import Any, AsyncIterator, Optional

from langgraph.checkpoint.memory import MemorySaver

from scientis.agent.graph import build_workflow
from scientis.agent.state import AgentState, WorkflowConfig

logger = logging.getLogger(__name__)


class DiscoveryRunner:
    """Runs the scientific discovery workflow."""

    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self._memory = MemorySaver()
        self._workflow = build_workflow().compile(checkpointer=self._memory)

    async def run(
        self, question: str, session_id: Optional[str] = None, stream: bool = False
    ) -> dict[str, Any]:
        """Run the full discovery workflow.

        Returns the final AgentState dict.
        """
        if session_id is None:
            session_id = f"sess-{uuid.uuid4().hex[:12]}"

        initial_state: AgentState = {
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

        thread_config = {"configurable": {"thread_id": session_id}}

        try:
            if stream:
                result_state = initial_state
                async for event in self._workflow.astream(
                    initial_state, thread_config
                ):
                    for node_name, node_output in event.items():
                        result_state.update(node_output)
                        yield {"node": node_name, "state": dict(result_state)}
            else:
                result_state = await self._workflow.ainvoke(
                    initial_state, thread_config
                )
                return dict(result_state) if result_state else initial_state
        except Exception:
            logger.exception("Workflow failed for session %s", session_id)
            return {**initial_state, "status": "error", "error_message": str(Exception)}

    async def resume_with_review(
        self, session_id: str, review_decisions: list[dict], reviewer: str = ""
    ) -> dict[str, Any]:
        """Resume a paused workflow with human review decisions."""
        thread_config = {"configurable": {"thread_id": session_id}}

        update = {
            "review_decisions": review_decisions,
            "reviewed_by": reviewer,
        }

        result_state = await self._workflow.ainvoke(
            update, thread_config
        )
        return dict(result_state) if result_state else {}

    def run_sync(self, question: str, session_id: Optional[str] = None) -> dict[str, Any]:
        """Synchronous wrapper for run()."""
        result = None

        async def _run():
            nonlocal result
            final = await self.run(question, session_id)
            # If streaming generator, consume it
            if hasattr(final, "__aiter__"):
                state = {}
                async for event in final:
                    state.update(event.get("state", {}))
                result = state
            else:
                result = final

        asyncio.run(_run())
        return result or {}


# Singleton
_runner: Optional[DiscoveryRunner] = None


def get_runner() -> DiscoveryRunner:
    global _runner
    if _runner is None:
        _runner = DiscoveryRunner()
    return _runner
