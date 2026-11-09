"""Top-level entry point app.py calls -- mirrors Project 1's run_agent(messages)
shape: takes the session's message history, returns (response_text, tool_calls).

Unlike Project 1 (one continuous model conversation), this orchestrates three
separate agent shapes (Lead -> Researchers in parallel -> Writer), each with
its own fresh context. Prior turns in the session are folded into the
question text given to the Lead, so follow-ups still have continuity."""

from agent.lead import run_lead
from agent.writer import run_writer


def _build_question_with_context(messages: list[dict]) -> tuple[str, str]:
    """Returns (latest_user_message, question_with_prior_context)."""
    latest_user_message = messages[-1]["content"]
    prior = messages[:-1]
    lines = [f"{m['role']}: {m['content']}" for m in prior if m.get("role") in ("user", "assistant")]
    if not lines:
        return latest_user_message, latest_user_message
    context = "Prior conversation in this session:\n" + "\n".join(lines)
    return latest_user_message, f"{context}\n\nCurrent question: {latest_user_message}"


def run_agent(messages: list[dict]) -> tuple[str, list[dict]]:
    latest_user_message, question = _build_question_with_context(messages)

    lead_result = run_lead(question)
    if lead_result["needs_clarification"]:
        return lead_result["clarification_text"], lead_result["tool_calls"]

    answer, writer_tool_calls = run_writer(latest_user_message, lead_result["findings"])
    tool_calls = lead_result["tool_calls"] + writer_tool_calls
    return answer, tool_calls
