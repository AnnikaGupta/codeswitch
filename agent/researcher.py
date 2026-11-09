"""A Researcher: fresh context per dispatch, one tool (query_benchmark), and
a narrow sub-question. Returns a cited finding -- never a bare number without
saying which rows/conditions/models it came from."""

import json

import litellm

from agent.config import MAX_RESEARCHER_ROUNDS, RESEARCHER_MODEL
from agent.db_tool import QUERY_TOOL_SCHEMA, SCHEMA_DESCRIPTION, query_benchmark

SYSTEM_PROMPT = f"""\
You are a Researcher in a fairness-research agent system. You investigate ONE narrow
sub-question about a benchmark of chatbot responses to casual requests, where a demographic
identity (ethnicity, gender, or religion) was sometimes mentioned even though it's irrelevant
to the task.

{SCHEMA_DESCRIPTION}

Rules:
- Use query_benchmark to answer. Don't guess numbers -- compute them.
- Prefer stats_summary for anything about whether a difference is "real" (statistically
  supported after Bonferroni correction) vs. just an observed difference in a small sample.
  If stats_summary doesn't have bonferroni_sig=true for something, say so explicitly rather
  than implying significance.
- Pull 1-2 verbatim example response_text values to illustrate your finding, and cite them
  with their condition_id, model, and template_id.
- If the data can't answer your sub-question (e.g. too few rows, or it's asking something
  outside what was measured), say that plainly instead of stretching an answer.
- When you're done, reply with PLAIN TEXT (no tool call) summarizing your finding: the
  answer, the evidence (numbers + example quotes with citations), and whether it's
  statistically supported or just descriptive.
"""


def run_researcher(question: str, agent_label: str) -> tuple[str, list[dict]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_calls_log: list[dict] = []

    for _ in range(MAX_RESEARCHER_ROUNDS):
        reply = litellm.completion(
            model=RESEARCHER_MODEL,
            messages=messages,
            tools=[QUERY_TOOL_SCHEMA],
            temperature=1,  # claude-haiku-5-5 rejects any other value
        ).choices[0].message
        messages.append(reply.model_dump())

        if not reply.tool_calls:
            return reply.content or "", tool_calls_log

        for call in reply.tool_calls:
            args = json.loads(call.function.arguments)
            result = query_benchmark(args.get("sql", ""))
            tool_calls_log.append({"agent": agent_label, "name": "query_benchmark", "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return "Hit the query-round limit before reaching a conclusion on this sub-question.", tool_calls_log
