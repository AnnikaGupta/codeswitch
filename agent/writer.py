"""The Writer: turns the Lead's collected Researcher findings into one final,
cited answer. Has its own query_benchmark access, used sparingly, mainly to
pull a clean verbatim example quote to illustrate a point."""

import json

import litellm

from agent.config import MAX_WRITER_ROUNDS, WRITER_MODEL
from agent.db_tool import QUERY_TOOL_SCHEMA, query_benchmark

SYSTEM_PROMPT = """\
You are the Writer in a fairness-research agent system. You receive a user's question and a
set of Researcher findings (each already cited to specific rows/conditions/models) and write
the final answer.

Rules:
- Organize the answer around the user's actual question, not around the Researchers' order.
- Explicitly separate three tiers of claim:
  1. Statistically supported (the finding cited a stats_summary row with bonferroni_sig=true).
  2. Descriptive pattern (a real difference observed, but not Bonferroni-significant, or too
     small a sample) -- say it's suggestive, not proven.
  3. What the data cannot establish -- e.g. you have no proprietary post-training checkpoints,
     so you can ONLY make correlational claims across the two subject models studied (claude,
     gpt), never a causal claim about why any one vendor's model behaves a certain way.
- Include 1-3 verbatim example quotes (use query_benchmark if you need a clean one) with their
  condition/model/template cited inline.
- If the Researcher findings don't actually cover what the user asked, say so plainly instead
  of papering over the gap.
- Keep it tight: a few short paragraphs or a short list, not an essay.
"""


def run_writer(question: str, findings: list[str]) -> tuple[str, list[dict]]:
    findings_block = "\n\n".join(findings) if findings else "(no findings)"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Original question: {question}\n\nResearcher findings:\n{findings_block}"},
    ]
    tool_calls_log: list[dict] = []

    for _ in range(MAX_WRITER_ROUNDS):
        reply = litellm.completion(
            model=WRITER_MODEL,
            messages=messages,
            tools=[QUERY_TOOL_SCHEMA],
            temperature=1,  # claude-sonnet-5-5 rejects any other value
        ).choices[0].message
        messages.append(reply.model_dump())

        if not reply.tool_calls:
            return reply.content or "", tool_calls_log

        for call in reply.tool_calls:
            args = json.loads(call.function.arguments)
            result = query_benchmark(args.get("sql", ""))
            tool_calls_log.append({"agent": "writer", "name": "query_benchmark", "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return "Hit the tool-call limit while writing up findings.", tool_calls_log
