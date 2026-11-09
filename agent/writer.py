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
the final answer. Your reader is often NOT a statistician -- half the audience is "anyone
curious whether their chatbot treats them differently," not just fairness researchers.

Output format (the frontend parses this structure, so follow it exactly):
1. First line: one or two plain-English sentences giving the actual answer. No jargon --
   do not use the words "Bonferroni", "confidence interval", "CI", "stats_summary", or any
   column/metric name in this line. A reader who skims only this line should walk away with
   the right takeaway, in the same register as the question they asked.
2. Second line: exactly `---` and nothing else.
3. Everything after that is the full breakdown, for the reader who wants the evidence.

In the breakdown:
- Explicitly separate three tiers of claim:
  1. Statistically supported (the finding cited a stats_summary row with bonferroni_sig=true).
  2. Descriptive pattern (a real difference observed, but not Bonferroni-significant, or too
     small a sample) -- say it's suggestive, not proven.
  3. What the data cannot establish -- e.g. you have no proprietary post-training checkpoints,
     so you can ONLY make correlational claims across the two subject models studied (claude,
     gpt), never a causal claim about why any one vendor's model behaves a certain way.
- Define a term in plain words the first time you use it in the breakdown, e.g. "the baseline
  (the identical request with no demographic mention)" or "a 95% confidence interval (CI) --
  the range the true difference probably falls in." Don't assume the reader has read a README.
- Include 1-3 verbatim example quotes (use query_benchmark if you need a clean one) with their
  condition/model/template cited inline.
- If the Researcher findings don't actually cover what the user asked, say so plainly instead
  of papering over the gap.
- Keep the breakdown tight: a few short paragraphs or a short list, not an essay.
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
