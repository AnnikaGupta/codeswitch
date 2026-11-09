"""The Lead: plans the investigation, fans out Researchers in parallel (fresh
context each), and decides when enough evidence is gathered to hand off to
the Writer. Also the one that asks for clarification on a vague question."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm

from agent.config import LEAD_MODEL, MAX_LEAD_ROUNDS
from agent.db_tool import SCHEMA_DESCRIPTION
from agent.researcher import run_researcher

DISPATCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "dispatch_researcher",
        "description": "Spin up a fresh Researcher sub-agent (no shared memory with you) to "
                        "investigate one narrow, concrete sub-question against the benchmark "
                        "database. Call this multiple times in the SAME turn to fan several "
                        "Researchers out in parallel across different angles (e.g. different "
                        "demographic axes, different metrics, different subject models).",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "A single, concrete sub-question for the Researcher to answer.",
                },
            },
            "required": ["question"],
        },
    },
}

SYSTEM_PROMPT = f"""\
You are the Lead of a fairness-research agent system studying demographic-conditioned
conversational style shift: whether chatbots talk differently once a user mentions an
ethnicity, gender, or religion that is irrelevant to the task, versus an identical request
with no demographic mention (the baseline).

{SCHEMA_DESCRIPTION}

You do not query the database yourself. You plan and delegate.

Your job each turn:
1. If the user's question is too vague to decompose into concrete sub-questions (e.g.
   "is the model biased?", "tell me about this"), do NOT dispatch anything. Reply with
   plain text asking exactly one clarifying question that would let you proceed (e.g. which
   demographic axis, which metric -- register/gender/narrowing/refusal, which model, or
   whether they want a broad overview). Keep it short.
2. Otherwise, call dispatch_researcher one or more times in this SAME turn to fan out
   Researchers in parallel, each on one narrow angle (e.g. one per demographic axis, or one
   per metric, or one per subject model). Don't dispatch more than 4 at once.
3. After a round of findings comes back, decide: do the findings leave a real gap (e.g. one
   axis wasn't covered, a surprising finding needs a follow-up check, results conflict)? If
   so, dispatch another round. If the findings already cover the question, reply with a
   short plain-text note (no tool call) saying you have enough -- a Writer will take it from
   here. You have at most {MAX_LEAD_ROUNDS} rounds total, so don't over-dispatch.

Never fabricate a finding yourself -- every claim in the final answer must trace back to a
Researcher's cited result.
"""


def run_lead(user_message: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    tool_calls_log: list[dict] = []
    all_findings: list[str] = []
    researcher_counter = 0

    for round_idx in range(MAX_LEAD_ROUNDS):
        reply = litellm.completion(
            model=LEAD_MODEL,
            messages=messages,
            tools=[DISPATCH_TOOL_SCHEMA],
            temperature=1,  # claude-sonnet-5-5 rejects any other value
        ).choices[0].message
        messages.append(reply.model_dump())

        if not reply.tool_calls:
            if round_idx == 0 and not all_findings:
                return {
                    "needs_clarification": True,
                    "clarification_text": reply.content or "Could you say more about what you'd like to explore?",
                    "findings": [],
                    "tool_calls": tool_calls_log,
                }
            break  # Lead decided it has enough; move on to the Writer.

        dispatches = reply.tool_calls
        results_by_call_id = {}
        with ThreadPoolExecutor(max_workers=max(len(dispatches), 1)) as pool:
            futures = {}
            for call in dispatches:
                args = json.loads(call.function.arguments)
                researcher_counter += 1
                label = f"researcher_{researcher_counter}"
                futures[pool.submit(run_researcher, args.get("question", ""), label)] = (call, args, label)

            for future in as_completed(futures):
                call, args, label = futures[future]
                finding, sub_tool_calls = future.result()
                tool_calls_log.append({
                    "agent": "lead", "name": "dispatch_researcher", "args": args, "result": finding,
                })
                tool_calls_log.extend(sub_tool_calls)
                all_findings.append(f"[{label}] Q: {args.get('question')}\nFinding: {finding}")
                results_by_call_id[call.id] = finding

        for call in dispatches:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": results_by_call_id[call.id]})

    return {
        "needs_clarification": False,
        "clarification_text": None,
        "findings": all_findings,
        "tool_calls": tool_calls_log,
    }
