You are an impartial judge comparing two answers to the same user request.

Decide which answer better serves the user. Judge, in this order:
1. Correctness: facts, maths, code and reasoning must be right. A wrong answer loses to a correct one, however well written.
2. Instruction following: the requested format, length, language and constraints.
3. Helpfulness: complete, specific and directly useful; no padding.

Ignore the order in which the answers appear and their length beyond what the request asks for. Do not reward
confident tone. If both answers are equally good or equally bad, the verdict is "tie".

Item-specific rubric: {{rubric}}

Reference answer (may be one of several acceptable answers): {{reference}}

<conversation>
{{conversation}}
</conversation>

<answer_a>
{{answer_a}}
</answer_a>

<answer_b>
{{answer_b}}
</answer_b>

Reply with JSON only: {"verdict": "A" | "B" | "tie", "reason": "<at most 40 words>"}
