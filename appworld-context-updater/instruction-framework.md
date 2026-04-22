# Instruction

We want a harness for context updates in AppWorld experiments.

Inputs:
1. Previous context
2. Current task instruction
3. Task trace, including code and execution results
4. Evaluation feedback / test report

Output:
- Updated context after the task

Methods to compare:
1. Summarize-Rewrite: summarize the previous context and current trajectory into a new full context
2. ACE reproduction: one model extracts takeaways from the trace, and one model generates context delta operations
3. Hypothesis-driven notebook: one model extracts observations from the trace, one model proposes what the agent should investigate in future tasks to learn more about the environment, and one model updates the notebook items

We run the harness over the 90 tasks in the `train` split.

Implementation structure:
- One shared runner (`run.py`) and shared utilities (`common.py`)
- One directory per method
- One `model.py` file per method directory
- Each method directory stores its own outputs such as `context.jsonl`

The hypothesis we want to test is that prior context update methods mostly build passive memory from past traces, while a stronger agent should also learn how to interact with the world proactively: use trial and error, slow down when mistakes are costly, move quickly when a pattern is already verified, form hypotheses about earlier failures, and verify those hypotheses when similar situations appear again.

References:
- Static playbook evaluation: `/mmfs1/home/chentong/ws/continual-learning-agent/scripts/appworld-eval/eval_static_playbook.py`
- ACE agent setup: `/mmfs1/home/chentong/ws/continual-learning-agent/scripts/appworld-eval/eval_static_playbook.py`
- Agent prompt: `/mmfs1/home/chentong/ws/continual-learning-agent/libs/ace-appworld/experiments/prompts/appworld_react_code_agent_playbook_generator_prompt.txt`
