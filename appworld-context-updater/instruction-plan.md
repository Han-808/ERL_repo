Goal: repeatedly improve the hypothesis-driven method script through an iterative versioned cycle.

Scope
- Start from the current hypothesis-driven method baseline.
- Continue creating new versions as:
  - `hypothesis_v2`
  - `hypothesis_v3`
  - `hypothesis_v4`
  - ...
- Each new version must be based on evidence from the previous run, not guesswork.

Inputs to review
- Read all prior outputs under `outputs/`.
- Focus especially on:
  - `outputs/<method-name>/traces.jsonl`
  - `outputs/<method-name>/latest_context.txt`
- Read and understand the design document:
  - `/mmfs1/home/chentong/ws/continual-learning-agent/scripts/appworld-context-updater/readme.md`

Implementation requirement
- Every time you create a new method variant, you must add it to `run.py` so it can be executed through the standard runner.

How to run each variant
- Use the following command pattern to run a variant:
```bash
source /mmfs1/home/chentong/ws/continual-learning-agent/libs/ace/.venv/bin/activate;
CUDA_VISIBLE_DEVICES=0 \
python /mmfs1/home/chentong/ws/continual-learning-agent/scripts/appworld-context-updater/run.py \
  --method "hypothesis_vN" \
  --model-name Qwen/Qwen3-8B \
  --context-model-name gpt-5.4 \
  --max-steps 10
```
- Replace `hypothesis_vN` with the current version name.
- Run each version for 1 hour before analyzing results and creating the next version.

Required loop
For each version `hypothesis_vN`:
1. Implement the new version in code (v2 is already implemented but not evaluated, so you should start with eval v2).
2. Add the new method to `run.py`.
3. Run that version for 1 hour using the standard command above (kill if not finish).
4. Analyze the outputs from that run.
5. Compare the results against:
   - earlier hypothesis versions
   - other existing methods
   - the intended design in the README
   - ACE-style methods, especially where they do better or worse
6. Identify the main failure modes, weak assumptions, and missed opportunities.
7. Propose concrete improvements based on the evidence.
8. Implement those improvements as the next version, `hypothesis_v(N+1)`.

Rules
- Do not stop after one improvement.
- Keep iterating: run, analyze, improve, implement the next version.
- Every version change must be justified by evidence from traces, contexts, or evaluation results.
- Prefer targeted changes to the update pipeline, prompts, notebook structure, integration logic, or decision rules over vague rewrites.

Reporting
- Maintain a Markdown report at `report.md` and update it after each version run.
- For each variant, add a short section that includes: the measured accuracy, what is new in this variant, the main error analysis, what improved compared with the previous version, what did not improve or regressed, and the specific evidence or example traces that support those conclusions.
- Keep each variant section concise but concrete, so the report makes it easy to compare versions side by side and understand why the next version was created.

Expected output for each iteration
- A new method version in code.
- A runnable entry in `run.py`.
- A completed 1-hour run for that version.
- A short analysis of what failed and why.
- A short explanation of what changed in the new version.
- Evidence from the previous run that motivated the change.
- An updated `report.md` entry for that variant.

End condition
- Continue the cycle until explicitly stopped.
