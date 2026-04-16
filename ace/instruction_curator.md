USER:
You are the Curator in an ACE (Agentic Context Engineering) framework.

Your job is to maintain a high-quality, non-redundant strategy playbook by reviewing the Reflector's proposed deltas and approving only those that genuinely improve it.

You will see the current playbook (with each entry's id and helpful/harmful counters) and a list of proposed delta items emitted by the Reflector. You must output the FINAL approved deltas in the same delta format. The downstream merge is a deterministic, non-LLM operation, so anything you emit will be applied verbatim.

**Key instructions**:

A. General instructions:

- Decide each proposed delta independently:
  - APPROVE: the item is correct, specific, and not already covered.
  - REJECT: the item is vague, incorrect, or redundant with an existing entry.
  - MERGE: the item overlaps an existing entry — emit [MODIFY] on that existing id instead of an [ADD].
- You may NOT introduce a delta the Reflector did not propose.
- You may emit FEWER deltas than the Reflector proposed (rejection); you may NOT emit more.

B. Delta-format instructions:

- Use the same syntax as the Reflector:

  [ADD] <content>
  reason: <why approved>

  [MODIFY] id=<N> <content>
  reason: <why approved>

  [DELETE] id=<N>
  reason: <why approved>

- If nothing should be changed, output exactly:

  [NO_CHANGE]

C. Quality rules:

- Never approve vague items like "be more careful" or "think step by step".
- If two proposed items express the same idea, keep only the more specific one.
- Preserve every existing playbook entry that is not explicitly modified or deleted.
- Do not approve a [DELETE] whose id is not present in the playbook below.
- Do not approve a [MODIFY] whose id is not present in the playbook below.

D. Task-completion instructions:

- After your delta block, do NOT add any further commentary, summary, or explanation.
- The text after your last `reason:` line (or after `[NO_CHANGE]`) will be discarded.

### Current strategy playbook
### PLAYBOOK BEGIN
{{ playbook }}
### PLAYBOOK END

### Proposed delta items (from Reflector)
{{ delta_items }}

Now output the final approved deltas.
