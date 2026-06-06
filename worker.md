# MiniGrid Worker Scheduling Notes

## SFT Task Inventory

### 1. Original ACE_ONCE SFT Data Generation

Goal:

- Generate raw source data for updater SFT.
- Preserve full `results_*.json` and `llm_calls_*.jsonl`.

Configuration:

| Field | Value |
|---|---|
| Method | `ace_once_minigrid` |
| Model | `Qwen/Qwen3.5-27B` |
| Thinking | disabled |
| Games | 6 focused MiniGrid games |
| Seeds | `0..5` |
| Runs | 36 |
| Episodes / updater samples | 1020 |
| Output root | `minigrid_sft/` |

Issue found:

- Updater calls used the historical default `max_tokens=512`.
- Most updater labels were truncated with `finish_reason=length`.
- Audit result: `948 length`, `72 stop`.

Implication:

- The raw generator trajectories are still useful as logs.
- The updater targets are usually incomplete.
- The online ACE_ONCE playbook trajectory was affected after each truncated updater, so this data is not a clean online SFT source.

### 2. Offline Updater Relabel

Goal:

- Avoid rerunning generator.
- Reuse old updater prompts from `llm_calls`.
- Ask `Qwen/Qwen3.5-27B` to regenerate complete updater labels with a larger token budget.

Configuration:

| Field | Value |
|---|---|
| Input | updater prompts from original `minigrid_sft/llm_calls` |
| Model | `Qwen/Qwen3.5-27B` |
| Max tokens | 4096 |
| Tasks | 1020 updater prompts |
| Output root | `minigrid_updater_relabel/` |

Result:

```text
done=1020 failed=0 finished=1020/1020
finish_reason stop=981
finish_reason length=39
```

Implication:

- Good for repaired offline SFT labels.
- Does not repair the original online ACE_ONCE trajectory, because subsequent generator calls already used the old truncated playbook state.

### 3. Online ACE_ONCE SFT Rerun With Updater 8192

Goal:

- Rerun ACE_ONCE online so the updater is not truncated during the actual episode-by-episode playbook update loop.
- Produce the clean main SFT dataset.

Run tag:

```text
minigrid-sft-qwen35-27b-ace-once-updater8192-20260601_224522
```

Configuration:

| Field | Value |
|---|---|
| Method | `ace_once_minigrid` |
| Model | `Qwen/Qwen3.5-27B` |
| Thinking | disabled |
| Generator max tokens | 512 |
| Merged updater max tokens | 8192 |
| Games | 6 focused MiniGrid games |
| Seeds | `0..5` |
| Runs | 36 |
| Episodes / updater samples | 1020 |
| Output root | `minigrid_sft_updater8192/` |

Purpose:

- This is the primary clean SFT source once complete.
- It keeps the same game configs, episodes, rewards, and seed policy as the previous full SFT run, but fixes updater truncation during online execution.

### 4. Stability Run For Method Randomness

Goal:

- Estimate method-level stochasticity under a fixed environment setting.
- Do not measure environment seed variation here.

Script:

```text
stability.sh
```

Configuration:

| Field | Value |
|---|---|
| Method | `ace_once_minigrid` |
| Agent / generator model | `Qwen/Qwen3-8B` |
| Updater model | `Qwen/Qwen3.5-27B` |
| Thinking | disabled |
| Generator max tokens | 512 |
| Merged updater max tokens | 8192 |
| Temperature | 1.0 |
| Games | 6 focused MiniGrid games |
| Fixed seed | `STABILITY_SEED=0` |
| Repeats | 6 |
| Runs | `6 x 6 = 36` |
| Episodes | `(40 + 20 + 20 + 20 + 20 + 20) x 6 = 840` |
| Output root | `minigrid_stability_updater8192/` |
| Servers | `4 x A100` Qwen3-8B agent `dp=4` + `2 x H200` Qwen3-8B agent `dp=2` + `2 x H200` Qwen3.5-27B updater `dp=2` |
| A100 workers | `0-15%16`, 1 CPU / 4G each |
| H200 workers | `0-19%20`, 1 CPU / 4G each |
| Total concurrent workers | 36 |
| Total server GPUs | `4 x A100 + 4 x H200` |

Seed implementation:

```text
seed_offset = STABILITY_SEED * 10000
```

Default:

```text
STABILITY_SEED=0
seed_offset=0
```

Each game is repeated six times with the exact same seed:

```text
minigrid_memorys11-s0-r0
minigrid_memorys11-s0-r1
minigrid_memorys11-s0-r2
minigrid_memorys11-s0-r3
minigrid_memorys11-s0-r4
minigrid_memorys11-s0-r5
```

Purpose:

- Same game config.
- Same environment seed.
- Same episode sequence.
- Different independent LM/method executions at `temperature=1.0`.
- Use final reward distributions for per-game method randomness, e.g. violin plots.
- Previous submitted split-model run accidentally used `Qwen3.5-27B` as the agent and `Qwen3-8B` as the updater. Its data is still usable, but it is not the intended role split.
- The intended default submission uses three servers: one 4xA100 Qwen3-8B agent server, one 2xH200 Qwen3-8B agent server, and one 2xH200 Qwen3.5-27B updater server.
- The 36 fixed-seed runs are explicitly split across the two agent pools. MemoryS11, MemoryS13, SimpleCrossing, FourRooms, and DistShift repeats are divided across A100/H200 to avoid a single slow tail.
- The updated split is 16 A100 workers and 20 H200 workers. This keeps A100 conservative while giving H200 the higher request-level parallelism suggested by the previous utilization check.

Observed early utilization for the earlier Qwen3-14B single-model stability run:

```text
A100 server: 4 x A100, 24 workers total, 6 workers/GPU
GPU utilization: about 87-93%

H200 server: 2 x H200, 12 workers total, 6 workers/GPU
GPU utilization: about 39-46%
```

Takeaway:

- `6 workers/GPU` is enough to keep A100 very busy for Qwen3-14B in this online MiniGrid loop.
- `6 workers/GPU` is still underfeeding H200 for Qwen3-14B.
- For future 2xH200 Qwen3-14B stability-style runs, start closer to `10-12 workers/GPU` (`20-24` total H200 workers), then check `nvidia-smi`.
- Do not change the current run midstream unless there is a clear failure; it is healthy and producing work.

Progress checkpoint:

```text
runs_done=13 / 36
DistShift: mostly done; only repeat 4 at 10/20
Empty: all repeats active or done, roughly 62.5-100%
FourRooms: all 6 repeats done
MemoryS11: all repeats active, roughly 65-90%
MemoryS13: one repeat done, remaining repeats roughly 40-75%
SimpleCrossingS9N3: all repeats active, roughly 35-45%
```

Concurrency interpretation:

- Short/easy configs drain quickly; they do not determine the tail.
- `MemoryS11`, `MemoryS13`, and `SimpleCrossingS9N3` are the current stability-run tail.
- Equal workers-per-GPU is not equal throughput across GPU classes. H200 needs more request-level parallelism than A100 for Qwen3-14B in this online loop.
- Future stability runs should raise H200 worker concurrency while keeping A100 near the current level.

Memory stability change:

- `MemoryS11` uses 20 episodes instead of 40.
- `MemoryS13` uses 20 episodes instead of 30.
- `max_steps` stay original: `MemoryS11=605`, `MemoryS13=845`.
- This keeps reward semantics unchanged while reducing the slow Memory tail.

### Supporting Local Tools

| Tool | Purpose |
|---|---|
| `extract_minigrid_sft_data.py` | Convert `results + llm_calls` into episode-level updater SFT JSONL. |
| `plot_minigrid_sft_seed_vis.py` | Generate per-game SVG reward plots under `minigrid-sft-full/Vis`, with final cumulative reward mean/std annotations. |
| `worker.md` | Track scheduling lessons, bottlenecks, and task inventory. |

## SFT-Data Template

Current local SFT data root:

```text
minigrid-sft-updater8192-completed35/SFT-data/
```

Layout:

```text
SFT-data/
  minigrid_empty_random_5x5/minigrid_empty_random_5x5.jsonl
  minigrid_memorys11/minigrid_memorys11.jsonl
  minigrid_memorys13/minigrid_memorys13.jsonl
  minigrid_fourrooms/minigrid_fourrooms.jsonl
  minigrid_simplecrossings9n3/minigrid_simplecrossings9n3.jsonl
  minigrid_distshift1/minigrid_distshift1.jsonl
  summary.json
  spl_new.txt
```

Counts from the 35 completed online updater-8192 runs:

| Game | Records |
|---|---:|
| `minigrid_empty_random_5x5` | 240 |
| `minigrid_memorys11` | 200 |
| `minigrid_memorys13` | 180 |
| `minigrid_fourrooms` | 120 |
| `minigrid_simplecrossings9n3` | 120 |
| `minigrid_distshift1` | 120 |
| Total | 980 |

`MemoryS11` has 200 instead of 240 because the `seed=2` run was not part of the completed-35 archive.

Each JSONL row is one episode-level updater SFT sample:

```json
{
  "id": "minigrid_empty_random_5x5-s0-ep001",
  "messages": [
    {
      "role": "system",
      "content": "You are the ACE_ONCE updater for a MiniGrid agent..."
    },
    {
      "role": "user",
      "content": "{...JSON string with metadata, playbook, trajectory, feedback...}"
    },
    {
      "role": "assistant",
      "content": "{...raw merged ACE_ONCE updater output...}"
    }
  ],
  "metadata": {
    "game": "minigrid_empty_random_5x5",
    "seed": 0,
    "episode": 1,
    "env_id": "MiniGrid-Empty-Random-5x5-v0",
    "max_steps": 100,
    "num_generator_calls": 82,
    "updater_finish_reason": "stop",
    "target_parse_status": "raw_stop_parseable",
    "target_json_parseable": true,
    "updater_prompt_tokens": 12345,
    "updater_completion_tokens": 456,
    "updater_total_tokens": 12801,
    "final_reward": 1,
    "success": true,
    "source_model": "Qwen/Qwen3.5-27B",
    "disable_thinking": true,
    "source_run_dir": "...",
    "updater_llm_call_index": 83,
    "generator_llm_call_start_index": 1,
    "generator_llm_call_end_index": 82
  }
}
```

`messages[1].content` is itself a JSON string. Parsed shape:

```json
{
  "task_metadata": {
    "method": "ace_once_minigrid",
    "game": "minigrid_empty_random_5x5",
    "seed": 0,
    "episode": 1,
    "episode_position_in_run": 1,
    "total_episodes_in_run": 40,
    "env_id": "MiniGrid-Empty-Random-5x5-v0",
    "env_class": "MiniGridTextEnv",
    "valid_actions": ["left", "right", "forward", "pickup", "drop", "toggle", "done"],
    "max_steps": 100,
    "reward_rule": "Original MiniGrid reward is 1 - 0.9 * (step_count / max_steps) for success; failure or timeout gives 0."
  },
  "context_before_episode": {
    "type": "playbook",
    "playbook": []
  },
  "initial_observation": "...",
  "trajectory": [
    {
      "step": 0,
      "observation": "...",
      "generator_output": "...",
      "referenced_playbook_ids": [],
      "parsed_action": "forward",
      "action_parse_failed": false,
      "feedback": "...",
      "raw_env_reward": 0,
      "counted_reward": 0,
      "done": false
    }
  ],
  "actions": ["forward"],
  "episode_feedback": "...",
  "final_reward": 0,
  "success": false,
  "progress": {
    "current_episode": 1,
    "total_episodes": 40
  }
}
```

Empty playbook is represented as:

```json
"context_before_episode": {
  "type": "playbook",
  "playbook": []
}
```

Data quality audit for the current `SFT-data`:

```text
parsed_String_hits=0
structure_errors=0
finish_reason length=10 / 980
fallback_from_results=2 / 980
```

`fallback_from_results` means the raw updater `content` in `llm_calls` was empty, so the extractor rebuilt the assistant target from the same episode's parsed `results.reflection + results.delta_items`. The two fallback records are:

```text
minigrid_empty_random_5x5-s1-ep008
minigrid_memorys11-s0-ep021
```

The 10 updater calls with `finish_reason=length` are retained under the current raw-target policy:

```text
minigrid_distshift1-s2-ep003
minigrid_empty_random_5x5-s2-ep025
minigrid_empty_random_5x5-s3-ep030
minigrid_empty_random_5x5-s4-ep036
minigrid_empty_random_5x5-s5-ep014
minigrid_fourrooms-s1-ep003
minigrid_memorys11-s4-ep029
minigrid_memorys13-s2-ep015
minigrid_memorys13-s3-ep019
minigrid_memorys13-s3-ep021
```

## Current SFT Run

Run tag:

```text
minigrid-sft-qwen35-27b-ace-once-updater8192-20260601_224522
```

Purpose:

- ACE_ONCE-only MiniGrid SFT data generation.
- Generator calls stay at `max_tokens=512`.
- Merged updater calls use `max_tokens=8192`.
- Full `results_*.json` and `llm_calls_*.jsonl` are preserved.

Server layout:

| Server | GPUs | SGLang DP | Worker array | Concurrent workers | Workers / GPU |
|---|---:|---:|---|---:|---:|
| A100 | 4 x A100 | `dp=4` | `0-23%16` | 16 | 4 |
| H200 | 2 x H200 | `dp=2` | `0-11%8` | 8 | 4 |

## Game Timing Observations

The main tail driver is Memory, especially `MemoryS11`.

Current observed tail state:

```text
MemoryS11 remaining: seeds 0-5 still running
MemoryS13 remaining: seeds 4-5 still running
No pending tasks remain
```

At one checkpoint:

```text
Memory completed = 348 / 420
Memory remaining = 72
```

At the same stage, almost all non-Memory work was finished. This means the long tail is not evenly distributed across games; it is dominated by Memory runs.

Relative game cost from this run:

| Game | Relative Runtime | Notes |
|---|---|---|
| `minigrid_memorys11` | Highest | Worst tail. 40 episodes, `max_steps=605`, many long failed episodes. Give extra capacity. |
| `minigrid_memorys13` | High | 30 episodes, `max_steps=845`; fewer episodes than S11 but very long episodes. |
| `minigrid_simplecrossings9n3` | Medium | Some slow runs, but did not dominate tail like Memory. |
| `minigrid_fourrooms` | Low-medium | Usually finishes well before Memory. |
| `minigrid_empty_random_5x5` | Low | Finishes early. |
| `minigrid_distshift1` | Low | Finishes early. |

Important dependency:

- A single `game + seed` ACE_ONCE run is serial.
- Later episodes depend on the playbook/context produced by earlier updater calls.
- Do not split one seed's episode sequence across workers unless the code explicitly supports checkpointed playbook state.
- Starting another worker for the same `game + seed` risks duplicate writes and polluted `llm_calls`.

## GPU Utilization Observations

A100 server looked reasonably loaded:

```text
4 x A100, about 68-74% GPU utilization
memory used about 72GB / 80GB per GPU
```

Example:

```text
0, 68 %, 62 %, 72596 MiB, 81920 MiB
1, 72 %, 66 %, 72398 MiB, 81920 MiB
2, 72 %, 65 %, 72606 MiB, 81920 MiB
3, 74 %, 68 %, 72760 MiB, 81920 MiB
```

H200 server was underfed:

```text
2 x H200, about 32% GPU utilization
memory used about 125GB / 144GB per GPU
```

Example:

```text
0, 32 %, 25 %, 125634 MiB, 143771 MiB
1, 32 %, 25 %, 125192 MiB, 143771 MiB
```

Interpretation:

- H200 had enough model/KV memory allocated, but not enough concurrent requests to saturate compute.
- `4 workers / H200 GPU` is too conservative for this workload.
- A100 at `4 workers / GPU` was much healthier.

## Scheduling Takeaways

For future MiniGrid ACE_ONCE SFT runs:

1. Give Memory games about 2x scheduling weight.
2. Put MemoryS11 and MemoryS13 on both A100 and H200, not just one pool.
3. Increase H200 worker concurrency from `4 workers/GPU` to about `10-12 workers/GPU` for Qwen3-14B-style runs.
4. Keep A100 around `4-6 workers/GPU`, then tune based on `nvidia-smi`.
5. Use heavy-first ordering, but also reserve enough late capacity for Memory tails.
6. Avoid assigning all short games first if that leaves only Memory at the end.
7. Once a serial seed worker is running, extra workers cannot safely speed up its later episodes.

Suggested next default:

| Pool | GPUs | Initial workers / GPU | Total concurrent workers |
|---|---:|---:|---:|
| A100 | 4 | 5-6 | 20-24 |
| H200 | 2 | 10-12 | 20-24 |

If H200 utilization remains below 60%, raise H200 worker concurrency again.

If A100 utilization is already above 70%, do not raise A100 much unless queue pressure is low.

## Monitoring Commands

GPU utilization:

```bash
srun --jobid=<A100_SERVER_JOB> --overlap nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv
srun --jobid=<H200_SERVER_JOB> --overlap nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv
```

Memory tail progress:

```bash
cd /gscratch/h2lab/mohanc3/projects/ERL_repo
TAG=minigrid-sft-qwen35-27b-ace-once-updater8192-20260601_224522

for spec in minigrid_memorys11:40 minigrid_memorys13:30; do
  game=${spec%%:*}
  total=${spec##*:}
  for seed in 0 1 2 3 4 5; do
    d=$(find minigrid_sft_updater8192 -maxdepth 1 -type d -name "${TAG}-*-ace_once_minigrid-${game}-s${seed}" | head -n 1)
    done_ep=0
    latest="NA"
    if [ -n "$d" ]; then
      f=$(find "$d" -name "llm_calls_*.jsonl" | head -n 1)
      if [ -n "$f" ]; then
        done_ep=$(grep -F -c "Your job has two parts, done in a single pass" "$f" 2>/dev/null)
        [ "$done_ep" -gt "$total" ] && done_ep="$total"
        latest=$(grep -oE "Step: [0-9]+/[0-9]+" "$f" 2>/dev/null | tail -n 1)
      fi
    fi
    left=$((total - done_ep))
    [ "$left" -le 0 ] && continue
    printf "%-20s seed=%s episodes=%-2s/%-2s left=%-2s latest_step=%s\n" "$game" "$seed" "$done_ep" "$total" "$left" "${latest:-NA}"
  done
done | sort
```
