# Single-turn pass@k result audit

Result folder:
`pass@k-singleturn_8b/passk-qwen3-8b-nothink-k40-y8-z10-35122524-combined`

## Run sanity

- Complete run: 80 state rows, 640 sample rows, 6400 rollouts.
- Each environment has states 1 through 40.
- Each state has 8 independent samples and 80 games total.
- Each sample has 10 games.
- All `reference_prompt_matches_template` values are `True`.
- Rollout keys are unique and seeds have no duplicates.
- No zero-step or empty-action rollouts were found.
- The few "move failed" strings are normal Sokoban environment feedback, not infrastructure errors.

## Main trend

The expected stability signal would be a clear downward trend in
`std_sample_pass_rate` as `state_x` increases.

This run does not show a strong version of that trend.

FrozenLake:

- Average mean pass rate increases from 0.2713 in states 1-10 to 0.3588 in states 31-40.
- Mean pass rate has a positive state correlation: 0.4501.
- Average sample std decreases from 0.1654 in states 1-10 to 0.1225 in states 11-20, but rises again to 0.1464 in states 31-40.
- Std correlation with state is weakly negative: -0.1452.
- Interpretation: performance improves, but sample stability only weakly improves and is not monotonic.

Sokoban:

- Average mean pass rate is roughly flat to slightly worse late: 0.3225 in states 1-10, 0.3600 in states 11-20, 0.3638 in states 21-30, then 0.2913 in states 31-40.
- Mean pass rate has weak negative state correlation: -0.1010.
- Average sample std stays nearly flat: 0.1504, 0.1420, 0.1546, 0.1491 across each 10-state block.
- Std correlation with state is almost zero: -0.0108.
- Interpretation: no clear convergence/stability trend.

## Outliers and abnormal states

FrozenLake:

- State 1 is a high-variance outlier by IQR: mean 0.2625, std 0.2326, sample range 0.0 to 0.7.
- State 6 is also very volatile: mean 0.2750, std 0.2252, range 0.0 to 0.6.
- State 21 has strong mean but high variance: mean 0.4625, std 0.2066, range 0.1 to 0.8.
- State 35 is the best mean pass rate: 0.4750.
- State 40 is the worst mean pass rate: 0.1625, with two zero-pass samples.

Sokoban:

- No formal IQR std outlier was found.
- State 3 has the highest std: mean 0.3375, std 0.2264, range 0.0 to 0.7.
- State 18 has very high mean and high std: mean 0.4750, std 0.2188, range 0.2 to 0.9.
- State 15 is the best mean pass rate: 0.4875.
- State 6 is the worst mean pass rate: 0.1625.
- Late states 31-40 are lower than the middle of training, despite larger notebooks.

## Notebook behavior

FrozenLake:

- 26 unique notebook hashes across 40 states.
- Notebook size grows from 11 lines to a max of 25, then ends at 18.
- There are repeated notebook plateaus, for example states 2-4, 6-8, 11-14, and 34-36.

Sokoban:

- 33 unique notebook hashes across 40 states.
- Notebook size grows from 11 lines to a max of 38, then ends at 25.
- There are some short repeated plateaus, but less repetition than FrozenLake.

Interpretation: the notebooks sometimes stabilize textually for a few states, but this does not reliably translate into lower pass-rate variance.

## Supplementary plot

`plot_state_summary.py` is a useful supplemental plotting script because it plots
`mean_sample_pass_rate` and `std_sample_pass_rate` by `state_x` for each environment.

Matplotlib was unavailable locally and on Hyak, so the original PNG generation was skipped.
A no-matplotlib SVG equivalent was generated at:

`plots/state_accuracy_by_env.svg`

