# Complete Visuals Summary - 2026-05-19

This folder collects only locally available visuals generated from complete data.

Included:

1. `01_online_40state_variants`
   - Online 40-state comparison of original `notebook_minimal`, `notebook_minimal_thinkahead`, and `notebook_minimal_mechanism`.
   - Includes FrozenLake, Sokoban, and combined running-accuracy plots.

2. `02_y8z20_original_vs_updated_ablation`
   - Complete y8z20 original-vs-updated ablation visuals from seed 20260509.
   - Includes overlap violin plots and median/std line plots.

3. `03_latest_y8z20_baseline_passk`
   - Complete latest y8z20 baseline passk visuals.
   - Includes violin and median/std plots for combined, FrozenLake, and Sokoban.

Excluded:

- `notebook_variants_passk_y8z20_seed20260509_plots_PARTIAL_LOCAL_DATA`
  - Excluded because local shard files are incomplete and marked partial.

Notes:

- `latest_y8z20_analysis_extract/visualizations` appears to duplicate `latest_y8z20_mixed_shards_visualizations`; the mixed-shards folder is used as the canonical copy here.
