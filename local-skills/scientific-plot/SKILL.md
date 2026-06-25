---
name: scientific-plot
description: Create, repair, and polish scientific plots for papers, reports, notebooks, and experiment analysis. Use for matplotlib, seaborn, pandas, numpy, or notebook plotting tasks involving publication-quality figures, ML/RL learning curves, ablations, error bars, confidence intervals, multi-panel figures, log scales, colorblind-safe palettes, figure export, or visual checks of scientific chart readability.
---

# Scientific Plot

Use this skill when producing or improving scientific figures.

## Workflow

1. Identify the figure purpose: comparison, trend, distribution, relationship, composition, or diagnostic.
2. Inspect the data shape and units before plotting. Preserve raw values and compute summaries in explicit named variables.
3. Choose a plot type that matches the claim:
   - Trends: line plot with confidence interval or standard error band.
   - Comparisons: dot/box/violin/bar only when aggregation is clear.
   - Distributions: histogram, KDE, ECDF, box, or violin.
   - Correlations: scatter with density/alpha and fitted trend only when justified.
   - Matrices/images: heatmap with labeled colorbar and fixed aspect when appropriate.
4. Apply publication-safe styling: readable fonts, direct labels where useful, colorblind-safe colors, light gridlines, no unnecessary chart ink.
5. Export deterministic artifacts: PDF/SVG for vector use, PNG for quick inspection, and keep the script/notebook cell that generated them.
6. Visually verify the output for clipped labels, overlapping legends, misleading axes, unreadable text, and incorrect uncertainty semantics.

## Bundled Helpers

Use `scripts/plot_style.py` when a Python plotting helper is useful. It provides:

- Okabe-Ito colorblind-safe palette constants.
- `set_paper_style()` for matplotlib defaults.
- `figure_size()` for common single-column, double-column, and square layouts.
- `savefig_all()` to export PDF, SVG, and PNG variants consistently.

Example:

```python
from pathlib import Path
import matplotlib.pyplot as plt
from plot_style import OKABE_ITO, set_paper_style, figure_size, savefig_all

set_paper_style()
fig, ax = plt.subplots(figsize=figure_size("single", height_ratio=0.70))
ax.plot(x, mean, color=OKABE_ITO["blue"], label="method")
ax.fill_between(x, lo, hi, color=OKABE_ITO["blue"], alpha=0.18, linewidth=0)
ax.set_xlabel("Environment steps")
ax.set_ylabel("Return")
ax.legend(frameon=False)
savefig_all(fig, Path("figures/learning_curve"))
```

For deeper plotting choices and quality checks, read `references/plotting.md`.

## Standards

- Use explicit axis labels with units.
- Do not truncate axes unless the visual clearly marks the truncation.
- Do not use dual y-axes unless there is no clearer alternative.
- Show uncertainty and sample sizes when comparing experimental results.
- Use log scales only when they answer the scientific question; label ticks clearly.
- Keep legends outside dense data regions, or replace them with direct labels.
- Prefer vector export for line art and text-heavy figures.
- Make plots reproducible from code; avoid manual edits that cannot be regenerated.
