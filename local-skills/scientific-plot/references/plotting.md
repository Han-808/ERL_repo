# Scientific Plotting Reference

## Figure Planning

- Start from the scientific question, not the chart type.
- State what each visual encoding means: x/y position, color, marker, line style, area, and alpha.
- Make units explicit in axis labels and captions.
- Avoid decorative effects that compete with the data.

## ML/RL Experiment Plots

- Plot individual seeds faintly when the number of seeds is small enough to inspect.
- Plot the mean or median as the main line only after deciding which statistic is scientifically appropriate.
- Use standard error or confidence interval bands for uncertainty. Use standard deviation only when the goal is to show population spread.
- Report the number of seeds/runs in the caption or legend.
- Use shared x/y scales for method comparisons unless the point is to show scale differences.
- Smooth curves only for readability. Keep raw or lightly transparent traces available, and state the smoothing window.

## Error Bars And Intervals

- Do not call standard deviation a confidence interval.
- For paired experiments, compute paired differences and visualize the distribution or paired confidence interval.
- Prefer bootstrap confidence intervals when assumptions are unclear and sample sizes permit.
- Show exact points where possible for small n.

## Styling

- Use colorblind-safe palettes such as Okabe-Iito.
- Use direct labels for a small number of lines; use legends for larger sets.
- Keep text large enough for the final publication size, not just for the notebook view.
- Export vector formats for line art, but inspect exported PDF/SVG for font and clipping issues.

## Checklist Before Finalizing

- Axis labels include units.
- Tick labels are legible at final size.
- Legend does not cover important data.
- Color and line style remain distinguishable in grayscale.
- Y-axis baseline and log scale choices are scientifically justified.
- Figure can be regenerated from committed code or notebook cells.
