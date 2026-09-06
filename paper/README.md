# The paper

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`main.pdf` is tracked, so it can be read without a TeX install.
`main.tex.orig` is the template as it stood before the results existed, kept
only so the diff is auditable.

## Which figures the paper actually uses

| file | used? |
|---|---|
| `figures/fig_propagation.pdf` | **yes** — generated from the 87 prediction files, 3 folds x 3 seeds |
| the TikZ pipeline diagram | **yes** — drawn inline in `main.tex`, no file |
| `figures/fig1_map.pdf` | **no, and it is stale** |
| `figures/fig2_event_timeline.pdf` | **no, and it is stale** |
| `figures/fig3_event_impact.pdf` | **no, and it is stale** |
| `figures/fig4_weather.pdf` | **no, and it is stale** |

The four stale figures were drawn from an earlier event table of 149 fixtures at
AT&T Park, Oakland Coliseum and Oracle Arena. The current `events.csv` holds 95
fixtures at four venues - SAP Center (69), Shoreline Amphitheatre (12), Avaya
Stadium (9), Levi's Stadium (5) - and the weather source moved from Open-Meteo
to two ASOS stations. They are kept rather than deleted because the plotting
work is reusable, but **nothing in them should be quoted**. Regenerating them
against the current data is a real, unclaimed task.

`results/figures/spacetime_101-N_*.png` in the repo root is the space-time
diagram. It is not in the paper yet - it would cost about 0.4 of a page.
