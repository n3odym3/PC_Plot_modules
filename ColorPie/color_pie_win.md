# Color Pie Plot Module

The **Color Pie Plot** module provides circular (pie chart) visualization of categorical color distributions. It is designed to consume color classification metrics (from modules like `ColorMapper`) and render publication-ready diagrams with percentage overlays.

---

## Technical Specifications

- **Accepted Inputs**:
  - `IOTypes.CMD_DICT`: Command dictionary with `action: "plot_pie"` and `distribution: {category: {"count": N, "color": [R, G, B], "percent": P}}`.
- **Outputs**:
  - `Frame`: `IOTypes.FRAME` (Emits rendered BGR frame of the pie plot to video recorders or image viewers).

---

## Key Features & Controls

- **Automatic Color Mapping**: Uses the exact RGB colors assigned to each categorical color group by upstream image classification modules.
- **Percentage Labels**: Automatically computes and annotates relative wedge percentages on the chart.
- **Dark Theme Styling**: Premium dark background (`#1e1e1e`) matching the application aesthetic with subtle wedge borders.
- **Exporting Options**:
  - **Copy Image**: Copies the current pie chart image directly to the system clipboard.
  - **Export TSV**: Copies categorical counts and percentages formatted as Tab-Separated Values for immediate pasting into spreadsheet applications.
  - **Save**: Opens an interactive file dialog to save the plot as a PNG image file.
