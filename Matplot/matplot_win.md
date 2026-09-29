# Matplotlib Viewer Module

The **Matplotlib Viewer** module renders high-resolution publication-quality 2D plots using Matplotlib and Seaborn directly inside the application canvas. It receives plot packets emitted from upstream nodes (such as `Lineplot`, `Histogram`, or analysis modules) via `CMD_DICT`.

---

## Technical Specifications

- **Accepted Inputs**:
  - `IOTypes.CMD_DICT`: Plot packet with `action: "plot_data"`, `series: [{"label": ..., "x": [...], "y": [...]}], title, xlabel, ylabel, x_lim, y_lim, log_x`.
- **Outputs**:
  - `Frame`: `IOTypes.FRAME` (Emits rendered BGR video frames of the plot canvas).

---

## Key Features & Controls

- **Seaborn Styling**: Automatically formats plots using Seaborn (`darkgrid` style, `muted` palette).
- **Automatic Aggregation & Error Bands**: Detects repeated observations at identical X coordinates and renders standard deviation ($\pm 1\sigma$) shaded error bands.
- **Logarithmic Scaling**: Respects and applies logarithmic scale transformations when passed from upstream modules.
- **Configurable Canvas & DPI**: Adjust resolution (e.g. 800x600, 1920x1080) and dots-per-inch (DPI) on the fly via the `Advanced Options` modal.
- **Clipboard Export**: One-click copying of the high-resolution rendered graphic directly to the system clipboard.
- **Frame Emission**: Broadcasts the rendered graphic as a native `IOTypes.FRAME` (BGR) to downstream video recorders or image viewports.
