# Histogram Module

The **Histogram** module computes, analyzes, and displays empirical frequency and probability density distributions for 1D continuous numerical data in real time.

---

## Technical Specifications

- **Accepted Inputs**:
  - `IOTypes.SAMPLE` / `IOTypes.ROI_SAMPLE` (Dictionary containing `x`, `y`, `name`, `uuid`, and `action`).
  - `IOTypes.CMD_DICT` (Dictionary supporting actions `"add serie"`, `"remove serie"`, `"update serie name"`, `"clear"`).
  - `IOTypes.DATALIST` (Direct numerical array input).
- **Outputs**:
  - `DataFrame`: `IOTypes.DATAFRAME` (Long-format pandas DataFrame with columns `serie`, `x`, `y`, `point` and plot metadata in `df.attrs`).
  - `Datalist`: `IOTypes.DATALIST` (List of points representing bin centers and counts).

---

## Key Features & Controls

- **Probability Density & Count**: Toggle between normalized probability density (`Density` where $\int f(x) dx = 1$) and raw discrete frequency counts (`Count`).
- **Gaussian Kernel Smoothing**: Apply edge-padded Gaussian convolution smoothing with adjustable window radius to reduce discretization noise.
- **Temporal Rolling Average**: Smooth noisy streaming distributions across multiple frames using a configurable history window buffer (`Avg Win`).
- **Logarithmic Scaling**: Toggle logarithmic transformation on the vertical axis (`Log Y`) to inspect low-frequency distribution tails.
- **Shaded Areas**: Toggle translucent area fill beneath distribution curves.
- **Nearest-Point Cursor Inspection**: Interactive mouse hover tooltip displaying bin index, bin center coordinate, and frequency value.
- **TSV Clipboard Export**: Instantly export aligned distribution tables across all active series to the clipboard for pasting into spreadsheet tools.
- **Downstream Streaming (`Send Plot`)**: Broadcasts active curves as a DataFrame (`IOTypes.DATAFRAME`) with axis limits and configuration packets to downstream modules (e.g. `Matplot`).
