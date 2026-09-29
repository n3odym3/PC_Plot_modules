# Plot & Visualization Modules for Pipeline Creator

A collection of real-time graphing, data plotting, statistical distribution, and signal visualization modules for [**Pipeline Creator**](https://github.com/n3odym3/Pipeline_creator).

This repository contains plotting, data buffering, and statistical visualization modules designed for plug-and-play integration into the Pipeline Creator ecosystem. Whether inspecting continuous multi-channel time series, computing statistical distributions, rendering publication-quality Matplotlib/Seaborn graphics, or plotting vector optical flow on polar charts, these modules provide real-time interactive visual feedback and convenient data export.

---

## Installation

To install and use one or more modules in **Pipeline Creator**:

1. **Install Module Dependencies**:

   Install the required Python packages within your Pipeline Creator environment:
   ```bash
   pip install numpy matplotlib seaborn opencv-python loguru
   # Optional: Windows clipboard integration for copying plot images directly
   pip install pywin32
   ```

2. **Copy Modules to Pipeline Creator**:
   
   Copy and paste the folder(s) of the desired module(s) directly into the `modules/` folder (or your plot modules subfolder) of your **Pipeline Creator** installation:
   ```
   Pipeline_creator/
   ├── modules/
   │   ├── Boxplot/
   │   ├── ColorPie/
   │   ├── Histogram/
   │   ├── Lineplot/
   │   ├── Matplot/
   │   ├── Polarplot/
   │   ├── Sample_container/
   │   ├── clipboard_injector.py
   │   └── ...
   ```
   > [!NOTE]
   > `clipboard_injector.py` is a shared helper used by several visualization modules (`Lineplot`, `Matplot`, `Polarplot`, `ColorPie`, `Sample_container`) to copy graphs and tabular data to the system clipboard. Ensure it is placed alongside the modules or in your Python path.

3. **Launch Pipeline Creator**:

   The copied modules will be automatically detected and made available in the Pipeline Creator node editor.

---

## Available Modules

| Module | Description | Key Dependencies |
| :--- | :--- | :--- |
| [**Lineplot**](./Lineplot/) | Interactive 2D multi-series curve and time-series visualization. Supports dynamic hover tooltips, interactive draggable vertical threshold/trigger line, rolling-average smoothing, logarithmic X scaling, autoscale, TSV clipboard export, and downstream plot streaming (`CMD`). | `numpy`, `dearpygui`, `loguru` |
| [**Histogram**](./Histogram/) | Real-time 1D frequency and probability density distribution analysis. Supports discrete counts vs. normalized probability density ($\int f(x)dx=1$), Gaussian kernel convolution smoothing, temporal rolling average, logarithmic Y scaling, interactive hover inspection, and TSV clipboard export. | `numpy`, `dearpygui`, `loguru` |
| [**Boxplot**](./Boxplot/) | Statistical time distribution and parametric/non-parametric metric visualizer. Computes mean, std, SEM, median, IQR ($Q_1-Q_3$), and min/max whiskers across time or custom bins. Supports 4 rendering modes (Candlestick Boxplot, Mean $\pm$ Std Ribbon, Min/Max Envelope, Scatter + Mean) and CSV export. | `numpy`, `dearpygui`, `loguru` |
| [**Matplot**](./Matplot/) | Publication-ready 2D plotting module powered by Matplotlib and Seaborn (`darkgrid`). Ingests structured plot packets (`CMD_DICT`), automatically aggregates repeated observations with standard deviation shaded error bands ($\pm 1\sigma$), offers custom DPI and resolutions, direct clipboard copying, and emits rendered BGR video frames (`FRAME`). | `matplotlib`, `seaborn`, `opencv-python`, `numpy`, `dearpygui` |
| [**Polarplot**](./Polarplot/) | Circular vector orientation and angular distribution visualizer (Rose diagrams) with real-time circular statistics (mean angle $\bar{\theta}$, directionality index $R$, mean speed). Offloaded to an asynchronous multiprocessing worker (`ProcessingBase`) for zero-lag rendering. Supports velocity weighting, batch vs. rolling window accumulation, HSV optical flow coloring, and BGR frame output. | `matplotlib`, `opencv-python`, `numpy`, `dearpygui`, `loguru` |
| [**ColorPie**](./ColorPie/) | Circular pie chart visualization of categorical color distributions (designed to consume classification metrics from upstream modules such as `ColorMapper`). Displays relative percentage overlays, supports image clipboard export, TSV export, and frame streaming (`FRAME`). | `matplotlib`, `opencv-python`, `numpy`, `dearpygui` |
| [**Sample_container**](./Sample_container/) | Centralized dataset repository and manager for `SAMPLE` signals. Features drag-and-drop reordering, interactive mini-plot hover previews, duplicate resolution (`override`/`append`), multi-column CSV import/export with configurable timebase/X referencing, and reactive event streaming (`select`, `unselect`, `rename`). | `dearpygui`, `numpy`, `loguru` |

---

## Shared Utilities

- **[`clipboard_injector.py`](./clipboard_injector.py)**: Cross-platform clipboard helper enabling one-click copying of rendered plot graphics (PNG format, Windows `win32clipboard` CF_DIB) and tabular data directly to the operating system clipboard for instant pasting into applications like Excel, Word, Prism, or image editors.
