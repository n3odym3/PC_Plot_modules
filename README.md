# Plot & Visualization Modules for Pipeline Creator

A collection of real-time graphing, data plotting, statistical distribution, and signal visualization modules for [**Pipeline Creator**](https://github.com/n3odym3/Pipeline_creator).

This repository contains plotting, data buffering, and statistical visualization modules designed for plug-and-play integration into the Pipeline Creator ecosystem. Whether inspecting continuous multi-channel time series, computing statistical distributions, filtering data with scriptable pandas pipelines, rendering publication-quality Matplotlib/Seaborn graphics, or plotting vector optical flow on polar charts, these modules provide real-time interactive visual feedback, high-throughput pipeline communication, and convenient data export.

---

## Installation

To install and use one or more modules in **Pipeline Creator**:

1. **Install Module Dependencies**:

   Install the required Python packages within your Pipeline Creator environment:
   ```bash
   pip install numpy pandas scipy matplotlib seaborn opencv-python loguru
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
   > `clipboard_injector.py` is a shared helper used by several visualization modules (`Lineplot`, `Matplot`, `Polarplot`, `ColorPie`, `Sample_container`) to copy graphs and tabular data to the system clipboard. Ensure it is placed in the core folder of your Pipeline Creator installation.

3. **Launch Pipeline Creator**:

   The copied modules will be automatically detected and made available in the Pipeline Creator node editor.

---

## Available Modules

| Module | Description | Inputs | Outputs | Key Dependencies |
| :--- | :--- | :--- | :--- | :--- |
| [**Lineplot**](./Lineplot/) | Interactive 2D multi-series curve and time-series visualization. Supports dynamic hover tooltips, interactive draggable vertical threshold/trigger line, multi-algorithm smoothing (Rolling Median, Rolling Mean, Savitzky-Golay, Gaussian, Exponential EMA), logarithmic X scaling, autoscale, TSV clipboard export, and downstream DataFrame streaming (`Send Plot`). | `SAMPLE`<br>`ROI_SAMPLE`<br>`DATALIST`<br>`CMD_DICT` | `DataFrame` (`DATAFRAME`)<br>`Datalist` (`DATALIST`) | `numpy`, `pandas`, `scipy`, `dearpygui`, `loguru` |
| [**Matplot**](./Matplot/) | Publication-ready 2D plotting module powered by Matplotlib and Seaborn (`darkgrid`, `muted`). Ingests long-format DataFrames (`DATAFRAME`) with axis and color metadata in `df.attrs`. Features a scriptable **Data Explorer** (`Data Filter`) with row-level pandas queries, intra-group per-$X$ slicing (e.g. `[-1]` for last points), aggregations (`Mean ± SD` with shaded error bands $\pm 1\sigma$, `Median`, `Min`, `Max`, `Raw points`), or custom Python/Pandas script mode (`df = ...`). Includes fast preview table, TSV data copy, custom DPI/resolutions, direct clipboard copying, and BGR video frame emission. | `DATAFRAME` | `Frame` (`FRAME`) | `pandas`, `matplotlib`, `seaborn`, `opencv-python`, `numpy`, `dearpygui`, `loguru` |
| [**Histogram**](./Histogram/) | Real-time 1D frequency and probability density distribution analysis. Supports discrete counts vs. normalized probability density ($\int f(x)dx=1$), Gaussian kernel convolution smoothing, temporal rolling average, logarithmic Y scaling, interactive hover inspection, TSV clipboard export, and downstream DataFrame streaming (`Send Plot`). | `SAMPLE`<br>`ROI_SAMPLE`<br>`DATALIST`<br>`CMD_DICT` | `DataFrame` (`DATAFRAME`)<br>`Datalist` (`DATALIST`) | `numpy`, `pandas`, `dearpygui`, `loguru` |
| [**Boxplot**](./Boxplot/) | Statistical time distribution and parametric/non-parametric metric visualizer. Computes mean ($\mu$), std ($\sigma$), SEM ($\sigma/\sqrt{N}$), median ($Q_2$), IQR ($Q_1-Q_3$), min/max whiskers, and sample counts ($N$) across time or custom bins. Supports 4 rendering modes (Candlestick Boxplot, Mean $\pm$ Std Ribbon, Min/Max Envelope, Scatter + Mean) and CSV summary export. | `SAMPLE`<br>`ROI_SAMPLE`<br>`DATALIST`<br>`CMD_DICT` | `Summary Sample` (`SAMPLE`)<br>`Datalist` (`DATALIST`)<br>`CMD` (`CMD_DICT`) | `numpy`, `dearpygui`, `loguru` |
| [**Polarplot**](./Polarplot/) | Circular vector orientation and angular distribution visualizer (Rose diagrams) with real-time circular statistics (mean angle $\bar{\theta}$, directionality index $R$, mean speed). Offloaded to an asynchronous multiprocessing worker (`PolarPlotProcessor` / `ProcessingBase`) for zero-lag rendering. Supports velocity weighting, batch vs. rolling window accumulation, HSV optical flow coloring, coordinate inversion, and BGR frame output. | `VECTORS`<br>`SAMPLE`<br>`DATALIST`<br>`CMD_DICT`<br>`TRIGGER` | `Frame` (`FRAME`)<br>`CMD` (`CMD_DICT`) | `matplotlib`, `opencv-python`, `numpy`, `dearpygui`, `loguru` |
| [**ColorPie**](./ColorPie/) | Circular pie chart visualization of categorical color distributions (designed to consume classification metrics from upstream modules such as `ColorMapper`). Displays relative percentage overlays, dark theme styling, image clipboard export, TSV export, and frame streaming. | `CMD_DICT` | `Frame` (`FRAME`) | `matplotlib`, `opencv-python`, `numpy`, `dearpygui` |
| [**Sample_container**](./Sample_container/) | Centralized dataset repository and manager for `SAMPLE` signals. Features drag-and-drop reordering, interactive mini-plot hover previews, duplicate conflict resolution (`override` / `append`), multi-column CSV import/export with configurable timebase/X referencing, and reactive event streaming (`select`, `unselect`, `rename`). | `SAMPLE`<br>`FOLDER_PATH`<br>`TRIGGER`<br>`CMD_DICT` | `Data` (`SAMPLE`) | `dearpygui`, `numpy`, `loguru` |

---

## Inter-Module Data Protocols

### 1. `IOTypes.DATAFRAME` (DataFrame Streaming)
Used by `Lineplot` and `Histogram` to stream structured series datasets to `Matplot`:
- **Payload**: Long-format `pandas.DataFrame` with standard columns:
  - `serie`: Identifier / label of the series (string or category).
  - `x`: Independent variable (e.g. time, wavelength, light step, or bin center).
  - `y`: Dependent variable (e.g. signal amplitude, fluorescence, probability density).
  - `point` *(optional)*: 0-based index of observation within the series.
- **Metadata (`df.attrs`)**: Stored in DataFrame attributes dictionary:
  - `title`: Plot title string.
  - `xlabel`, `ylabel`: Coordinate axis labels.
  - `x_lim`, `y_lim`: Axis limit tuples `(min, max)`.
  - `log_x` / `log_y`: Boolean flags for logarithmic scale conversion.
  - `colors`: Optional series-to-hex color mapping dictionary `{"SerieName": "#RRGGBB"}`.

### 2. `IOTypes.SAMPLE` / `IOTypes.ROI_SAMPLE`
Standard continuous signal structure exchanged between `Sample_container`, `Lineplot`, `Boxplot`, and `Histogram`:
- **Payload**: Dictionary containing:
  - `x`: 1D array of horizontal coordinates (timestamps, indices, etc.).
  - `y`: 1D array of signal amplitudes or values.
  - `name`: Human-readable identifier for the curve.
  - `uuid`: Unique identifier for tracking and state preservation.
  - `action`: State event (`"select"`, `"unselect"`, `"rename"`, `"delete"`, `"clear"`).

### 3. `IOTypes.FRAME`
Raw BGR video frames ($H \times W \times 3$, `uint8` NumPy ndarray) emitted by `Matplot`, `Polarplot`, and `ColorPie` for direct streaming to Pipeline Creator image viewports, display windows, or video encoder nodes.

### 4. `IOTypes.VECTORS`
Directional vector field payload ingested by `Polarplot`:
- **Payload**: Dictionary with `angles` (1D array in radians) and `magnitudes` (1D array of vector speeds or lengths).

---

## Shared Utilities

- **[`clipboard_injector.py`](./clipboard_injector.py)**: Cross-platform clipboard helper enabling one-click copying of rendered plot graphics (PNG format, Windows `win32clipboard` CF_DIB) and tabular data (TSV) directly to the operating system clipboard for instant pasting into applications like Excel, Word, Prism, or image editors.
