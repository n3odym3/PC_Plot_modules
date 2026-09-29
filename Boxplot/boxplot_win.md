# Boxplot Module

The **Boxplot & Statistical Time Distribution** module calculates and visualizes statistical metrics over time (or any grouped independent variable X). For each unique time step or time bin, it computes parametric and non-parametric distributions.

---

## Technical Specifications

- **Accepted Inputs**:
  - `IOTypes.SAMPLE` / `IOTypes.ROI_SAMPLE` (Dictionary with `x` [time], `y` [values], `name`, `uuid`, and `action`).
  - `IOTypes.DATALIST` (Numerical arrays `(x, y)`).
  - `IOTypes.CMD_DICT` (Remote commands such as `{"clear": True}`).
- **Outputs**:
  - `Summary Sample`: `IOTypes.SAMPLE` (Emits computed statistics dictionary: `mean`, `std`, `sem`, `min`, `max`, `median`, `q25`, `q75`, `counts`).
  - `Datalist`: `IOTypes.DATALIST` (Legacy datalist output).
  - `CMD`: `IOTypes.CMD_DICT`.

---

## Computed Metrics

For each time coordinate or discrete time bin, the module computes:
- **Mean** ($\mu$) & **Standard Deviation** ($\sigma$)
- **Standard Error of the Mean (SEM)** ($\sigma / \sqrt{N}$)
- **Median** ($Q_2$), **25th Percentile** ($Q_1$), and **75th Percentile** ($Q_3$) (Interquartile Range - IQR)
- **Minimum & Maximum** (Whiskers)
- **Sample Count** ($N$)

---

## Visualization Modes

- `Candlestick Boxplot (IQR + Min/Max)`: Candlestick representation where the body spans the Interquartile Range ($Q_{25}-Q_{75}$) and whiskers span extreme $Min-Max$ observations.
- `Mean +/- Std Ribbon`: Central mean line accompanied by a translucent $\pm 1\sigma$ standard deviation envelope.
- `Min/Max Envelope`: Shaded band delimiting the full extent of observed minimum and maximum values.
- `Scatter + Mean`: All raw scatter points overlaid with the calculated mean trend line.

---

## UI Controls

- **Display Mode** (Combo): Switches between the 4 visual representations.
- **Median / Mean / +/-Std / Min/Max / Scatter** (Checkboxes): Selectively toggle individual visual layers.
- **Bins** (Checkbox & Input): Enables binning along the X axis and configures the number of discrete intervals (5 to 200).
- **Autoscale** (Button): Automatically fits both horizontal and vertical axes to active data.
- **Export CSV** (Button): Exports a complete tabular summary of all statistical metrics across time steps.
- **Clear** (Button): Clears all plotted series and resets memory.
