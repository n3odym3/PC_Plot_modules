from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional, Tuple
import uuid

import dearpygui.dearpygui as dpg
from loguru import logger
import numpy as np

from config.display_scaling import display_scaling
from core.app_state import app_state
from core.file_explorer import file_explorer
from core.input_output_types import IOTypes
from core.window_base import WindowBase
from core.tools.clipboard_injector import clipboardinjector


class Boxplot_win(WindowBase):
    """
    Boxplot and statistical distribution visualization module.

    Groups continuous or discrete data series by independent variable (time or bin),
    computes key parametric and non-parametric summary statistics (mean, standard deviation,
    standard error, median, interquartile range, min, max), and renders multi-layer interactive
    visualizations (candlestick boxplot, ribbons, envelopes, and scatter overlays).
    """

    description: str = "Interactive boxplot and statistical time distribution visualization."

    # 1. Window Configuration
    label = "Boxplot"
    win_width = display_scaling.scale(380)
    win_height = display_scaling.scale(280)

    # 2. Input/Output definitions
    accepted_input_types = [
        IOTypes.SAMPLE,
        IOTypes.ROI_SAMPLE,
        IOTypes.DATALIST,
        IOTypes.CMD_DICT,
    ]
    outputs = {
        "Summary Sample": IOTypes.SAMPLE,
        "Datalist": IOTypes.DATALIST,
        "CMD": IOTypes.CMD_DICT,
    }

    # 3. Persistent fields
    persistent_fields = [
        "label",
        "display_mode",
        "show_mean",
        "show_median",
        "show_std_ribbon",
        "show_minmax_ribbon",
        "show_scatter",
        "time_binning",
        "n_bins",
        "autoscale",
    ]

    # Persistent field defaults
    display_mode: str = "Candlestick Boxplot (IQR + Min/Max)"
    show_mean: bool = True
    show_median: bool = True
    show_std_ribbon: bool = False
    show_minmax_ribbon: bool = False
    show_scatter: bool = False
    time_binning: bool = False
    n_bins: int = 50
    autoscale: bool = True

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Boxplot window module.

        Args:
            **kwargs: Configuration parameters and persistent fields forwarded to WindowBase.
        """
        super().__init__(**kwargs)

        self.display_mode = str(self.display_mode if self.display_mode is not None else "Candlestick Boxplot (IQR + Min/Max)")
        self.show_mean = bool(self.show_mean if self.show_mean is not None else True)
        self.show_median = bool(self.show_median if self.show_median is not None else True)
        self.show_std_ribbon = bool(self.show_std_ribbon if self.show_std_ribbon is not None else False)
        self.show_minmax_ribbon = bool(self.show_minmax_ribbon if self.show_minmax_ribbon is not None else False)
        self.show_scatter = bool(self.show_scatter if self.show_scatter is not None else False)
        self.time_binning = bool(self.time_binning if self.time_binning is not None else False)
        self.n_bins = int(self.n_bins if self.n_bins is not None else 50)
        self.autoscale = bool(self.autoscale if self.autoscale is not None else True)

        # Data storage: { series_uuid: { 'raw_x': ..., 'raw_y': ..., 'stats': {...}, 'name': str } }
        self.series_data: Dict[str, Dict[str, Any]] = {}
        self.selected_uuid: Optional[str] = None

        # Tags
        self.plot_tag = f"box_plot_{self.UUID}"
        self.xaxis_tag = f"box_xaxis_{self.UUID}"
        self.yaxis_tag = f"box_yaxis_{self.UUID}"
        self.mode_combo_tag = f"box_mode_{self.UUID}"
        self.show_mean_chk_tag = f"box_mean_chk_{self.UUID}"
        self.show_median_chk_tag = f"box_median_chk_{self.UUID}"
        self.show_std_chk_tag = f"box_std_chk_{self.UUID}"
        self.show_minmax_chk_tag = f"box_minmax_chk_{self.UUID}"
        self.show_scatter_chk_tag = f"box_scatter_chk_{self.UUID}"
        self.binning_chk_tag = f"box_bin_chk_{self.UUID}"
        self.n_bins_tag = f"box_nbins_{self.UUID}"
        self.status_text_tag = f"box_status_{self.UUID}"

        # Color themes
        self._setup_themes()

        scale = display_scaling.scale
        with self.window():
            # Controls header
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    items=[
                        "Candlestick Boxplot (IQR + Min/Max)",
                        "Mean +/- Std Ribbon",
                        "Min/Max Envelope",
                        "Scatter + Mean",
                    ],
                    default_value=self.display_mode,
                    tag=self.mode_combo_tag,
                    width=scale(210),
                    callback=self._on_display_mode_changed,
                )
                dpg.add_button(label="Autoscale", callback=lambda: self.autofit_axis())
                dpg.add_button(label="Export CSV", callback=self._export_csv_cb)
                dpg.add_button(label="Export TSV", callback=self.export_tsv_cb)
                dpg.add_button(label="Clear", callback=self.clear_plot)

            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="Median",
                    tag=self.show_median_chk_tag,
                    default_value=self.show_median,
                    callback=self._update_plot_views,
                )
                dpg.add_checkbox(
                    label="Mean",
                    tag=self.show_mean_chk_tag,
                    default_value=self.show_mean,
                    callback=self._update_plot_views,
                )
                dpg.add_checkbox(
                    label="+/-Std",
                    tag=self.show_std_chk_tag,
                    default_value=self.show_std_ribbon,
                    callback=self._update_plot_views,
                )
                dpg.add_checkbox(
                    label="Min/Max",
                    tag=self.show_minmax_chk_tag,
                    default_value=self.show_minmax_ribbon,
                    callback=self._update_plot_views,
                )
                dpg.add_checkbox(
                    label="Scatter",
                    tag=self.show_scatter_chk_tag,
                    default_value=self.show_scatter,
                    callback=self._update_plot_views,
                )
                dpg.add_checkbox(
                    label="Bins",
                    tag=self.binning_chk_tag,
                    default_value=self.time_binning,
                    callback=self._on_binning_toggled,
                )
                dpg.add_input_int(
                    tag=self.n_bins_tag,
                    default_value=self.n_bins,
                    min_value=5,
                    max_value=200,
                    width=scale(55),
                    step=0,
                    show=self.time_binning,
                    callback=self._on_binning_toggled,
                )

            dpg.add_text(
                "Ready - Connect a data source",
                tag=self.status_text_tag,
                color=(160, 200, 255),
            )
            dpg.add_separator()

            # Plot container
            with dpg.plot(tag=self.plot_tag, height=-1, width=-1, no_menus=True):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag=self.xaxis_tag)
                dpg.add_plot_axis(dpg.mvYAxis, label="Value", tag=self.yaxis_tag)

        self.update_permission()

    def update_permission(self) -> None:
        """Adjust module permissions and UI visibility based on the application mode."""
        is_user = app_state.mode == "user"
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def close(self) -> None:
        """Clean up DPG plot themes on module close."""
        theme_attrs = [
            "theme_mean_line",
            "theme_std_shade",
            "theme_minmax_shade",
            "theme_scatter_dots",
            "theme_box_lines",
            "theme_median_line",
        ]
        for attr in theme_attrs:
            tag = getattr(self, attr, None)
            if tag and dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        super().close()

    def _setup_themes(self) -> None:
        """Initialize Dear PyGui plot series color and styling themes."""
        with dpg.theme() as self.theme_mean_line:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 200, 60, 255), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.5, category=dpg.mvThemeCat_Plots)

        with dpg.theme() as self.theme_std_shade:
            with dpg.theme_component(dpg.mvShadeSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Fill, (80, 160, 255, 80), category=dpg.mvThemeCat_Plots)

        with dpg.theme() as self.theme_minmax_shade:
            with dpg.theme_component(dpg.mvShadeSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Fill, (100, 220, 140, 45), category=dpg.mvThemeCat_Plots)

        with dpg.theme() as self.theme_scatter_dots:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (255, 120, 120, 90), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 3.0, category=dpg.mvThemeCat_Plots)

        with dpg.theme() as self.theme_box_lines:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (100, 200, 255, 220), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1.5, category=dpg.mvThemeCat_Plots)

        with dpg.theme() as self.theme_median_line:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 60, 80, 255), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.0, category=dpg.mvThemeCat_Plots)

    def _on_display_mode_changed(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Handle display mode combo selection changes."""
        self.display_mode = app_data
        self._update_plot_views()

    def _on_binning_toggled(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Handle time binning toggle and bin count adjustments."""
        if dpg.does_item_exist(self.binning_chk_tag):
            self.time_binning = bool(dpg.get_value(self.binning_chk_tag))
        if dpg.does_item_exist(self.n_bins_tag):
            self.n_bins = int(dpg.get_value(self.n_bins_tag))
            dpg.configure_item(self.n_bins_tag, show=self.time_binning)

        for uid in list(self.series_data.keys()):
            raw_x = self.series_data[uid]["raw_x"]
            raw_y = self.series_data[uid]["raw_y"]
            self.series_data[uid]["stats"] = self.compute_statistics(raw_x, raw_y)

        self._update_plot_views()

    def compute_statistics(self, x_arr: np.ndarray, y_arr: np.ndarray) -> Dict[str, Any]:
        """
        Group values by X (or discrete time bins) and compute statistical distribution metrics.

        Args:
            x_arr: 1D array of X coordinates (e.g. time points).
            y_arr: 1D array of measured values.

        Returns:
            Dict[str, Any]: Dictionary containing unique X points, mean, std, sem, min, max,
                            median, 25th/75th percentiles, and counts.
        """
        if len(x_arr) == 0 or len(y_arr) == 0:
            return {}

        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_clean = x_arr[mask]
        y_clean = y_arr[mask]

        if len(x_clean) == 0:
            return {}

        if self.time_binning and self.n_bins > 1:
            x_min, x_max = float(np.min(x_clean)), float(np.max(x_clean))
            if x_max == x_min:
                x_max = x_min + 1.0
            bin_edges = np.linspace(x_min, x_max, self.n_bins + 1)
            bin_indices = np.digitize(x_clean, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
            unique_x = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            groups = [y_clean[bin_indices == i] for i in range(self.n_bins)]
            valid_mask = [len(g) > 0 for g in groups]
            unique_x = unique_x[valid_mask]
            groups = [g for g in groups if len(g) > 0]
        else:
            unique_x, inverse_indices = np.unique(x_clean, return_inverse=True)
            groups = [y_clean[inverse_indices == i] for i in range(len(unique_x))]

        n_groups = len(unique_x)
        means = np.zeros(n_groups, dtype=float)
        stds = np.zeros(n_groups, dtype=float)
        sems = np.zeros(n_groups, dtype=float)
        mins = np.zeros(n_groups, dtype=float)
        maxs = np.zeros(n_groups, dtype=float)
        medians = np.zeros(n_groups, dtype=float)
        q25s = np.zeros(n_groups, dtype=float)
        q75s = np.zeros(n_groups, dtype=float)
        counts = np.zeros(n_groups, dtype=int)

        for i, grp in enumerate(groups):
            counts[i] = len(grp)
            means[i] = float(np.mean(grp))
            stds[i] = float(np.std(grp))
            sems[i] = stds[i] / np.sqrt(len(grp)) if len(grp) > 1 else 0.0
            mins[i] = float(np.min(grp))
            maxs[i] = float(np.max(grp))
            medians[i] = float(np.median(grp))
            q25s[i] = float(np.percentile(grp, 25))
            q75s[i] = float(np.percentile(grp, 75))

        return {
            "x": unique_x,
            "mean": means,
            "std": stds,
            "sem": sems,
            "min": mins,
            "max": maxs,
            "median": medians,
            "q25": q25s,
            "q75": q75s,
            "counts": counts,
            "raw_x": x_clean,
            "raw_y": y_clean,
        }

    def add_data_series(
        self,
        uid: Optional[str],
        x: Any,
        y: Any,
        name: str = "Sample",
        xlabel: str = "Time (s)",
        ylabel: str = "Value",
    ) -> None:
        """
        Add or update a statistical data series.

        Args:
            uid: Unique identifier for the series.
            x: X coordinates or time array.
            y: Measurement values array.
            name: Series label for display and legends.
            xlabel: Label for the horizontal plot axis.
            ylabel: Label for the vertical plot axis.
        """
        if uid is None:
            uid = str(uuid.uuid4())

        x_arr = np.asarray(x, dtype=float).ravel() if x is not None else np.array([])
        y_arr = np.asarray(y, dtype=float).ravel() if y is not None else np.array([])

        if len(x_arr) == 0 and len(y_arr) > 0:
            x_arr = np.arange(len(y_arr), dtype=float)

        stats = self.compute_statistics(x_arr, y_arr)

        self.series_data[uid] = {
            "name": name,
            "raw_x": x_arr,
            "raw_y": y_arr,
            "stats": stats,
            "xlabel": xlabel,
            "ylabel": ylabel,
        }
        self.selected_uuid = uid

        if dpg.does_item_exist(self.xaxis_tag) and xlabel:
            dpg.set_item_label(self.xaxis_tag, xlabel)
        if dpg.does_item_exist(self.yaxis_tag) and ylabel:
            dpg.set_item_label(self.yaxis_tag, ylabel)

        if dpg.does_item_exist(self.status_text_tag):
            n_pts = len(x_arr)
            n_grp = len(stats.get("x", []))
            dpg.set_value(self.status_text_tag, f"Loaded: {name} ({n_pts} points across {n_grp} time steps)")
            dpg.configure_item(self.status_text_tag, color=(100, 255, 120))

        self._update_plot_views()
        self._emit_summary_sample(uid)

    def _update_plot_views(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Redraw all active statistical visualization layers inside the Dear PyGui plot."""
        if dpg.does_item_exist(self.show_median_chk_tag):
            self.show_median = bool(dpg.get_value(self.show_median_chk_tag))
        if dpg.does_item_exist(self.show_mean_chk_tag):
            self.show_mean = bool(dpg.get_value(self.show_mean_chk_tag))
        if dpg.does_item_exist(self.show_std_chk_tag):
            self.show_std_ribbon = bool(dpg.get_value(self.show_std_chk_tag))
        if dpg.does_item_exist(self.show_minmax_chk_tag):
            self.show_minmax_ribbon = bool(dpg.get_value(self.show_minmax_chk_tag))
        if dpg.does_item_exist(self.show_scatter_chk_tag):
            self.show_scatter = bool(dpg.get_value(self.show_scatter_chk_tag))

        if dpg.does_item_exist(self.yaxis_tag):
            children = dpg.get_item_children(self.yaxis_tag, 1) or []
            for child in children:
                dpg.delete_item(child)

        for uid, s_data in self.series_data.items():
            stats = s_data.get("stats", {})
            if not stats or len(stats.get("x", [])) == 0:
                continue

            x = list(stats["x"])
            mean = list(stats["mean"])
            std = list(stats["std"])
            mins = list(stats["min"])
            maxs = list(stats["max"])
            q25 = list(stats["q25"])
            q75 = list(stats["q75"])
            median = list(stats["median"])
            name = s_data.get("name", "Sample")

            # 1. Raw Scatter Points
            if self.show_scatter or self.display_mode == "Scatter + Mean":
                raw_x = list(stats["raw_x"])
                raw_y = list(stats["raw_y"])
                scatter_tag = f"box_scat_{uid}"
                dpg.add_scatter_series(raw_x, raw_y, label=f"{name} (Points)", parent=self.yaxis_tag, tag=scatter_tag)
                dpg.bind_item_theme(scatter_tag, self.theme_scatter_dots)

            # 2. Min/Max Shaded Envelope
            if (self.show_minmax_ribbon and self.display_mode != "Candlestick Boxplot (IQR + Min/Max)") or self.display_mode == "Min/Max Envelope":
                mm_tag = f"box_mm_shade_{uid}"
                dpg.add_shade_series(x, mins, y2=maxs, label=f"{name} (Min-Max)", parent=self.yaxis_tag, tag=mm_tag)
                dpg.bind_item_theme(mm_tag, self.theme_minmax_shade)

            # 3. Mean +/- Std Dev Ribbon
            if (self.show_std_ribbon and self.display_mode != "Candlestick Boxplot (IQR + Min/Max)") or self.display_mode == "Mean +/- Std Ribbon":
                std_upper = (stats["mean"] + stats["std"]).tolist()
                std_lower = np.maximum(0.0, stats["mean"] - stats["std"]).tolist()
                std_tag = f"box_std_shade_{uid}"
                dpg.add_shade_series(x, std_lower, y2=std_upper, label=f"{name} (+/-1 Std)", parent=self.yaxis_tag, tag=std_tag)
                dpg.bind_item_theme(std_tag, self.theme_std_shade)

            # 4. Candlestick Boxplot Representation (IQR Box + Min/Max Whiskers)
            if self.display_mode == "Candlestick Boxplot (IQR + Min/Max)":
                if len(x) > 1:
                    dx = float(np.median(np.diff(x)))
                    candle_weight = max(0.05, min(dx * 0.35, 5.0))
                else:
                    candle_weight = 0.25

                candle_tag = f"box_candle_{uid}"
                dpg.add_candle_series(
                    dates=x,
                    opens=q25,
                    closes=q75,
                    lows=mins,
                    highs=maxs,
                    label=f"{name} (Candle IQR [Q25-Q75] & Min-Max)",
                    parent=self.yaxis_tag,
                    tag=candle_tag,
                    bull_color=(100, 190, 255, 230),
                    bear_color=(100, 190, 255, 230),
                    weight=candle_weight,
                    tooltip=True,
                )

            # 5. Median Line
            if self.show_median:
                med_tag = f"box_median_{uid}"
                dpg.add_line_series(x, median, label=f"{name} (Median)", parent=self.yaxis_tag, tag=med_tag)
                dpg.bind_item_theme(med_tag, self.theme_median_line)

            # 6. Mean Line
            if self.show_mean and self.display_mode != "Min/Max Envelope":
                line_tag = f"box_mean_{uid}"
                dpg.add_line_series(x, mean, label=f"{name} (Mean)", parent=self.yaxis_tag, tag=line_tag)
                dpg.bind_item_theme(line_tag, self.theme_mean_line)

        if self.autoscale:
            self.autofit_axis()

    def autofit_axis(self) -> None:
        """Automatically fit both horizontal and vertical axes to active data bounds."""
        if dpg.does_item_exist(self.xaxis_tag):
            dpg.fit_axis_data(self.xaxis_tag)
        if dpg.does_item_exist(self.yaxis_tag):
            dpg.fit_axis_data(self.yaxis_tag)

    def _emit_summary_sample(self, uid: str) -> None:
        """
        Emit statistical summary sample to connected downstream modules.

        Args:
            uid: Identifier of the series to emit.
        """
        s_data = self.series_data.get(uid)
        if not s_data:
            return
        stats = s_data.get("stats", {})
        if not stats or len(stats.get("x", [])) == 0:
            return

        payload = {
            "uuid": uid,
            "name": f"{s_data.get('name', 'Sample')} (Stats)",
            "x": stats["x"].tolist() if hasattr(stats["x"], "tolist") else stats["x"],
            "y": stats["mean"].tolist() if hasattr(stats["mean"], "tolist") else stats["mean"],
            "mean": stats["mean"].tolist() if hasattr(stats["mean"], "tolist") else stats["mean"],
            "std": stats["std"].tolist() if hasattr(stats["std"], "tolist") else stats["std"],
            "min": stats["min"].tolist() if hasattr(stats["min"], "tolist") else stats["min"],
            "max": stats["max"].tolist() if hasattr(stats["max"], "tolist") else stats["max"],
            "median": stats["median"].tolist() if hasattr(stats["median"], "tolist") else stats["median"],
            "q25": stats["q25"].tolist() if hasattr(stats["q25"], "tolist") else stats["q25"],
            "q75": stats["q75"].tolist() if hasattr(stats["q75"], "tolist") else stats["q75"],
            "counts": stats["counts"].tolist() if hasattr(stats["counts"], "tolist") else stats["counts"],
            "action": "select",
        }

        conns = self.connections.get("Summary Sample", [])
        for mod in conns:
            try:
                mod.input_cb(sample=payload, data_type=IOTypes.SAMPLE)
            except Exception as e:
                logger.debug(f"[{self.label}] Summary Sample emission failed: {e}")

    def _export_csv_cb(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Export statistical summary table to a CSV file."""
        if not self.selected_uuid or self.selected_uuid not in self.series_data:
            logger.warning(f"[{self.label}] Export aborted: no series selected.")
            return
        s_data = self.series_data[self.selected_uuid]
        stats = s_data.get("stats", {})
        if not stats or len(stats.get("x", [])) == 0:
            logger.warning(f"[{self.label}] Export aborted: selected series has no statistics.")
            return

        name = s_data.get("name", "stats").replace(" ", "_")
        default_filename = f"{name}_boxplot_stats.csv"

        def _save_callback(filepath: str, selected_ext: str = "") -> None:
            if not filepath:
                return
            try:
                with open(filepath, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Time_s", "Count", "Mean", "StdDev", "SEM", "Min", "Q25", "Median", "Q75", "Max"])
                    for i in range(len(stats["x"])):
                        writer.writerow([
                            f"{stats['x'][i]:.4f}",
                            stats["counts"][i],
                            f"{stats['mean'][i]:.4f}",
                            f"{stats['std'][i]:.4f}",
                            f"{stats['sem'][i]:.4f}",
                            f"{stats['min'][i]:.4f}",
                            f"{stats['q25'][i]:.4f}",
                            f"{stats['median'][i]:.4f}",
                            f"{stats['q75'][i]:.4f}",
                            f"{stats['max'][i]:.4f}",
                        ])
                if dpg.does_item_exist(self.status_text_tag):
                    dpg.set_value(self.status_text_tag, f"Exported stats to {filepath}")
                    dpg.configure_item(self.status_text_tag, color=(100, 255, 120))
                logger.success(f"[{self.label}] Stats exported to {filepath}")
            except Exception as e:
                logger.error(f"[{self.label}] CSV export failed: {e}")

        file_explorer.save_file(
            default_path=default_filename,
            extensions=[("CSV files", "*.csv"), ("All files", "*.*")],
            callback=_save_callback,
        )

    def export_tsv_cb(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Export statistical summary table to system clipboard in TSV format."""
        if not self.selected_uuid or self.selected_uuid not in self.series_data:
            logger.warning(f"[{self.label}] Export aborted: no series selected.")
            return
        s_data = self.series_data[self.selected_uuid]
        stats = s_data.get("stats", {})
        if not stats or len(stats.get("x", [])) == 0:
            logger.warning(f"[{self.label}] Export aborted: selected series has no statistics.")
            return

        lines = ["Time_s\tCount\tMean\tStdDev\tSEM\tMin\tQ25\tMedian\tQ75\tMax"]
        for i in range(len(stats["x"])):
            lines.append(
                f"{stats['x'][i]:.4f}\t"
                f"{stats['counts'][i]}\t"
                f"{stats['mean'][i]:.4f}\t"
                f"{stats['std'][i]:.4f}\t"
                f"{stats['sem'][i]:.4f}\t"
                f"{stats['min'][i]:.4f}\t"
                f"{stats['q25'][i]:.4f}\t"
                f"{stats['median'][i]:.4f}\t"
                f"{stats['q75'][i]:.4f}\t"
                f"{stats['max'][i]:.4f}"
            )

        clipboardinjector.send_text("\n".join(lines))
        if dpg.does_item_exist(self.status_text_tag):
            dpg.set_value(self.status_text_tag, "Stats exported to TSV clipboard.")
            dpg.configure_item(self.status_text_tag, color=(100, 255, 120))
        logger.success(f"[{self.label}] Stats exported to clipboard in TSV format.")

    def clear_plot(
        self,
        sender: Any = None,
        app_data: Any = None,
        user_data: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Remove all series, clear plot canvas, and reset selection state."""
        self.series_data.clear()
        self.selected_uuid = None
        if dpg.does_item_exist(self.yaxis_tag):
            children = dpg.get_item_children(self.yaxis_tag, 1) or []
            for child in children:
                dpg.delete_item(child)
        if dpg.does_item_exist(self.status_text_tag):
            dpg.set_value(self.status_text_tag, "Plot cleared.")
            dpg.configure_item(self.status_text_tag, color=(160, 200, 255))

    def is_ready(self) -> bool:
        """
        Check if the module is ready to receive input data.

        Returns:
            bool: Always True.
        """
        return True

    def input_cb(self, *args: Any, **kwargs: Any) -> None:
        """
        Handle incoming sample or command data from upstream modules.

        Supported inputs:
            - SAMPLE / ROI_SAMPLE: Dictionary with keys 'x', 'y', 'name', 'uuid', 'action'.
            - DATALIST: Numerical arrays (x, y).
            - CMD_DICT: Commands including {'clear': True}.
        """
        sample = kwargs.get("sample")
        data = kwargs.get("data") if "data" in kwargs else (args[0] if args else None)
        data_type = kwargs.get("data_type", None)

        if sample or data_type == IOTypes.SAMPLE:
            s_dict = sample or (data if isinstance(data, dict) else {})
            action = s_dict.get("action", "select")
            uid = s_dict.get("uuid", str(uuid.uuid4()))

            if action in ["unselect", "delete"]:
                if uid in self.series_data:
                    del self.series_data[uid]
                    self._update_plot_views()
            elif action == "clear" or s_dict.get("clear"):
                self.clear_plot()
            else:
                x_val = s_dict.get("x")
                y_val = s_dict.get("y")
                name_val = s_dict.get("name", "Sample")
                xlabel = s_dict.get("xlabel", "Time (s)")
                ylabel = s_dict.get("ylabel", "Value")
                self.add_data_series(uid, x_val, y_val, name=name_val, xlabel=xlabel, ylabel=ylabel)
            return

        if isinstance(data, (tuple, list)) and len(data) >= 2:
            if isinstance(data[0], (np.ndarray, list)) and isinstance(data[1], (np.ndarray, list)):
                self.add_data_series(None, data[0], data[1], name="Sample")
                return

        if data_type == IOTypes.CMD_DICT and isinstance(data, dict):
            if data.get("clear"):
                self.clear_plot()
            return


EXPORTED_CLASS = Boxplot_win
EXPORTED_NAME = "Boxplot"
