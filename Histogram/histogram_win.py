from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import uuid

import dearpygui.dearpygui as dpg
from loguru import logger
import numpy as np

from config.display_scaling import display_scaling
from core.app_state import app_state
from core.input_output_types import IOTypes
from core.window_base import WindowBase
from core.tools.clipboard_injector import clipboardinjector


class Histogram_win(WindowBase):
    """
    Histogram and distribution analysis module.

    Computes empirical frequency or probability density distributions for 1D numerical data.
    Supports real-time rolling averaging, Gaussian kernel smoothing, logarithmic Y scaling,
    shaded area rendering, and TSV clipboard export.
    """

    description: str = "Interactive 1D histogram and density distribution viewer."

    # 1. Window Configuration
    label = "Histogram"
    win_width = display_scaling.scale(350)
    win_height = display_scaling.scale(240)

    # 2. Input/Output definitions
    accepted_input_types = [
        IOTypes.SAMPLE,
        IOTypes.ROI_SAMPLE,
        IOTypes.DATALIST,
        IOTypes.CMD_DICT,
    ]
    outputs = {
        "Datalist": IOTypes.DATALIST,
        "DataFrame": IOTypes.DATAFRAME,
    }

    # 3. Persistent fields
    persistent_fields = [
        "label",
        "autoscale",
        "bins",
        "density",
        "log_y",
        "smooth",
        "smooth_window",
        "shading",
        "averaging",
        "avg_window",
    ]

    # Persistent field defaults
    autoscale: bool = True
    bins: int = 50
    density: bool = True
    log_y: bool = False
    smooth: bool = True
    smooth_window: int = 5
    shading: bool = True
    averaging: bool = True
    avg_window: int = 5

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Histogram window module.

        Args:
            **kwargs: Configuration parameters and persistent fields forwarded to WindowBase.
        """
        if "uuid_val" in kwargs and "uuid" not in kwargs:
            kwargs["uuid"] = kwargs["uuid_val"]

        super().__init__(**kwargs)

        self.autoscale = bool(self.autoscale if self.autoscale is not None else True)
        self.bins = int(self.bins if self.bins is not None else 50)
        self.density = bool(self.density if self.density is not None else True)
        self.log_y = bool(self.log_y if self.log_y is not None else False)
        self.smooth = bool(self.smooth if self.smooth is not None else True)
        self.smooth_window = int(self.smooth_window if self.smooth_window is not None else 5)
        self.shading = bool(self.shading if self.shading is not None else True)
        self.averaging = bool(self.averaging if self.averaging is not None else True)
        self.avg_window = int(self.avg_window if self.avg_window is not None else 5)

        # Unique DPG tags
        self.plot_tag = f"histogram_plot_{self.UUID}"
        self.anot_check_tag = f"histogram_anot_check_{self.UUID}"
        self.autoscale_check_tag = f"histogram_autoscale_check_{self.UUID}"
        self.density_check_tag = f"histogram_density_check_{self.UUID}"
        self.bins_tag = f"histogram_bins_{self.UUID}"
        self.logY_check_tag = f"histogram_logy_check_{self.UUID}"
        self.smooth_check_tag = f"histogram_smooth_check_{self.UUID}"
        self.smooth_win_tag = f"histogram_smooth_win_{self.UUID}"
        self.shading_check_tag = f"histogram_shading_check_{self.UUID}"
        self.avg_check_tag = f"histogram_avg_check_{self.UUID}"
        self.avg_win_tag = f"histogram_avg_win_{self.UUID}"

        self.xaxis_tag = f"histogram_x_axis_{self.UUID}"
        self.yaxis_tag = f"histogram_y_axis_{self.UUID}"
        self.closest_point_anot_tag = f"histogram_closest_point_anot_{self.UUID}"
        self.handler_tag = f"histogram_handler_{self.UUID}"
        self.export_btn_tag = f"histogram_export_btn_{self.UUID}"

        # Data storage
        self._is_closing: bool = False
        self._cached_kernel_win: Optional[int] = None
        self._cached_kernel: Optional[np.ndarray] = None
        self.raw_data: Dict[str, Dict[str, Any]] = {}          # {UUID: {"data": np.array, "name": str, "color": ...}}
        self.raw_data_history: Dict[str, List[np.ndarray]] = {}  # {UUID: [data1, data2, ...]}
        self.hist_data: Dict[str, Dict[str, Any]] = {}         # {UUID: {"x": [], "y": [], "name": str}}
        self.selected_uuid: Optional[str] = None

        scale = display_scaling.scale
        with self.window():
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Anotation", tag=self.anot_check_tag, default_value=True)
                dpg.add_checkbox(label="Autoscale", tag=self.autoscale_check_tag, default_value=self.autoscale, callback=self.toggle_autoscale)
                dpg.add_checkbox(label="Density", tag=self.density_check_tag, default_value=self.density, callback=self.toggle_density)
                dpg.add_checkbox(label="Log Y", tag=self.logY_check_tag, default_value=self.log_y, callback=self.toggle_log_y)
                dpg.add_checkbox(label="Shading", tag=self.shading_check_tag, default_value=self.shading, callback=self.toggle_shading)

            with dpg.group(horizontal=True):
                dpg.add_drag_int(label="Bins", tag=self.bins_tag, default_value=self.bins, min_value=1, max_value=500, width=scale(70), callback=self.update_bins)
                dpg.add_checkbox(label="Smooth", tag=self.smooth_check_tag, default_value=self.smooth, callback=self.toggle_smooth)
                dpg.add_drag_int(label="Win", tag=self.smooth_win_tag, default_value=self.smooth_window, min_value=1, max_value=50, width=scale(50), callback=self.update_smooth_win)
                dpg.add_checkbox(label="Avg", tag=self.avg_check_tag, default_value=self.averaging, callback=self.toggle_averaging)
                dpg.add_drag_int(label="Avg Win", tag=self.avg_win_tag, default_value=self.avg_window, min_value=2, max_value=200, width=scale(50), callback=self.update_avg_win)
                dpg.add_button(label="Export TSV", tag=self.export_btn_tag, callback=self.export_tsv_cb)
                dpg.add_button(label="Send", callback=self.send_plot_cb)
                dpg.add_button(label="Clear", callback=self.clear_plot)

            with dpg.plot(label="Histogram Serie", height=-1, width=-1, tag=self.plot_tag):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Value", tag=self.xaxis_tag)
                dpg.add_plot_axis(dpg.mvYAxis, label="Density" if self.density else "Count", tag=self.yaxis_tag, log_scale=self.log_y)

            with dpg.handler_registry(tag=self.handler_tag):
                dpg.add_mouse_move_handler(callback=self.plot_change_callback)

            dpg.configure_item(self.plot_tag, anti_aliased=True)
            dpg.configure_item(self.plot_tag, crosshairs=True)

        self.update_permission()

    def update_permission(self) -> None:
        """Adjust module permissions and UI visibility based on the application mode."""
        mode = app_state.mode
        elements = [
            self.anot_check_tag,
            self.density_check_tag,
            self.bins_tag,
            self.smooth_check_tag,
            self.smooth_win_tag,
            self.shading_check_tag,
            self.avg_check_tag,
            self.avg_win_tag,
        ]

        show = (mode != "user")
        for tag in elements:
            if dpg.does_item_exist(tag):
                if show:
                    dpg.show_item(tag)
                else:
                    dpg.hide_item(tag)

        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=(mode == "user"))

    def on_close(self) -> None:
        """Clean up handler registries and plot themes before closing."""
        if getattr(self, "_is_closing", False):
            return
        self._is_closing = True
        if hasattr(self, "handler_tag") and dpg.does_item_exist(self.handler_tag):
            dpg.delete_item(self.handler_tag)
        for uid in list(self.raw_data.keys()):
            theme_tag = f"hist_theme_{uid}_{self.UUID}"
            if dpg.does_item_exist(theme_tag):
                dpg.delete_item(theme_tag)
        super().on_close()

    def close(self) -> None:
        """Clean up handler registries and plot themes and close window."""
        self.on_close()
        super().close()

    def toggle_autoscale(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Toggle automatic axis fitting on data updates."""
        self.autoscale = bool(app_data)
        if self.autoscale:
            self.autofit_axis()

    def toggle_density(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Toggle between probability density and raw frequency counts."""
        self.density = bool(app_data)
        dpg.set_item_label(self.yaxis_tag, "Density" if self.density else "Count")
        self.replot_all()

    def update_bins(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Update histogram bin partition count."""
        self.bins = int(app_data)
        self.replot_all()

    def toggle_smooth(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Toggle Gaussian smoothing filter on/off."""
        self.smooth = bool(app_data)
        self.replot_all()

    def update_smooth_win(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Update Gaussian smoothing kernel width."""
        self.smooth_window = int(app_data)
        self.replot_all()

    def toggle_shading(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Toggle translucent fill beneath histogram curves."""
        self.shading = bool(app_data)
        self.replot_all()

    def toggle_averaging(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Toggle rolling time averaging across incoming frames."""
        self.averaging = bool(app_data)
        self.replot_all()

    def update_avg_win(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Update buffer capacity for rolling temporal averaging."""
        self.avg_window = int(app_data)
        for uid in self.raw_data_history:
            if len(self.raw_data_history[uid]) > self.avg_window:
                self.raw_data_history[uid] = self.raw_data_history[uid][-self.avg_window:]
        self.replot_all()

    def toggle_log_y(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Toggle logarithmic vertical axis scaling."""
        self.log_y = bool(app_data)
        dpg.delete_item(self.xaxis_tag)
        dpg.delete_item(self.yaxis_tag)

        dpg.add_plot_axis(dpg.mvXAxis, label="Value", tag=self.xaxis_tag, parent=self.plot_tag)
        dpg.add_plot_axis(
            dpg.mvYAxis,
            label="Density" if self.density else "Count",
            tag=self.yaxis_tag,
            log_scale=self.log_y,
            parent=self.plot_tag,
        )
        self.replot_all()

    def _smooth_curve(self, y: np.ndarray, window: int) -> np.ndarray:
        """
        Apply Gaussian-weighted smoothing over curve coordinates.

        Args:
            y: Input 1D array of curve values.
            window: Smoothing filter radius.

        Returns:
            np.ndarray: Smoothed non-negative curve values.
        """
        if window <= 1 or len(y) < 3:
            return y
        radius = int(window)
        if self._cached_kernel_win != radius or self._cached_kernel is None:
            sigma = max(1.0, radius / 2.5)
            x_k = np.arange(-radius, radius + 1)
            kernel = np.exp(-0.5 * (x_k / sigma) ** 2)
            kernel /= kernel.sum()
            self._cached_kernel_win = radius
            self._cached_kernel = kernel
        pad_y = np.pad(y, radius, mode="edge")
        smoothed = np.convolve(pad_y, self._cached_kernel, mode="valid")
        return np.maximum(0.0, smoothed)

    def replot_all(self) -> None:
        """Recompute and redraw all stored distribution series."""
        for uid in list(self.raw_data.keys()):
            self.plot_serie(uid)

    def export_tsv_cb(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Export all plotted distributions to system clipboard in aligned TSV format."""
        try:
            if not self.hist_data:
                return

            series_list = []
            all_x = set()

            for uid, hist in self.hist_data.items():
                series_dict = dict(zip(hist["x"], hist["y"]))
                series_list.append({"label": hist["name"], "data": series_dict})
                all_x.update(hist["x"])

            sorted_x = sorted(list(all_x))
            tsv_lines = []
            headers = ["X"] + [s["label"] for s in series_list]
            tsv_lines.append("\t".join(headers))

            for x in sorted_x:
                row = [str(x)]
                for series in series_list:
                    y_val = series["data"].get(x, "")
                    row.append(str(y_val))
                tsv_lines.append("\t".join(row))

            tsv_content = "\n".join(tsv_lines)
            clipboardinjector.send_text(tsv_content)
            logger.info(f"[{self.label}] Exported {len(series_list)} histogram series to clipboard as TSV")
        except Exception as e:
            logger.error(f"[{self.label}] Error exporting TSV: {e}")

    def send_plot_cb(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Package active plot state and dispatch via DataFrame output."""
        try:
            all_series = []
            for uid, hist in self.hist_data.items():
                all_series.append({
                    "label": hist["name"],
                    "x": hist["x"],
                    "y": hist["y"],
                })

            if not all_series:
                logger.warning(f"[{self.label}] No data to send")
                return

            import pandas as pd

            df = pd.concat(
                [
                    pd.DataFrame({"serie": s["label"], "x": s["x"], "y": s["y"], "point": range(len(s["x"]))})
                    for s in all_series
                ],
                ignore_index=True,
            )

            df.attrs.update(
                title=self.label,
                xlabel=dpg.get_item_label(self.xaxis_tag),
                ylabel=dpg.get_item_label(self.yaxis_tag),
                x_lim=dpg.get_axis_limits(self.xaxis_tag),
                y_lim=dpg.get_axis_limits(self.yaxis_tag),
                log_y=dpg.get_value(self.logY_check_tag),
            )

            self.trigger_cb(out_key="DataFrame", data=df)
            logger.info(f"[{self.label}] Sent {len(all_series)} series to DataFrame output")
        except Exception as e:
            logger.error(f"[{self.label}] Error sending plot: {e}")

    def find_closest_point(
        self,
        mouse_x: float,
        mouse_y: float,
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_scale: Optional[float] = None,
        y_scale: Optional[float] = None,
    ) -> Optional[int]:
        """Find index of data point closest to current mouse coordinates."""
        if len(x_data) == 0 or len(y_data) == 0:
            return None
        if x_scale is None:
            ptp_x = np.ptp(x_data)
            x_scale = 1.0 / ptp_x if ptp_x != 0 else 1.0
        if y_scale is None:
            ptp_y = np.ptp(y_data)
            y_scale = 1.0 / ptp_y if ptp_y != 0 else 1.0
        distances = np.hypot((x_data - mouse_x) * x_scale, (y_data - mouse_y) * y_scale)
        return int(np.argmin(distances))

    def plot_change_callback(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Inspect closest point and render on-hover cursor tooltip annotation."""
        if not self.hist_data or not dpg.does_item_exist(self.plot_tag) or not dpg.is_item_hovered(self.plot_tag):
            return

        if dpg.get_value(self.anot_check_tag):
            uid_to_anot = self.selected_uuid if self.selected_uuid in self.hist_data else next(iter(self.hist_data))
            hist = self.hist_data[uid_to_anot]

            x_np = hist.get("x_np")
            y_np = hist.get("y_np")
            if x_np is None or y_np is None:
                x_np = np.asarray(hist["x"], dtype=np.float64)
                y_np = np.asarray(hist["y"], dtype=np.float64)

            mouse_x, mouse_y = dpg.get_plot_mouse_pos()
            point_index = self.find_closest_point(
                mouse_x,
                mouse_y,
                x_np,
                y_np,
                x_scale=hist.get("x_scale"),
                y_scale=hist.get("y_scale"),
            )

            if point_index is not None:
                px = float(x_np[point_index])
                py = float(y_np[point_index])
                annot = f"Index : {point_index}\nBin Center : {px:.4f}\nValue : {py:.4f}"
                if dpg.does_item_exist(self.closest_point_anot_tag):
                    dpg.configure_item(self.closest_point_anot_tag, label=annot, default_value=(px, py))
                else:
                    dpg.add_plot_annotation(
                        tag=self.closest_point_anot_tag,
                        label=annot,
                        default_value=(px, py),
                        offset=(25, -25),
                        color=[255, 255, 0, 255],
                        parent=self.plot_tag,
                    )
            else:
                if dpg.does_item_exist(self.closest_point_anot_tag):
                    dpg.delete_item(self.closest_point_anot_tag)
        else:
            if dpg.does_item_exist(self.closest_point_anot_tag):
                dpg.delete_item(self.closest_point_anot_tag)

    def autofit_axis(self, x: bool = True, y: bool = True) -> None:
        """Fit plot axes to active data bounds."""
        if x and dpg.does_item_exist(self.xaxis_tag):
            dpg.fit_axis_data(self.xaxis_tag)
        if y and dpg.does_item_exist(self.yaxis_tag):
            dpg.fit_axis_data(self.yaxis_tag)

    def clear_plot(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Remove all series, clear plot canvas, and reset history buffers."""
        children = dpg.get_item_children(self.yaxis_tag, 1)
        if children:
            for child in children:
                item_type = dpg.get_item_type(child)
                if item_type in ("mvAppItemType::mvLineSeries", "mvAppItemType::mvShadeSeries"):
                    dpg.delete_item(child)

        if dpg.does_item_exist(self.closest_point_anot_tag):
            dpg.delete_item(self.closest_point_anot_tag)

        for uid in list(self.raw_data.keys()):
            theme_tag = f"hist_theme_{uid}_{self.UUID}"
            if dpg.does_item_exist(theme_tag):
                dpg.delete_item(theme_tag)

        self.raw_data.clear()
        self.raw_data_history.clear()
        self.hist_data.clear()
        self.selected_uuid = None

    def remove_serie(self, uid: str) -> None:
        """
        Remove a single series from memory and canvas.

        Args:
            uid: Identifier of the series to remove.
        """
        if uid in self.raw_data:
            del self.raw_data[uid]
        if uid in self.raw_data_history:
            del self.raw_data_history[uid]
        if uid in self.hist_data:
            del self.hist_data[uid]

        serie_id_line = f"hist_line_{uid}_{self.UUID}"
        serie_id_shade = f"hist_shade_{uid}_{self.UUID}"
        theme_tag = f"hist_theme_{uid}_{self.UUID}"

        if dpg.does_item_exist(serie_id_line):
            dpg.delete_item(serie_id_line)
        if dpg.does_item_exist(serie_id_shade):
            dpg.delete_item(serie_id_shade)
        if dpg.does_item_exist(theme_tag):
            dpg.delete_item(theme_tag)

        if self.selected_uuid == uid:
            self.selected_uuid = None

    def plot_serie(self, uid: str) -> None:
        """
        Compute histogram coordinates and render line/shade series for given UUID.

        Args:
            uid: Unique identifier of the series.
        """
        info = self.raw_data.get(uid)
        if not info:
            return
        data = info["data"]
        name = info["name"]

        if self.averaging and uid in self.raw_data_history and len(self.raw_data_history[uid]) > 0:
            history_list = self.raw_data_history[uid]
            combined_data = history_list[0] if len(history_list) == 1 else np.concatenate(history_list)
            if len(combined_data) == 0:
                return

            counts, bin_edges = np.histogram(combined_data, bins=self.bins, density=self.density)

            if not self.density:
                counts = counts / len(history_list)

            bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
            counts_arr = np.asarray(counts, dtype=np.float64)
        else:
            if len(data) == 0:
                return

            counts, bin_edges = np.histogram(data, bins=self.bins, density=self.density)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
            counts_arr = np.asarray(counts, dtype=np.float64)

        if self.smooth:
            counts_arr = self._smooth_curve(counts_arr, self.smooth_window)

        x_hist = bin_centers.tolist()
        y_hist = counts_arr.tolist()
        ptp_x = float(bin_centers[-1] - bin_centers[0]) if len(bin_centers) > 1 else 0.0
        ptp_y = float(np.ptp(counts_arr)) if len(counts_arr) > 1 else 0.0
        self.hist_data[uid] = {
            "x": x_hist,
            "y": y_hist,
            "x_np": bin_centers,
            "y_np": counts_arr,
            "x_scale": 1.0 / ptp_x if ptp_x != 0.0 else 1.0,
            "y_scale": 1.0 / ptp_y if ptp_y != 0.0 else 1.0,
            "name": name,
        }

        color = info.get("color")
        if not color:
            palette = (
                (31, 119, 180),
                (255, 127, 14),
                (44, 160, 44),
                (214, 39, 40),
                (148, 103, 189),
                (140, 86, 75),
                (227, 119, 194),
                (127, 127, 127),
                (188, 189, 34),
                (23, 190, 207),
            )
            keys_list = list(self.raw_data.keys())
            idx = keys_list.index(uid) if uid in keys_list else 0
            color = palette[idx % len(palette)]
            info["color"] = color

        r, g, b = int(color[0]), int(color[1]), int(color[2])

        theme_tag = f"hist_theme_{uid}_{self.UUID}"
        if not dpg.does_item_exist(theme_tag):
            with dpg.theme(tag=theme_tag):
                with dpg.theme_component(dpg.mvAll):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, (r, g, b, 255), category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_Fill, (r, g, b, 60), category=dpg.mvThemeCat_Plots)

        serie_id_line = f"hist_line_{uid}_{self.UUID}"
        serie_id_shade = f"hist_shade_{uid}_{self.UUID}"

        if self.shading:
            if dpg.does_item_exist(serie_id_shade):
                dpg.configure_item(serie_id_shade, x=x_hist, y1=y_hist, show=True)
            else:
                dpg.add_shade_series(x=x_hist, y1=y_hist, parent=self.yaxis_tag, tag=serie_id_shade)
                dpg.bind_item_theme(serie_id_shade, theme_tag)
        else:
            if dpg.does_item_exist(serie_id_shade):
                dpg.delete_item(serie_id_shade)

        if dpg.does_item_exist(serie_id_line):
            dpg.configure_item(serie_id_line, x=x_hist, y=y_hist)
            if name:
                dpg.set_item_label(serie_id_line, name)
        else:
            dpg.add_line_series(x=x_hist, y=y_hist, label=name, parent=self.yaxis_tag, tag=serie_id_line)
            dpg.bind_item_theme(serie_id_line, theme_tag)

        self.selected_uuid = uid
        if self.autoscale:
            self.autofit_axis()

    def add_raw_data(self, uid: Optional[str], data: Any, name: str) -> None:
        """
        Store raw data array and trigger distribution computation.

        Args:
            uid: Unique identifier for the series.
            data: Numerical values to bin into distribution.
            name: Series label.
        """
        if uid is None:
            uid = str(uuid.uuid4())
        arr = np.asarray(data, dtype=np.float64).ravel()
        clean = arr[np.isfinite(arr)]
        if len(clean) == 0:
            return

        existing_color = self.raw_data[uid].get("color") if uid in self.raw_data else None
        self.raw_data[uid] = {"data": clean, "name": name, "color": existing_color}

        if uid not in self.raw_data_history:
            self.raw_data_history[uid] = []
        self.raw_data_history[uid].append(clean)
        if len(self.raw_data_history[uid]) > self.avg_window:
            self.raw_data_history[uid] = self.raw_data_history[uid][-self.avg_window:]

        self.plot_serie(uid)

    def is_ready(self) -> bool:
        """
        Check if the module is ready to receive input data.

        Returns:
            bool: Always True.
        """
        return True

    def input_cb(self, *args: Any, **kwargs: Any) -> None:
        """
        Handle incoming samples or commands from upstream modules.

        Supported inputs:
            - SAMPLE / ROI_SAMPLE: Dict with keys 'x', 'y', 'name', 'uuid', 'action'.
            - CMD_DICT: Commands ('add serie', 'remove serie', 'update serie name', 'clear').
            - DATALIST: Numerical arrays.
        """
        sample = kwargs.get("sample")
        if sample:
            action = sample.get("action", "select")
            uid = sample.get("uuid")

            if action in ["unselect", "delete"]:
                if uid:
                    self.remove_serie(uid)
            elif action == "clear" or sample.get("clear"):
                self.clear_plot()
            elif action == "rename":
                new_name = sample.get("name", "Sample")
                if uid in self.raw_data:
                    self.raw_data[uid]["name"] = new_name
                    self.plot_serie(uid)
            else:
                data = sample.get("y")
                if data is None:
                    data = sample.get("x")
                name = sample.get("name", "Sample")
                if data is not None:
                    self.add_raw_data(uid, data, name)
            return

        if kwargs.get("data_type") == IOTypes.CMD_DICT:
            cmd = kwargs.get("cmd") if kwargs.get("cmd") is not None else kwargs.get("data")

            if isinstance(cmd, dict) and cmd.get("action") == "add serie":
                data = cmd.get("data", {})
                dist_data = data.get("y")
                if dist_data is None:
                    dist_data = data.get("x")
                self.add_raw_data(uid=data.get("uuid", None), data=dist_data, name=data.get("name", "Serie"))

            if isinstance(cmd, dict) and cmd.get("action") == "remove serie":
                data = cmd.get("data", {})
                uid = data.get("uuid", None)
                if uid is not None:
                    self.remove_serie(uid)

            if isinstance(cmd, dict) and cmd.get("action") == "update serie name":
                data = cmd.get("data", {})
                uid = data.get("uuid", None)
                new_name = data.get("name", None)
                if uid is not None and new_name is not None:
                    if uid in self.raw_data:
                        self.raw_data[uid]["name"] = new_name
                        self.plot_serie(uid)

            if isinstance(cmd, dict) and (cmd.get("action") == "clear" or cmd.get("clear")):
                self.clear_plot()
            return

        data = kwargs.get("y") if kwargs.get("y") is not None else kwargs.get("data")
        if data is None and args and isinstance(args[0], list):
            data = args[0]
        if data is None and kwargs.get("x") is not None:
            data = kwargs.get("x")

        name = kwargs.get("name", "Serie")
        uid = kwargs.get("uuid", None)
        if data is not None:
            self.add_raw_data(uid, data, name)

    def trigger_cb(self, out_key: Optional[str] = None, data: Any = None) -> None:
        """
        Dispatch data to all modules connected to a specific output port.

        Args:
            out_key: Output port name ("Datalist" or "DataFrame").
            data: Payload object to emit.
        """
        if out_key in self.connections:
            for module in self.connections[out_key]:
                try:
                    module.input_cb(data=data, data_type=self.outputs[out_key])
                except Exception as e:
                    logger.error(f"[{self.label}] Error triggering output {out_key}: {e}")


EXPORTED_CLASS = Histogram_win
EXPORTED_NAME = "Histogram"
