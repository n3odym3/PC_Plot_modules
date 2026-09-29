from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

import dearpygui.dearpygui as dpg
import numpy as np
from loguru import logger

from core.window_base import WindowBase
from core.input_output_types import IOTypes
from core.app_state import app_state
from .clipboard_injector import clipboardinjector


class Lineplot_win(WindowBase):
    """
    Interactive 2D multi-series line plot module.

    Supports continuous real-time streaming, dynamic multi-axis allocation (primary and secondary right axes),
    interactive cursor hover inspection, Draggable vertical marker line, multiple smoothing algorithms,
    logarithmic X-axis scaling, and TSV clipboard export.
    """

    description: str = "Interactive multi-series 2D line plot with smoothing, multi-axis, and cursor annotation."

    autoscale: bool
    smooth: bool
    smooth_algo: str
    smooth_window: int
    log_x: bool
    selected_serie: Optional[Union[str, int]]
    raw_data: Dict[str, Dict[str, Any]]
    secondary_axes: Dict[str, Dict[str, Any]]
    series_axis: Dict[str, str]
    series_colors: Dict[str, List[int]]

    plot_tag: str
    anot_check_tag: str
    autoscale_check_tag: str
    smooth_check_tag: str
    smooth_algo_tag: str
    smooth_win_tag: str
    logX_check_tag: str
    xaxis_tag: str
    yaxis_tag: str
    legend_tag: str
    plot_popup_tag: str
    popup_content_tag: str
    closest_point_anot_tag: str
    handler_tag: str
    export_btn_tag: str
    axes_btn_tag: str
    drag_line_check_tag: str
    drag_line_tag: str

    _IMPLOT_COLORS: List[Tuple[int, int, int, int]] = [
        (76, 114, 176, 255),
        (221, 132, 82, 255),
        (85, 168, 104, 255),
        (196, 78, 82, 255),
        (129, 114, 179, 255),
        (147, 120, 96, 255),
        (218, 139, 195, 255),
        (140, 140, 140, 255),
        (204, 185, 116, 255),
        (100, 181, 205, 255),
    ]

    smooth_algo_items: List[str] = [
        "Rolling Median",
        "Rolling Mean",
        "Exponential Moving Average",
        "Savitzky-Golay",
        "Gaussian Filter",
        "Cumulative Moving Average",
    ]

    def __init__(
        self,
        label: str = "Lineplot",
        win_width: int = 300,
        win_height: int = 200,
        pos: Tuple[int, int] = (10, 10),
        uuid: Optional[str] = None,
        outputs: Optional[Dict[str, Any]] = None,
        visible: bool = True,
        autoscale: bool = True,
        smooth: bool = False,
        smooth_algo: str = "Rolling Median",
        smooth_window: int = 5,
        log_x: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Lineplot window module.

        Args:
            label: Window display title.
            win_width: Initial window width in pixels.
            win_height: Initial window height in pixels.
            pos: Initial (x, y) window position on screen.
            uuid: Optional unique module identifier.
            outputs: Optional predefined output connections dictionary.
            visible: Whether the window is visible on creation.
            autoscale: Automatically fit axes when data updates.
            smooth: Enable real-time curve smoothing.
            smooth_algo: Smoothing filter algorithm name.
            smooth_window: Smoothing window length.
            log_x: Apply logarithmic scaling to horizontal axis.
            **kwargs: Additional configuration parameters and persistent fields.
        """
        self._persistent_fields = ["label", "autoscale", "smooth", "smooth_algo", "smooth_window", "log_x"]

        super().__init__(
            label=label,
            pos=pos,
            win_width=win_width,
            win_height=win_height,
            uuid=uuid,
            outputs=outputs,
            visible=visible,
            **kwargs,
        )

        self.plot_tag = f"lineplot_plot_{self.UUID}"
        self.anot_check_tag = f"lineplot_anot_check_{self.UUID}"
        self.autoscale_check_tag = f"lineplot_autoscale_check_{self.UUID}"
        self.smooth_check_tag = f"lineplot_smooth_check_{self.UUID}"
        self.smooth_algo_tag = f"lineplot_smooth_algo_{self.UUID}"
        self.smooth_win_tag = f"lineplot_smooth_win_{self.UUID}"
        self.logX_check_tag = f"lineplot_logx_check_{self.UUID}"

        self.xaxis_tag = f"lineplot_x_axis_{self.UUID}"
        self.yaxis_tag = f"lineplot_y_axis_{self.UUID}"
        self.legend_tag = f"lineplot_legend_{self.UUID}"
        self.plot_popup_tag = f"lineplot_popup_{self.UUID}"
        self.popup_content_tag = f"lineplot_popup_content_{self.UUID}"
        self.closest_point_anot_tag = f"closest_point_anot_{self.UUID}"
        self.handler_tag = f"lineplot_handler_{self.UUID}"
        self.export_btn_tag = f"lineplot_export_btn_{self.UUID}"
        self.axes_btn_tag = f"lineplot_axes_btn_{self.UUID}"
        self.drag_line_check_tag = f"lineplot_vline_check_{self.UUID}"
        self.drag_line_tag = f"lineplot_vline_{self.UUID}"

        self.autoscale = autoscale
        self.smooth = smooth
        self.smooth_algo = smooth_algo if smooth_algo in self.smooth_algo_items else "Rolling Median"
        self.smooth_window = smooth_window
        self.log_x = log_x
        self.selected_serie = None
        self.raw_data = {}
        self.secondary_axes = {}
        self.series_axis = {}
        self.series_colors = {}

        self.accepted_input_types = [
            IOTypes.SAMPLE,
            IOTypes.ROI_SAMPLE,
            IOTypes.DATALIST,
            IOTypes.CMD_DICT,
        ]

        self.outputs = {
            "Datalist": IOTypes.DATALIST,
            "CMD": IOTypes.CMD_DICT,
        }
        self.connections = {k: [] for k in self.outputs}

        with dpg.window(
            label=self.label,
            width=self.win_width,
            height=self.win_height,
            pos=self.pos,
            tag=self.winID,
            show=self.visible,
        ):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="Anotation",
                    tag=self.anot_check_tag,
                    default_value=True,
                    callback=lambda s, a: (
                        None if a else (
                            dpg.delete_item(self.closest_point_anot_tag)
                            if dpg.does_item_exist(self.closest_point_anot_tag)
                            else None
                        )
                    ),
                )
                dpg.add_checkbox(
                    label="Drag Line",
                    tag=self.drag_line_check_tag,
                    default_value=False,
                    callback=self.toggle_drag_line,
                )
                dpg.add_checkbox(
                    label="Autoscale",
                    tag=self.autoscale_check_tag,
                    default_value=self.autoscale,
                    callback=self.toggle_autoscale,
                )
                dpg.add_checkbox(
                    label="Smooth",
                    tag=self.smooth_check_tag,
                    default_value=self.smooth,
                    callback=self.toggle_smooth,
                )
                dpg.add_combo(
                    items=self.smooth_algo_items,
                    default_value=self.smooth_algo,
                    tag=self.smooth_algo_tag,
                    width=120,
                    callback=self.update_smooth_algo,
                )
                dpg.add_drag_int(
                    label="Win",
                    tag=self.smooth_win_tag,
                    default_value=self.smooth_window,
                    min_value=1,
                    max_value=50,
                    width=50,
                    callback=self.update_smooth_win,
                )
                dpg.add_checkbox(
                    label="Log X",
                    tag=self.logX_check_tag,
                    default_value=self.log_x,
                    callback=self.toggle_log_x,
                )
                dpg.add_button(
                    label="Axes",
                    tag=self.axes_btn_tag,
                    callback=lambda: (
                        self._refresh_popup_menu(),
                        dpg.show_item(self.plot_popup_tag) if dpg.does_item_exist(self.plot_popup_tag) else None,
                    ),
                )
                with dpg.popup(parent=self.axes_btn_tag, mousebutton=dpg.mvMouseButton_Left, tag=self.plot_popup_tag):
                    with dpg.group(width=280):
                        dpg.add_text("Y-Axes Configuration", color=(150, 200, 255))
                        dpg.add_separator()
                        dpg.add_group(tag=self.popup_content_tag)
                dpg.add_button(label="Export TSV", tag=self.export_btn_tag, callback=self.export_tsv_cb)
                dpg.add_button(label="Send Plot", callback=self.send_plot_cb)
                dpg.add_button(label="Clear Plot", callback=self.clear_plot)

            with dpg.plot(label="Line Serie", height=-1, width=-1, tag=self.plot_tag):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag=self.xaxis_tag, log_scale=self.log_x)
                dpg.add_plot_axis(dpg.mvYAxis, label="Intensity (Y1)", tag=self.yaxis_tag)

            with dpg.handler_registry(tag=self.handler_tag):
                dpg.add_mouse_move_handler(callback=self.plot_change_callback)
                dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=self._on_mouse_right_click)

            dpg.configure_item(self.plot_tag, anti_aliased=True)
            dpg.configure_item(self.plot_tag, crosshairs=True)

        self.update_permission()

    def get_serie_tag(self, uid: Any) -> str:
        """
        Return consistent Dear PyGui item tag for a given series.
        """
        return f"prevplot_{self.UUID}_{uid}"

    def get_or_create_axis(self, axis_name: str) -> str:
        """
        Dynamically retrieve or create an additional Y-axis on the right plot edge.

        Args:
            axis_name: Descriptive name for the axis slot.

        Returns:
            str: Dear PyGui tag of the Y-axis.
        """
        if axis_name in ("Primary", "Primary (Left)", "Primary (Y1)", "Default"):
            return self.yaxis_tag

        if axis_name in ("Secondary 2 (Right)", "Secondary 2", "Tertiary", "Y3"):
            slot_type = dpg.mvYAxis3
            label = "Secondary 2"
            tag = f"lineplot_dyn_yaxis3_{self.UUID}"
        else:
            slot_type = dpg.mvYAxis2
            label = "Secondary 1"
            tag = f"lineplot_dyn_yaxis2_{self.UUID}"

        if not dpg.does_item_exist(tag):
            dpg.add_plot_axis(slot_type, label=label, tag=tag, parent=self.plot_tag, opposite=True)
        else:
            dpg.set_item_label(tag, label)

        self.secondary_axes[axis_name] = {"tag": tag, "slot": slot_type}
        return tag

    def get_all_y_axes(self) -> List[str]:
        """
        Return all active Y-axis tags (primary left axis plus active right secondary axes).
        """
        axes = [self.yaxis_tag]
        for info in self.secondary_axes.values():
            tag = info["tag"] if isinstance(info, dict) else info
            if dpg.does_item_exist(tag) and tag not in axes:
                axes.append(tag)
        return axes

    def cleanup_unused_axes(self) -> None:
        """
        Delete dynamically created secondary Y axes when no series are assigned to them.
        """
        active_axes = set(self.series_axis.values())
        for axis_name in list(self.secondary_axes.keys()):
            if axis_name not in active_axes:
                info = self.secondary_axes[axis_name]
                tag = info["tag"] if isinstance(info, dict) else info
                if dpg.does_item_exist(tag):
                    dpg.delete_item(tag)
                del self.secondary_axes[axis_name]

    def set_series_axis(self, uid: str, axis_name: str) -> None:
        """
        Assign a series to a designated Y-axis and refresh the plot view.
        """
        self.series_axis[uid] = axis_name
        self.replot_all()

    def reset_all_axes(self) -> None:
        """
        Reset all series back to the primary Y-axis.
        """
        self.series_axis.clear()
        self.replot_all()

    def _apply_series_color(self, serie_id: str, uid: str) -> None:
        """
        Update the line color of an existing series theme.
        """
        if uid not in self.series_colors:
            return
        color = self.series_colors[uid]
        theme_tag = f"lineplot_scolor_{self.UUID}_{uid}"
        if not dpg.does_item_exist(theme_tag):
            return
        dpg.delete_item(theme_tag, children_only=True)
        tc = dpg.add_theme_component(dpg.mvLineSeries, parent=theme_tag)
        dpg.add_theme_color(dpg.mvPlotCol_Line, color[:4], category=dpg.mvThemeCat_Plots, parent=tc)
        if dpg.does_item_exist(serie_id):
            dpg.bind_item_theme(serie_id, theme_tag)

    def _ensure_series_theme(self, serie_id: str, uid: str, color: Any) -> None:
        """
        Pre-create the styling theme for a series.
        """
        theme_tag = f"lineplot_scolor_{self.UUID}_{uid}"
        if dpg.does_item_exist(theme_tag):
            dpg.delete_item(theme_tag)
        t = dpg.add_theme(tag=theme_tag)
        tc = dpg.add_theme_component(dpg.mvLineSeries, parent=t)
        dpg.add_theme_color(dpg.mvPlotCol_Line, color[:4], category=dpg.mvThemeCat_Plots, parent=tc)
        if dpg.does_item_exist(serie_id):
            dpg.bind_item_theme(serie_id, theme_tag)

    def _on_legend_select_serie(self, sender: Any, app_data: Any, user_data: Any, *args: Any, **kwargs: Any) -> None:
        """
        Select a series for cursor inspection via the legend context menu.
        """
        self.select_serie(None, None, user_data)

    def _on_color_change(self, sender: Any, app_data: Any, user_data: Any, *args: Any, **kwargs: Any) -> None:
        """
        Apply custom color selected from legend color picker.
        """
        uid, serie_id = user_data
        raw = list(app_data)
        if isinstance(raw[0], float) and max(raw[:3]) <= 1.0:
            color = [int(c * 255) for c in raw[:3]] + [255]
        else:
            color = [int(c) for c in raw[:3]] + [255]
        self.series_colors[uid] = color
        self._apply_series_color(serie_id, uid)

    def _attach_legend_menu(self, serie_id: str, uid: str, name: str) -> None:
        """
        Attach interactive right-click context menu options to an ImPlot legend item.
        """
        series_keys = list(self.raw_data.keys())
        idx = series_keys.index(uid) if uid in series_keys else 0
        default_color = self._IMPLOT_COLORS[idx % len(self._IMPLOT_COLORS)]
        cur_color = self.series_colors.get(uid, default_color)

        self._ensure_series_theme(serie_id, uid, cur_color)

        dpg.add_selectable(
            label="Select",
            callback=self._on_legend_select_serie,
            user_data=serie_id,
            parent=serie_id,
        )
        dpg.add_separator(parent=serie_id)

        _axis_map = {"Default": "Primary (Left)", "1": "Secondary 1 (Right)", "2": "Secondary 2 (Right)"}
        _rev_map = {
            "Primary (Left)": "Default",
            "Primary": "Default",
            "Primary (Y1)": "Default",
            "Default": "Default",
            "Secondary 1 (Right)": "1",
            "Secondary": "1",
            "Secondary (Y2)": "1",
            "Secondary 1": "1",
            "Secondary (Right)": "1",
            "Secondary 2 (Right)": "2",
            "Secondary 2": "2",
            "Tertiary": "2",
            "Y3": "2",
        }

        cur = self.series_axis.get(uid, "Primary (Left)")
        cur_display = _rev_map.get(cur, "Default")

        def _on_axis_combo(sender: Any, app_data: Any, user_data: Any, *args: Any, **kwargs: Any) -> None:
            self.set_series_axis(user_data, _axis_map.get(app_data, "Primary (Left)"))

        with dpg.group(horizontal=True, parent=serie_id):
            dpg.add_text("Y axis :")
            dpg.add_combo(
                items=["Default", "1", "2"],
                default_value=cur_display,
                callback=_on_axis_combo,
                user_data=uid,
                width=110,
            )
        dpg.add_separator(parent=serie_id)

        dpg.add_color_edit(
            label="Color",
            default_value=cur_color,
            callback=self._on_color_change,
            user_data=(uid, serie_id),
            parent=serie_id,
            no_inputs=True,
            no_alpha=True,
        )

    def _refresh_popup_menu(self) -> None:
        """
        Populate the configuration popup menu with options to set target series and assign axes.
        """
        if not dpg.does_item_exist(self.popup_content_tag):
            return

        dpg.delete_item(self.popup_content_tag, children_only=True)

        if not self.raw_data:
            dpg.add_text("No active series", parent=self.popup_content_tag, color=(160, 160, 160))
            return

        dpg.add_text("Annotation Target Series:", parent=self.popup_content_tag, color=(150, 200, 255))

        series_items = []
        series_map = {}
        current_selected_name = None

        for uid, info in list(self.raw_data.items()):
            s_id = self.get_serie_tag(uid)
            s_name = info.get("name", "Serie")
            disp_name = f"{s_name} (#{str(uid)[:5]})" if len(self.raw_data) > 1 else s_name
            series_items.append(disp_name)
            series_map[disp_name] = s_id
            if self.selected_serie == s_id or self.selected_serie == uid:
                current_selected_name = disp_name

        if not current_selected_name and series_items:
            current_selected_name = series_items[0]
            self.selected_serie = series_map[current_selected_name]

        def _on_annot_serie_change(sender: Any, app_data: Any, user_data: Any, *args: Any, **kwargs: Any) -> None:
            if app_data in series_map:
                self.selected_serie = series_map[app_data]

        dpg.add_combo(
            items=series_items,
            default_value=current_selected_name,
            callback=_on_annot_serie_change,
            parent=self.popup_content_tag,
            width=-1,
        )

        dpg.add_separator(parent=self.popup_content_tag)
        dpg.add_text("Assign Y-Axis per Series:", parent=self.popup_content_tag, color=(220, 220, 220))
        dpg.add_separator(parent=self.popup_content_tag)

        axis_options = ["Primary (Left)", "Secondary 1 (Right)", "Secondary 2 (Right)"]

        for uid, info in list(self.raw_data.items()):
            cur_axis = self.series_axis.get(uid, "Primary (Left)")
            if cur_axis in ("Primary", "Primary (Y1)"):
                cur_axis = "Primary (Left)"
            elif cur_axis in ("Secondary", "Secondary (Y2)", "Secondary 1", "Secondary (Right)"):
                cur_axis = "Secondary 1 (Right)"
            elif cur_axis in ("Secondary 2", "Tertiary", "Secondary 2 (Right)"):
                cur_axis = "Secondary 2 (Right)"

            with dpg.group(horizontal=True, parent=self.popup_content_tag):
                name_display = info["name"][:14] + ("..." if len(info["name"]) > 14 else "")
                dpg.add_text(f"{name_display}:", color=(200, 200, 200))

                def _on_combo_change(sender: Any, app_data: Any, user_data: Any, *args: Any, **kwargs: Any) -> None:
                    self.set_series_axis(user_data, app_data)

                dpg.add_combo(
                    items=axis_options,
                    default_value=cur_axis if cur_axis in axis_options else axis_options[0],
                    callback=_on_combo_change,
                    user_data=uid,
                    width=160,
                )

    def update_permission(self) -> None:
        """
        Adjust module permissions and UI visibility based on application mode.
        """
        mode = app_state.mode
        elements = [
            self.anot_check_tag,
            self.drag_line_check_tag,
            self.smooth_check_tag,
            self.smooth_algo_tag,
            self.smooth_win_tag,
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

    def toggle_autoscale(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Toggle automatic axis scaling."""
        self.autoscale = bool(app_data)
        if self.autoscale:
            self.autofit_axis()

    def toggle_smooth(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Toggle real-time smoothing filter."""
        self.smooth = bool(app_data)
        self.replot_all()

    def update_smooth_algo(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Update smoothing algorithm selection."""
        self.smooth_algo = str(app_data)
        self.replot_all()

    def update_smooth_win(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Update smoothing window length."""
        self.smooth_window = max(1, int(app_data))
        self.replot_all()

    def toggle_log_x(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Toggle logarithmic scaling for horizontal axis."""
        self.log_x = bool(app_data)

        dpg.delete_item(self.xaxis_tag)
        dpg.delete_item(self.yaxis_tag)
        for info in list(self.secondary_axes.values()):
            tag = info["tag"] if isinstance(info, dict) else info
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self.secondary_axes.clear()

        dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag=self.xaxis_tag, log_scale=self.log_x, parent=self.plot_tag)
        dpg.add_plot_axis(dpg.mvYAxis, label="Intensity", tag=self.yaxis_tag, parent=self.plot_tag)

        for uid, info in self.raw_data.items():
            serie_id = self.get_serie_tag(uid)
            axis_name = self.series_axis.get(uid, "Primary (Left)")
            target_axis_tag = self.get_or_create_axis(axis_name)
            plot_y = self.apply_smoothing(info["y"])
            dpg.add_line_series(x=info["x"], y=plot_y, label=info["name"], parent=target_axis_tag, tag=serie_id)
            self._attach_legend_menu(serie_id, uid, info["name"])
            self._apply_series_color(serie_id, uid)

        if self.autoscale:
            self.autofit_axis()

    def toggle_drag_line(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Toggle vertical marker line visibility."""
        if not app_data and dpg.does_item_exist(self.drag_line_tag):
            dpg.delete_item(self.drag_line_tag)

    def export_tsv_cb(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Export all plot data to clipboard in TSV format."""
        try:
            series_list = []
            for axis_tag in self.get_all_y_axes():
                children = dpg.get_item_children(axis_tag, 1)
                if children:
                    for child in children:
                        if dpg.get_item_type(child) == "mvAppItemType::mvLineSeries":
                            data = dpg.get_value(child)
                            label = dpg.get_item_label(child)
                            series_list.append({
                                "label": label,
                                "x": list(data[0]) if data[0] is not None else [],
                                "y": list(data[1]) if data[1] is not None else [],
                            })

            if not series_list:
                return

            series_dicts = []
            all_x_set = set()
            for s in series_list:
                s_map = {}
                for xi, yi in zip(s["x"], s["y"]):
                    s_map[xi] = yi
                    all_x_set.add(xi)
                series_dicts.append(s_map)

            sorted_x = sorted(list(all_x_set))
            max_rows = max(len(sorted_x), max((len(s["x"]) for s in series_list), default=0))

            tsv_lines = []
            headers = ["X (Aligned)"] + [s["label"] for s in series_list]
            for s in series_list:
                headers.append(f"{s['label']} (X)")
                headers.append(f"{s['label']} (Y)")

            tsv_lines.append("\t".join(headers))

            for r in range(max_rows):
                row = []
                if r < len(sorted_x):
                    cur_x = sorted_x[r]
                    row.append(str(cur_x))
                    for s_map in series_dicts:
                        row.append(str(s_map[cur_x]) if cur_x in s_map else "")
                else:
                    row.append("")
                    for _ in series_dicts:
                        row.append("")

                for s in series_list:
                    row.append(str(s["x"][r]) if r < len(s["x"]) else "")
                    row.append(str(s["y"][r]) if r < len(s["y"]) else "")

                tsv_lines.append("\t".join(row))

            tsv_content = "\n".join(tsv_lines)
            clipboardinjector.send_text(tsv_content)
            logger.info(f"Exported {len(series_list)} series to TSV")
        except Exception as e:
            logger.error(f"Error exporting TSV: {e}")

    def send_plot_cb(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """Extract all displayed series and emit plot packet via CMD output."""
        try:
            all_series = []
            for axis_tag in self.get_all_y_axes():
                children = dpg.get_item_children(axis_tag, 1)
                if children:
                    for child in children:
                        if dpg.get_item_type(child) == "mvAppItemType::mvLineSeries":
                            data = dpg.get_value(child)
                            label = dpg.get_item_label(child)
                            all_series.append({
                                "label": label,
                                "x": list(data[0]),
                                "y": list(data[1]),
                            })

            if not all_series:
                logger.warning("No data to send")
                return

            payload = {
                "action": "plot_data",
                "series": all_series,
                "title": self.label,
                "xlabel": dpg.get_item_label(self.xaxis_tag),
                "ylabel": dpg.get_item_label(self.yaxis_tag),
                "x_lim": dpg.get_axis_limits(self.xaxis_tag),
                "y_lim": dpg.get_axis_limits(self.yaxis_tag),
                "log_x": dpg.get_value(self.logX_check_tag),
            }

            self.trigger_cb(out_key="CMD", data=payload)
            logger.info(f"Sent {len(all_series)} series to CMD output")
        except Exception as e:
            logger.error(f"Error sending plot: {e}")

    def _smooth_rolling_median(self, arr: np.ndarray, window: int) -> List[float]:
        w = max(1, int(window))
        if w == 1 or len(arr) <= 1:
            return arr.tolist()
        if w >= len(arr):
            return [float(np.median(arr))] * len(arr)

        try:
            import pandas as pd
            s = pd.Series(arr)
            res = s.rolling(window=w, min_periods=1, center=True).median().to_numpy()
            return res.tolist()
        except ImportError:
            from scipy.signal import medfilt
            eff_w = w if w % 2 != 0 else w + 1
            if eff_w > len(arr):
                eff_w = len(arr) if len(arr) % 2 != 0 else len(arr) - 1
            if eff_w < 3:
                return arr.tolist()
            return medfilt(arr, kernel_size=eff_w).tolist()

    def _smooth_rolling_mean(self, arr: np.ndarray, window: int) -> List[float]:
        w = max(1, int(window))
        if w >= len(arr):
            return [float(np.mean(arr))] * len(arr)
        if w <= 1 or len(arr) <= 1:
            return arr.tolist()

        try:
            import pandas as pd
            s = pd.Series(arr)
            res = s.rolling(window=w, min_periods=1, center=True).mean().to_numpy()
            return res.tolist()
        except ImportError:
            half = w // 2
            pad_width = (half, w - 1 - half)
            padded = np.pad(arr, pad_width, mode="edge")
            kernel = np.ones(w) / w
            smoothed = np.convolve(padded, kernel, mode="valid")
            return smoothed.tolist()

    def _smooth_ema(self, arr: np.ndarray, window: int) -> List[float]:
        if len(arr) <= 1:
            return arr.tolist()
        span = max(1.0, float(window))
        alpha = 2.0 / (span + 1.0)
        out = np.empty_like(arr, dtype=float)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
        return out.tolist()

    def _smooth_savgol(self, arr: np.ndarray, window: int) -> List[float]:
        n = len(arr)
        if n < 3:
            return arr.tolist()
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            logger.error("scipy is required for Savitzky-Golay filter.")
            return arr.tolist()

        w = max(3, int(window))
        if w % 2 == 0:
            w += 1
        if w > n:
            w = n if n % 2 != 0 else n - 1
        if w < 3:
            return arr.tolist()

        poly = min(2, w - 1)
        try:
            filtered = savgol_filter(arr, window_length=w, polyorder=poly, mode="interp")
            return filtered.tolist()
        except Exception as e:
            logger.warning(f"Savitzky-Golay fallback triggered: {e}")
            return arr.tolist()

    def _smooth_gaussian(self, arr: np.ndarray, window: int) -> List[float]:
        if len(arr) <= 1:
            return arr.tolist()
        sigma = max(0.1, float(window) / 2.0)
        try:
            from scipy.ndimage import gaussian_filter1d
            res = gaussian_filter1d(arr, sigma=sigma, mode="nearest")
            return res.tolist()
        except ImportError:
            radius = max(1, int(window))
            x_k = np.arange(-radius, radius + 1)
            kernel = np.exp(-0.5 * (x_k / sigma) ** 2)
            kernel /= kernel.sum()
            pad_y = np.pad(arr, radius, mode="edge")
            smoothed = np.convolve(pad_y, kernel, mode="valid")
            return smoothed.tolist()

    def _smooth_cma(self, arr: np.ndarray) -> List[float]:
        if len(arr) <= 0:
            return []
        counts = np.arange(1, len(arr) + 1, dtype=float)
        res = np.cumsum(arr, dtype=float) / counts
        return res.tolist()

    def apply_smoothing(self, y_data: Any) -> List[float]:
        """
        Apply active smoothing filter algorithm to 1D y_data array.
        """
        if not self.smooth or len(y_data) <= 1:
            return list(y_data)

        arr = np.asarray(y_data, dtype=float)
        if self.smooth_algo == "Rolling Median":
            return self._smooth_rolling_median(arr, self.smooth_window)
        elif self.smooth_algo == "Rolling Mean":
            return self._smooth_rolling_mean(arr, self.smooth_window)
        elif self.smooth_algo == "Exponential Moving Average":
            return self._smooth_ema(arr, self.smooth_window)
        elif self.smooth_algo == "Savitzky-Golay":
            return self._smooth_savgol(arr, self.smooth_window)
        elif self.smooth_algo == "Gaussian Filter":
            return self._smooth_gaussian(arr, self.smooth_window)
        elif self.smooth_algo == "Cumulative Moving Average":
            return self._smooth_cma(arr)
        else:
            return self._smooth_rolling_median(arr, self.smooth_window)

    def replot_all(self) -> None:
        """
        Recompute smoothed curves and update all series across their designated Y-axes.
        """
        for uid, info in list(self.raw_data.items()):
            x = info["x"]
            y = info["y"]
            name = info["name"]
            plot_y = self.apply_smoothing(y)
            serie_id = self.get_serie_tag(uid)
            axis_name = self.series_axis.get(uid, "Primary (Left)")
            target_axis_tag = self.get_or_create_axis(axis_name)

            if dpg.does_item_exist(serie_id):
                current_parent = dpg.get_item_info(serie_id).get("parent")
                if current_parent != target_axis_tag:
                    dpg.delete_item(serie_id)
                    dpg.add_line_series(x=x, y=plot_y, label=name, parent=target_axis_tag, tag=serie_id)
                    self._attach_legend_menu(serie_id, uid, name)
                    self._apply_series_color(serie_id, uid)
                else:
                    dpg.configure_item(serie_id, x=x, y=plot_y)
                    if name:
                        dpg.set_item_label(serie_id, name)
                    self._apply_series_color(serie_id, uid)
            else:
                dpg.add_line_series(x=x, y=plot_y, label=name, parent=target_axis_tag, tag=serie_id)
                self._attach_legend_menu(serie_id, uid, name)
                self._apply_series_color(serie_id, uid)

        self.cleanup_unused_axes()
        if self.autoscale:
            self.autofit_axis()
        self._refresh_popup_menu()

    def find_closest_point(
        self,
        mouse_x: float,
        mouse_y: float,
        x_data: np.ndarray,
        y_data: np.ndarray,
    ) -> Optional[int]:
        """
        Find point index closest to given mouse coordinates.
        """
        if len(x_data) == 0 or len(y_data) == 0:
            return None
        ptp_x = np.ptp(x_data)
        ptp_y = np.ptp(y_data)
        x_scale = 1.0 / ptp_x if ptp_x != 0 else 1.0
        y_scale = 1.0 / ptp_y if ptp_y != 0 else 1.0
        distances = np.hypot((x_data - mouse_x) * x_scale, (y_data - mouse_y) * y_scale)
        return int(np.argmin(distances))

    def select_serie(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """
        Select active series for on-hover cursor annotation.
        """
        self.selected_serie = user_data

    def _on_mouse_right_click(self, sender: Any = None, app_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Remove plot annotation on right-click to avoid blocking ImPlot legend context menus.
        """
        if dpg.does_item_exist(self.closest_point_anot_tag):
            dpg.delete_item(self.closest_point_anot_tag)

    def _is_mouse_over_legend(self) -> bool:
        """
        Detect if cursor is hovering near the top-right legend area.
        """
        if not dpg.does_item_exist(self.plot_tag):
            return False
        try:
            p_min = dpg.get_item_rect_min(self.plot_tag)
            p_max = dpg.get_item_rect_max(self.plot_tag)
            mx, my = dpg.get_mouse_pos()
            plot_w = p_max[0] - p_min[0]
            plot_h = p_max[1] - p_min[1]
            if plot_w <= 0 or plot_h <= 0:
                return False
            num_series = len(self.raw_data)
            legend_w = min(320, plot_w * 0.45)
            legend_h = min(400, max(120, 35 * (num_series + 1) + 60))
            return (p_max[0] - legend_w <= mx <= p_max[0]) and (p_min[1] <= my <= p_min[1] + legend_h)
        except Exception:
            return False

    def plot_change_callback(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any) -> None:
        """
        Update on-hover point inspection tooltip when moving mouse across plot canvas.
        """
        if not dpg.is_item_hovered(self.plot_tag) or dpg.is_mouse_button_down(dpg.mvMouseButton_Right) or self._is_mouse_over_legend():
            if dpg.does_item_exist(self.closest_point_anot_tag):
                dpg.delete_item(self.closest_point_anot_tag)
            return

        if not dpg.get_value(self.anot_check_tag):
            if dpg.does_item_exist(self.closest_point_anot_tag):
                dpg.delete_item(self.closest_point_anot_tag)
            return

        all_line_series = []
        for axis_tag in self.get_all_y_axes():
            children = dpg.get_item_children(axis_tag, 1)
            if children:
                for child in children:
                    if dpg.get_item_type(child) == "mvAppItemType::mvLineSeries":
                        all_line_series.append(child)

        if all_line_series:
            if self.selected_serie is not None and dpg.does_item_exist(self.selected_serie) and dpg.get_item_type(self.selected_serie) == "mvAppItemType::mvLineSeries":
                plot = dpg.get_value(self.selected_serie)
                serie_label = dpg.get_item_label(self.selected_serie)
            else:
                plot = dpg.get_value(all_line_series[0])
                self.selected_serie = all_line_series[0]
                serie_label = dpg.get_item_label(all_line_series[0])

            if plot and len(plot) >= 2 and plot[0] is not None and plot[1] is not None:
                xplot = plot[0]
                yplot = plot[1]
                mouse_x, mouse_y = dpg.get_plot_mouse_pos()
                point_index = self.find_closest_point(mouse_x, mouse_y, np.array(xplot), np.array(yplot))

                if point_index is not None:
                    annot = f"[{serie_label}]\nIndex : {point_index}\nX : {xplot[point_index]:.4g}\nY : {yplot[point_index]:.4g}"
                    if dpg.does_item_exist(self.closest_point_anot_tag):
                        dpg.configure_item(self.closest_point_anot_tag, label=annot, default_value=(xplot[point_index], yplot[point_index]))
                    else:
                        dpg.add_plot_annotation(
                            tag=self.closest_point_anot_tag,
                            label=annot,
                            default_value=(xplot[point_index], yplot[point_index]),
                            offset=(25, -25),
                            color=[255, 255, 0, 255],
                            parent=self.plot_tag,
                        )
                else:
                    if dpg.does_item_exist(self.closest_point_anot_tag):
                        dpg.delete_item(self.closest_point_anot_tag)

    def autofit_axis(self, x: bool = True, y: bool = True) -> None:
        """
        Fit plot horizontal and vertical axes to active data bounds.
        """
        if x and dpg.does_item_exist(self.xaxis_tag):
            dpg.set_axis_limits_auto(self.xaxis_tag)
            dpg.fit_axis_data(self.xaxis_tag)
        if y:
            for axis_tag in self.get_all_y_axes():
                if dpg.does_item_exist(axis_tag):
                    dpg.set_axis_limits_auto(axis_tag)
                    dpg.fit_axis_data(axis_tag)

    def clear_plot(self, sender: Any = None, app_data: Any = None, user_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Remove all line series across all Y-axes, clear dynamic secondary axes, and reset selection state.
        """
        for axis_tag in self.get_all_y_axes():
            children = dpg.get_item_children(axis_tag, 1)
            if children:
                for child in children:
                    if dpg.get_item_type(child) == "mvAppItemType::mvLineSeries":
                        dpg.delete_item(child)

        for info in list(self.secondary_axes.values()):
            tag = info["tag"] if isinstance(info, dict) else info
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self.secondary_axes.clear()
        self.series_axis.clear()

        for uid in list(self.series_colors.keys()):
            theme_tag = f"lineplot_scolor_{self.UUID}_{uid}"
            if dpg.does_item_exist(theme_tag):
                dpg.delete_item(theme_tag)
        self.series_colors.clear()

        if dpg.does_item_exist(self.closest_point_anot_tag):
            dpg.delete_item(self.closest_point_anot_tag)

        self.raw_data.clear()
        self.selected_serie = None
        self._refresh_popup_menu()

    def plot_data(self, x: Any = None, y: Any = None, name: str = "Serie", UUID: Any = None) -> None:
        """
        Add or update a continuous 2D data series on the plot.

        Args:
            x: Array of X coordinates.
            y: Array of Y values.
            name: Display label for the series.
            UUID: Optional unique identifier.
        """
        if y is None:
            return
        if x is None:
            x = list(range(len(y)))

        x_list = list(x)
        y_list = list(y)
        uid = str(UUID) if UUID is not None else name

        self.raw_data[uid] = {
            "x": x_list,
            "y": y_list,
            "name": name,
            "uuid": UUID,
        }

        plot_y = self.apply_smoothing(y_list)
        serie_id = self.get_serie_tag(uid)
        axis_name = self.series_axis.get(uid, "Primary (Left)")
        target_axis_tag = self.get_or_create_axis(axis_name)

        if dpg.does_item_exist(serie_id):
            current_parent = dpg.get_item_info(serie_id).get("parent")
            if current_parent != target_axis_tag:
                dpg.delete_item(serie_id)
                dpg.add_line_series(x=x_list, y=plot_y, label=name, parent=target_axis_tag, tag=serie_id)
                self._attach_legend_menu(serie_id, uid, name)
                self._apply_series_color(serie_id, uid)
            else:
                dpg.configure_item(serie_id, x=x_list, y=plot_y)
                if name:
                    dpg.set_item_label(serie_id, name)
                self._apply_series_color(serie_id, uid)
        else:
            dpg.add_line_series(x=x_list, y=plot_y, label=name, parent=target_axis_tag, tag=serie_id)
            self._attach_legend_menu(serie_id, uid, name)
            self._apply_series_color(serie_id, uid)

        if self.autoscale:
            self.autofit_axis()

        if self.selected_serie is None or not dpg.does_item_exist(self.selected_serie):
            self.select_serie(None, None, serie_id)
        self._refresh_popup_menu()

    def is_ready(self) -> bool:
        """
        Check if the module is ready to receive input data.

        Returns:
            bool: Always True.
        """
        return True

    def input_cb(self, *args: Any, **kwargs: Any) -> None:
        """
        Handle incoming samples or command dictionaries from upstream modules.

        Supported inputs:
            - SAMPLE / ROI_SAMPLE: Dict with keys 'x', 'y', 'name', 'uuid', 'action'.
            - CMD_DICT: Commands ('position', 'add serie', 'remove serie', 'update serie name', 'clear').
            - DATALIST: Numerical arrays (x, y).
        """
        sample = kwargs.get("sample")
        if sample:
            action = sample.get("action", "select")
            uid = sample.get("uuid")
            uid = uid if uid is not None else sample.get("name", "Sample")
            serie_id = self.get_serie_tag(uid)

            if action in ["unselect", "delete"]:
                if uid in self.raw_data:
                    del self.raw_data[uid]
                if dpg.does_item_exist(serie_id):
                    dpg.delete_item(serie_id)
                self.cleanup_unused_axes()
            elif action == "clear" or sample.get("clear"):
                self.clear_plot()
            elif action == "rename":
                new_name = sample.get("name", "Sample")
                if uid in self.raw_data:
                    self.raw_data[uid]["name"] = new_name
                if dpg.does_item_exist(serie_id):
                    dpg.set_item_label(serie_id, new_name)
            else:
                self.plot_data(
                    x=sample.get("x"),
                    y=sample.get("y"),
                    name=sample.get("name", "Sample"),
                    UUID=uid,
                )
            return

        if kwargs.get("data_type") == IOTypes.CMD_DICT:
            cmd = kwargs.get("cmd") if kwargs.get("cmd") is not None else kwargs.get("data")

            if isinstance(cmd, dict) and cmd.get("action") == "position":
                position = cmd.get("value")
                if dpg.does_item_exist(self.drag_line_check_tag) and not dpg.get_value(self.drag_line_check_tag):
                    return
                if position is not None:
                    x_val = float(position)
                    if dpg.does_item_exist(self.drag_line_tag):
                        dpg.set_value(self.drag_line_tag, x_val)
                    else:
                        dpg.add_drag_line(
                            default_value=x_val,
                            tag=self.drag_line_tag,
                            parent=self.plot_tag,
                            color=[255, 0, 0, 255],
                            thickness=2,
                        )
                else:
                    if dpg.does_item_exist(self.drag_line_tag):
                        dpg.delete_item(self.drag_line_tag)
                return

            if cmd and cmd.get("action") == "add serie":
                data = cmd.get("data", {})
                self.plot_data(
                    x=data.get("x", None),
                    y=data.get("y", None),
                    name=data.get("name", None),
                    UUID=data.get("uuid", None),
                )

            if cmd and cmd.get("action") == "remove serie":
                data = cmd.get("data", {})
                UUID = data.get("uuid", None)
                uid = UUID if UUID is not None else data.get("name", "Serie")
                if uid in self.raw_data:
                    del self.raw_data[uid]
                serie_id = self.get_serie_tag(uid)
                if dpg.does_item_exist(serie_id):
                    dpg.delete_item(serie_id)
                self.cleanup_unused_axes()

            if cmd and cmd.get("action") == "update serie name":
                data = cmd.get("data", {})
                UUID = data.get("uuid", None)
                uid = UUID if UUID is not None else data.get("name", "Serie")
                new_name = data.get("name", None)
                if uid in self.raw_data and new_name is not None:
                    self.raw_data[uid]["name"] = new_name
                serie_id = self.get_serie_tag(uid)
                if new_name is not None and dpg.does_item_exist(serie_id):
                    dpg.set_item_label(serie_id, new_name)

            if cmd and (cmd.get("action") == "clear" or cmd.get("clear")):
                self.clear_plot()
            return

        y = kwargs.get("y") if kwargs.get("y") is not None else (args[0] if args and isinstance(args[0], list) else None)
        x = kwargs.get("x") if kwargs.get("x") is not None else (args[1] if len(args) > 1 and isinstance(args[1], list) else None)
        name = kwargs.get("name", "Serie")
        UUID = kwargs.get("uuid", None)
        self.plot_data(x=x, y=y, name=name, UUID=UUID)

    def trigger_cb(self, out_key: Optional[str] = None, data: Any = None) -> None:
        """
        Emit data payload to all connected downstream modules.

        Args:
            out_key: Output port name ("Datalist" or "CMD").
            data: Payload object to emit.
        """
        if out_key in self.connections:
            for module in self.connections[out_key]:
                try:
                    module.input_cb(data=data, data_type=self.outputs[out_key])
                except Exception as e:
                    logger.error(f"Error triggering output {out_key}: {e}")


EXPORTED_CLASS = Lineplot_win
EXPORTED_NAME = "Lineplot"
