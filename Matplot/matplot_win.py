from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import cv2
import seaborn as sns
from loguru import logger

from core.window_base import WindowBase
from core.input_output_types import IOTypes
from core.app_state import app_state
from modules.plot.clipboard_injector import clipboardinjector


class Matplot_win(WindowBase):
    """
    High-resolution Matplotlib & Seaborn rendering module.

    Receives structured multi-series plot data packets (`CMD_DICT` with `action: "plot_data"`,
    e.g. emitted from `Lineplot` or `Histogram`), applies advanced styling (Seaborn themes,
    standard error bands, log scales, translucent area shading), and renders to an interactive
    Dear PyGui GPU texture and BGR video frame stream.
    """

    description: str = "High-resolution Matplotlib and Seaborn plot renderer with Frame streaming."

    plot_title: str
    xlabel: str
    ylabel: str
    dpi: int
    canvas_width: int
    canvas_height: int
    show_legend: bool
    use_seaborn: bool
    use_shading: bool

    texture_registry_tag: str
    texture_tag: str
    texture_width: int
    texture_height: int
    last_bgr_frame: Optional[np.ndarray]
    last_plot_data: Optional[Dict[str, Any]]

    def __init__(
        self,
        label: str = "Matplotlib Viewer",
        win_width: int = 800,
        win_height: int = 600,
        pos: Tuple[int, int] = (50, 50),
        uuid: Optional[str] = None,
        outputs: Optional[Dict[str, Any]] = None,
        visible: bool = True,
        plot_title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        dpi: int = 100,
        canvas_width: int = 800,
        canvas_height: int = 600,
        show_legend: bool = True,
        use_seaborn: bool = True,
        use_shading: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Matplotlib Viewer window module.

        Args:
            label: Window display title.
            win_width: Initial window width in pixels.
            win_height: Initial window height in pixels.
            pos: Initial (x, y) window position on screen.
            uuid: Optional unique module identifier.
            outputs: Optional predefined output connections dictionary.
            visible: Whether the window is visible on creation.
            plot_title: Title override for the plot figure.
            xlabel: Label override for horizontal axis.
            ylabel: Label override for vertical axis.
            dpi: Rendering dots-per-inch resolution.
            canvas_width: Width of generated image canvas in pixels.
            canvas_height: Height of generated image canvas in pixels.
            show_legend: Toggle visibility of plot legend.
            use_seaborn: Apply Seaborn styling and error bands.
            use_shading: Apply translucent under-curve area shading.
            **kwargs: Additional persistent parameters.
        """
        if outputs is None:
            outputs = {
                "Frame": IOTypes.FRAME,
            }

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

        self.accepted_input_types = [IOTypes.CMD_DICT]
        self.connections = {k: [] for k in self.outputs}

        # Plot Configuration State
        self.plot_title = plot_title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.dpi = dpi
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.show_legend = show_legend
        self.use_seaborn = use_seaborn
        self.use_shading = use_shading

        self._persistent_fields = [
            "label", "plot_title", "xlabel", "ylabel", "dpi",
            "canvas_width", "canvas_height", "show_legend", "use_seaborn", "use_shading"
        ]

        # Texture Handling
        self.texture_registry_tag = f"matplot_registry_{self.UUID}"
        self.texture_tag = f"matplot_texture_{self.UUID}"
        self.texture_width = self.canvas_width
        self.texture_height = self.canvas_height
        self.last_bgr_frame = None
        self.last_plot_data = None

        with dpg.texture_registry(tag=self.texture_registry_tag, show=False):
            dpg.add_raw_texture(
                width=self.texture_width,
                height=self.texture_height,
                default_value=np.zeros(self.texture_width * self.texture_height * 3, dtype=np.float32),
                tag=self.texture_tag,
                format=dpg.mvFormat_Float_rgb,
            )

        with dpg.window(
            label=self.label,
            width=self.win_width,
            height=self.win_height,
            pos=self.pos,
            tag=self.winID,
            show=self.visible,
        ):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Clipboard", callback=self.copy_to_clipboard)
                dpg.add_button(label="Advanced Options", callback=self._show_advanced_options)

            dpg.add_separator()
            with dpg.group(tag=f"matplot_wrapper_{self.UUID}"):
                self._build_plot()

        self.update_permission()

    def update_permission(self) -> None:
        """
        Adjust module permissions and UI visibility based on the application mode.
        """
        mode = app_state.mode
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=(mode == "user"))

    def _show_advanced_options(self, *args: Any, **kwargs: Any) -> None:
        """
        Display configuration modal for adjusting dimensions, DPI, labels, and rendering styles.
        """
        config_win = f"matplot_config_{self.UUID}"
        if dpg.does_item_exist(config_win):
            dpg.show_item(config_win)
            dpg.focus_item(config_win)
            return

        with dpg.window(
            label=f"Config - {self.label}",
            tag=config_win,
            width=300,
            height=350,
            pos=(self.pos[0] + 50, self.pos[1] + 50),
        ):
            dpg.add_input_text(label="Title", default_value=self.plot_title, callback=lambda s, a: setattr(self, "plot_title", a))
            dpg.add_input_text(label="X Label", default_value=self.xlabel, callback=lambda s, a: setattr(self, "xlabel", a))
            dpg.add_input_text(label="Y Label", default_value=self.ylabel, callback=lambda s, a: setattr(self, "ylabel", a))
            dpg.add_separator()
            dpg.add_input_int(label="Width", default_value=self.canvas_width, callback=lambda s, a: setattr(self, "canvas_width", a))
            dpg.add_input_int(label="Height", default_value=self.canvas_height, callback=lambda s, a: setattr(self, "canvas_height", a))
            dpg.add_input_int(label="DPI", default_value=self.dpi, callback=lambda s, a: setattr(self, "dpi", a))
            dpg.add_checkbox(label="Show Legend", default_value=self.show_legend, callback=lambda s, a: setattr(self, "show_legend", a))
            dpg.add_checkbox(label="Seaborn Look", default_value=self.use_seaborn, callback=lambda s, a: setattr(self, "use_seaborn", a))
            dpg.add_checkbox(label="Shading", default_value=self.use_shading, callback=lambda s, a: setattr(self, "use_shading", a))
            dpg.add_separator()
            dpg.add_button(label="Apply and Refresh", width=-1, callback=self._apply_config)

    def _build_plot(self) -> None:
        """
        Construct Dear PyGui image plot canvas matching current texture dimensions.
        """
        with dpg.plot(height=-1, width=-1, no_menus=True, tag=f"matplot_plot_{self.UUID}", parent=f"matplot_wrapper_{self.UUID}"):
            xaxis = dpg.add_plot_axis(dpg.mvXAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True)
            with dpg.plot_axis(dpg.mvYAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True) as yaxis:
                dpg.add_image_series(self.texture_tag, [0, 0], [self.texture_width, self.texture_height])
                dpg.fit_axis_data(xaxis)
                dpg.fit_axis_data(yaxis)

    def _apply_config(self, *args: Any, **kwargs: Any) -> None:
        """
        Apply dimension and style updates, reallocating texture memory if canvas size changed.
        """
        if self.canvas_width != self.texture_width or self.canvas_height != self.texture_height:
            self.texture_width = self.canvas_width
            self.texture_height = self.canvas_height

            if dpg.does_item_exist(self.texture_tag):
                dpg.delete_item(self.texture_tag)
            if dpg.does_item_exist(f"matplot_plot_{self.UUID}"):
                dpg.delete_item(f"matplot_plot_{self.UUID}")

            dpg.add_raw_texture(
                width=self.texture_width,
                height=self.texture_height,
                default_value=np.zeros(self.texture_width * self.texture_height * 3, dtype=np.float32),
                tag=self.texture_tag,
                format=dpg.mvFormat_Float_rgb,
                parent=self.texture_registry_tag,
            )
            self._build_plot()

        if self.last_plot_data:
            self.render_plot(self.last_plot_data)

    def copy_to_clipboard(self, *args: Any, **kwargs: Any) -> None:
        """
        Copy rendered figure image directly to the system clipboard.
        """
        if self.last_bgr_frame is not None:
            clipboardinjector.send_image(self.last_bgr_frame)
            logger.success("Plot copied to clipboard.")
        else:
            logger.warning("No plot available to copy.")

    def is_ready(self) -> bool:
        """
        Check if the module is ready to receive input data.

        Returns:
            bool: Always True.
        """
        return True

    def input_cb(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """
        Handle incoming command dictionaries containing plot series data.
        """
        cmd = None
        if isinstance(data, dict):
            cmd = data
        elif kwargs.get("data_type") == IOTypes.CMD_DICT:
            cmd = kwargs.get("data")

        if cmd and cmd.get("action") == "plot_data":
            self.last_plot_data = cmd
            dpg.show_item(self.winID)
            dpg.focus_item(self.winID)
            self.render_plot(cmd)

    def render_plot(self, plot_data: Dict[str, Any]) -> None:
        """
        Render multi-series dataset into Matplotlib figure and update DPG texture and Frame output.

        Args:
            plot_data: Dictionary containing series arrays, titles, and axis limit configurations.
        """
        try:
            dpi = self.dpi
            fig = Figure(figsize=(self.texture_width / dpi, self.texture_height / dpi), dpi=dpi)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            if self.use_seaborn:
                sns.set_theme(style="darkgrid", palette="muted")
            else:
                plt.rcParams.update(plt.rcParamsDefault)
                ax.grid(True)

            series = plot_data.get("series", [])
            for s in series:
                x_vals = np.array(s["x"])
                y_vals = np.array(s["y"])

                unique_x = np.unique(x_vals)
                if len(unique_x) < len(x_vals):
                    # Duplicate X coordinates found: aggregate with standard deviation bands
                    means = []
                    stds = []
                    for ux in unique_x:
                        mask = x_vals == ux
                        points = y_vals[mask]
                        means.append(np.mean(points))
                        stds.append(np.std(points))

                    means_arr = np.array(means)
                    stds_arr = np.array(stds)

                    if self.use_seaborn:
                        sns.lineplot(x=x_vals, y=y_vals, label=s["label"], ax=ax, errorbar="sd", err_style="band")
                        color = ax.lines[-1].get_color()
                    else:
                        line, = ax.plot(unique_x, means_arr, label=s["label"])
                        color = line.get_color()
                        ax.fill_between(unique_x, means_arr - stds_arr, means_arr + stds_arr, color=color, alpha=0.2)

                    if self.use_shading:
                        ax.fill_between(unique_x, means_arr, color=color, alpha=0.15)
                else:
                    if self.use_seaborn:
                        sns.lineplot(x=s["x"], y=s["y"], label=s["label"], ax=ax)
                        color = ax.lines[-1].get_color()
                    else:
                        line, = ax.plot(s["x"], s["y"], label=s["label"])
                        color = line.get_color()

                    if self.use_shading:
                        ax.fill_between(s["x"], s["y"], color=color, alpha=0.15)

            ax.set_title(self.plot_title if self.plot_title else plot_data.get("title", "Plot"))
            ax.set_xlabel(self.xlabel if self.xlabel else plot_data.get("xlabel", "X"))
            ax.set_ylabel(self.ylabel if self.ylabel else plot_data.get("ylabel", "Y"))

            x_lim = plot_data.get("x_lim")
            y_lim = plot_data.get("y_lim")
            if x_lim:
                ax.set_xlim(x_lim)
            if y_lim:
                ax.set_ylim(y_lim)

            if plot_data.get("log_x"):
                ax.set_xscale("log")

            if self.show_legend:
                ax.legend()
            ax.grid(True)

            canvas.draw()
            rgba_buffer = canvas.buffer_rgba()
            img_rgba = np.asarray(rgba_buffer)

            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
            self.last_bgr_frame = img_bgr

            self._emit_frame(img_bgr)

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            flattened = img_rgb.ravel().astype(np.float32) / 255.0
            dpg.set_value(self.texture_tag, flattened)

            logger.success("Matplotlib plot rendered & emitted (BGR).")

        except Exception as e:
            logger.error(f"Failed to render Matplotlib plot to DPG: {e}")

    def _emit_frame(self, frame: np.ndarray) -> None:
        """
        Emit rendered plot frame to connected downstream modules.
        """
        for module in self.connections.get("Frame", []):
            if hasattr(module, "input_cb"):
                try:
                    module.input_cb(data=frame, data_type=IOTypes.FRAME)
                except Exception as e:
                    logger.debug(f"Matplot_win frame emission failed: {e}")


EXPORTED_CLASS = Matplot_win
EXPORTED_NAME = "Matplotlib Viewer"
