import sys
import cv2
import numpy as np
import subprocess
from loguru import logger

# Conditional import for Windows
if sys.platform == 'win32':
    try:
        import win32clipboard
        HAS_WIN32 = True
    except ImportError:
        HAS_WIN32 = False
        logger.warning("win32clipboard not found. Clipboard features may fail on Windows.")
else:
    HAS_WIN32 = False

class ClipboardInjector:
    def send_image(self, image: np.ndarray) -> None:
        """
        Sends an image to the clipboard (cross-platform).
        - Windows: win32clipboard (CF_DIB)
        - Linux: xclip or wl-copy (image/png)
        - Mac: Not implemented (requires pyobjc or complex scripting)
        
        Parameters:
            image (numpy.ndarray): The image to be sent (BGR format expected).
        """
        if sys.platform == 'win32' and HAS_WIN32:
            self._send_image_win32(image)
        elif sys.platform.startswith('linux'):
            self._send_image_linux(image)
        else:
            logger.warning(f"Clipboard image copy not supported on {sys.platform}")

    def _send_image_win32(self, image: np.ndarray) -> None:
        """
        Windows: Send image to clipboard.
        - Provides 'PNG' format for apps that support transparency (Photoshop, Word, Discord, browsers, etc.).
        - Provides 'CF_DIB' (24-bit BGR) for legacy apps and MS Paint, which do not support 32-bit DIBs or transparency.
        """
        if image is None or image.size == 0:
            return

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # 1. Prepare 24-bit BGR image for CF_DIB (MS Paint compatibility)
        if image.ndim == 3 and image.shape[2] == 4:
            # 4 channels (BGRA/RGBA): Alpha-blend over white background to form standard 24-bit image
            b, g, r, a = cv2.split(image)
            alpha = a.astype(np.float32) / 255.0
            b_out = np.clip(b.astype(np.float32) * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
            g_out = np.clip(g.astype(np.float32) * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
            r_out = np.clip(r.astype(np.float32) * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
            dib_image = cv2.merge([b_out, g_out, r_out])
        elif image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            dib_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            dib_image = image
        else:
            dib_image = image

        # Encode 24-bit BMP and strip the 14-byte BITMAPFILEHEADER to get CF_DIB
        success_bmp, bmp_buf = cv2.imencode('.bmp', dib_image)
        dib_data = bmp_buf.tobytes()[14:] if success_bmp else None

        # Also encode original image as PNG (preserves transparency for modern apps)
        success_png, png_buf = cv2.imencode('.png', image)
        png_data = png_buf.tobytes() if success_png else None

        clipboard_opened = False
        try:
            win32clipboard.OpenClipboard()
            clipboard_opened = True
            win32clipboard.EmptyClipboard()

            # Set registered PNG format first (for apps supporting alpha transparency)
            if png_data is not None:
                try:
                    png_format = win32clipboard.RegisterClipboardFormat("PNG")
                    if png_format:
                        win32clipboard.SetClipboardData(png_format, png_data)
                except Exception as e:
                    logger.debug(f"Failed to set PNG clipboard format: {e}")

            # Set CF_DIB format (MS Paint compatible)
            if dib_data is not None:
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, dib_data)

            logger.success("Image copied to clipboard.")
        except Exception as e:
            logger.error(f"Windows clipboard error: {e}")
        finally:
            if clipboard_opened:
                try:
                    win32clipboard.CloseClipboard()
                except Exception:
                    pass

    def _send_image_linux(self, image: np.ndarray) -> None:
        """Linux: Encode as PNG, pipe to xclip (X11) or wl-copy (Wayland)."""
        success, buffer = cv2.imencode('.png', image)
        if not success: return
        data = buffer.tobytes()

        # Try Wayland first
        try:
            subprocess.run(['wl-copy', '-t', 'image/png'], input=data, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Try X11 (xclip)
        try:
            subprocess.run(['xclip', '-selection', 'clipboard', '-t', 'image/png', '-i'], input=data, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.error("Linux clipboard error: Install 'xclip' (X11) or 'wl-copy' (Wayland)")

    def send_text(self, data: str) -> None:
        """Sends text to clipboard."""
        if sys.platform == 'win32' and HAS_WIN32:
            try:
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(data)
            except Exception: pass
            finally:
                try: win32clipboard.CloseClipboard()
                except Exception: pass
        elif sys.platform == 'darwin':
            try:
                p = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
                p.communicate(input=data.encode('utf-8'))
            except Exception: pass
        elif sys.platform.startswith('linux'):
             # Try wl-copy or xclip
            try:
                subprocess.run(['wl-copy'], input=data.encode('utf-8'), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                try:
                    subprocess.run(['xclip', '-selection', 'clipboard', '-i'], input=data.encode('utf-8'), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception: pass

clipboardinjector = ClipboardInjector()
