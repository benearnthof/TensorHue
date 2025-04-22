from __future__ import annotations  # backwards type hint compatability for Python 3.7 and 3.8
import os
import sys
import time
import math
import warnings
import readchar
from rich.console import Console
from rich.live import Live
import numpy as np
from tensorhue.colors import ColorScheme
from tensorhue._print_opts import PRINT_OPTS
from tensorhue.converters import tensor_to_numpy, mro_to_strings



def viz(tensor, **kwargs):
    try:
        mro_strings = mro_to_strings(tensor.__class__.__mro__)
        if "PIL.Image.Image" in mro_strings:
            _viz_image(tensor, **kwargs)
        else:
            _viz(tensor, **kwargs)
    except Exception as e:
        raise NotImplementedError(
            f"TensorHue currently does not support type {type(tensor)}. Please raise an issue if you want to visualize them.."
        ) from e


def _viz(tensor, colorscheme: ColorScheme = None, legend: bool = True, scale: int = 1, **kwargs):
    """
    Prints a tensor using colored Unicode art representation.

    Args:
        tensor (Any): The tensor to be visualized.
        colorscheme (ColorScheme, optional): The color scheme to use.
            Defaults to None, which means the global default color scheme is used.
        legend (bool, optional): Whether or not to include legend information (like the shape)
        scale (int, optional): Scales the size of the entire tensor up, making the unicode 'pixels' larger.
        **kwargs: Additional keyword arguments that are passed to the underlying viz function (vmin or vmax)
    """
    if not isinstance(scale, int):
        raise ValueError("scale must be an integer.")

    if colorscheme is None:
        colorscheme = PRINT_OPTS.colorscheme

    np_array = tensor_to_numpy(tensor)
    shape = np_array.shape
    ndim = np_array.ndim

    if ndim == 1:
        np_array = np_array[np.newaxis, :]
    elif ndim > 2:
        raise NotImplementedError(
            "Visualization of tensors with more than 2 dimensions is under development. Please slice them for now."
        )

    np_array = np.repeat(np.repeat(np_array, scale, axis=1), scale, axis=0)

    result_lines = _viz_2d(np_array, colorscheme, **kwargs)

    if legend:
        result_lines.append(f"[italic]shape = {shape}[/]")

    c = Console(log_path=False, record=False)
    c.print("\n".join(result_lines))


def _viz_2d(array_2d: np.ndarray, colorscheme: ColorScheme = None, **kwargs) -> list[str]:
    """
    Constructs a list of rich-compatible strings out of a 2D numpy array.

    Args:
        array_2d (np.ndarray): The 2-dimensional numpy array (or 3-dimensional if the values are already RGB).
        colorscheme (ColorScheme): The color scheme to use. If None, the array must be 3-dimensional (already RGB values).
        **kwargs: Additional keyword arguments that are passed to the underlying viz function (vmin or vmax)
    """
    terminal_width = get_terminal_size().columns
    shape = array_2d.shape

    if shape[1] > terminal_width:
        slice_left = (terminal_width - 5) // 2
        slice_right = slice_left + (terminal_width - 5) % 2
        if colorscheme is not None:
            colors_right = colorscheme(array_2d[:, -slice_right:])[..., :3]
        else:
            assert (
                array_2d.ndim == 3 and array_2d.shape[-1] == 3
            ), "Array shape must be 3-dimensional (*, *, 3) when colorscheme=None."
            colors_right = array_2d[:, -slice_right:, :]
    else:
        slice_left = shape[1]
        slice_right = colors_right = False

    if colorscheme is not None:
        colors_left = colorscheme(array_2d[:, :slice_left], **kwargs)[..., :3]
    else:
        assert (
            array_2d.ndim == 3 and array_2d.shape[-1] == 3
        ), "Array shape must be 3-dimensional (*, *, 3) when colorscheme=None."
        colors_left = array_2d[:, :slice_left, :]

    result_lines = _construct_unicode_string(colors_left, colors_right)

    return result_lines


def _construct_unicode_string(colors_left: np.ndarray, colors_right: np.ndarray) -> str:
    result_lines = [""]

    for y in range(0, colors_left.shape[0] - 1, 2):
        for x in range(colors_left.shape[1]):
            result_lines[
                -1
            ] += f"[rgb({colors_left[y, x, 0]},{colors_left[y, x, 1]},{colors_left[y, x, 2]}) on rgb({colors_left[y+1, x, 0]},{colors_left[y+1, x, 1]},{colors_left[y+1, x, 2]})]▀[/]"
        if isinstance(colors_right, np.ndarray):
            result_lines[-1] += " ··· "
            for x in range(colors_right.shape[1]):
                result_lines[
                    -1
                ] += f"[rgb({colors_right[y, x, 0]},{colors_right[y, x, 1]},{colors_right[y, x, 2]}) on rgb({colors_right[y+1, x, 0]},{colors_right[y+1, x, 1]},{colors_right[y+1, x, 2]})]▀[/]"
        result_lines.append("")

    if colors_left.shape[0] % 2 == 1:
        for x in range(colors_left.shape[1]):
            result_lines[-1] += f"[rgb({colors_left[-1, x, 0]},{colors_left[-1, x, 1]},{colors_left[-1, x, 2]})]▀[/]"
        if isinstance(colors_right, np.ndarray):
            result_lines[-1] += " ··· "
            for x in range(colors_right.shape[1]):
                result_lines[
                    -1
                ] += f"[rgb({colors_right[-1, x, 0]},{colors_right[-1, x, 1]},{colors_right[-1, x, 2]})]▀[/]"
    else:
        result_lines = result_lines[:-1]

    return result_lines


def _viz_image(image, legend: bool = False, thumbnail: bool = True, max_size: tuple[int, int] = None):
    """
    A special case of _viz that does not use the ColorScheme but instead treats the tensor as RGB or greyscale values directly.

    Args:
        image (PIL.Image.Image): The image to visualize
        legend (bool, optional): Whether or not to include legend information (like the shape)
        thumbnail (bool, optional): Scales down the image size to a thumbnail that fits into the terminal window
        max_size (tuple[int, int], optional): The maximum size (width, height) to which the image gets downsized to. Only used if thumbnail=True.
    """

    raise_max_size_warning = max_size and not thumbnail

    size = image.size
    mode = image.mode
    if max_size is None:
        terminal_size = get_terminal_size()
    else:
        terminal_size = os.terminal_size(max_size)
    max_size = (terminal_size.columns, (terminal_size.lines - 1) * 2)
    image = tensor_to_numpy(image, thumbnail=thumbnail, max_size=max_size)

    result_lines = _viz_2d(image)

    if legend:
        result_lines.append(f"[italic]size = {size}[/], [italic]mode = {mode}[/]")

    c = Console(log_path=False, record=False)
    c.print("\n".join(result_lines))

    if raise_max_size_warning:
        warnings.warn(
            "You specified a max_size, but set thumbnail to False. Your max_size will be ignored unless thumbnail=True."
        )


def get_terminal_size(default_width: int = 100, default_height: int = 70) -> os.terminal_size:
    """
    Returns the terminal size if the standard output is connected to a terminal. Otherwise, returns the defined default size.

    Args:
        default_width (int, optional): The default width to use if there is no terminal connected.
        default_height (int, optional): The default height to use if there is no terminal connected.
    """
    if sys.stdout.isatty():
        try:
            return os.get_terminal_size()
        except OSError:
            return os.terminal_size((default_width, default_height))
    else:
        return os.terminal_size((default_width, default_height))

# Visualizing volumetric data (For MRI or videos)
def viz_volume(tensor, axis=0, pause=0.1, colorscheme: ColorScheme = None, scale=1, legend=True, stride=1, **kwargs):
    """
    Visualizes a 3D tensor as an animation through slices.

    Args:
        tensor (Tensor or ndarray): 3D tensor to visualize (e.g. shape [D, H, W]).
        axis (int): Axis along which to slice.
        pause (float): Time to wait between frames in seconds.
        colorscheme (ColorScheme, optional): Color scheme to use.
        scale (int): Scaling factor for slice visualization.
        legend (bool): Show slice index and shape.
        **kwargs: Passed to color scheme.
    """
    np_array = tensor_to_numpy(tensor)
    
    if np_array.ndim != 3:
        raise ValueError("Only 3D tensors can be passed to viz_volume")

    np_array = np.moveaxis(np_array, axis, 0)  # Make slice axis first
    n_slices = np_array.shape[0]
    console = Console()
    with Live(console=console, refresh_per_second=1 / pause, screen=False) as live:
        for i in range(0, n_slices, stride):
            slice_2d = np_array[i]
            scaled = np.repeat(np.repeat(slice_2d, scale, axis=1), scale, axis=0)
            lines = _viz_2d(scaled, colorscheme, **kwargs)
            if legend:
               lines.append(f"[italic]slice = {i}/{n_slices - 1}, shape = {slice_2d.shape}[/]")
               # lines.append(f"[italic]shape = {shape}[/]")
            live.update("\n".join(lines))
            time.sleep(pause)


def viz_volume_manual(tensor, axis=0, colorscheme: ColorScheme = None, scale=1, legend=True, stride=1, **kwargs):
    """
    Interactive 3D tensor viewer using keyboard input.

    Keys:
    [a] ← : previous slice
    [d] → : next slice
    [q]   : quit
    """
    np_array = tensor_to_numpy(tensor)

    if np_array.ndim != 3:
        raise ValueError("Only 3D tensors can be passed to viz_volume_manual")

    np_array = np.moveaxis(np_array, axis, 0)
    n_slices = np_array.shape[0]
    idx = 0

    console = Console()

    def get_render(idx):
        slice_2d = np_array[idx]
        scaled = np.repeat(np.repeat(slice_2d, scale, axis=1), scale, axis=0)
        lines = _viz_2d(scaled, colorscheme, **kwargs)
        if legend:
            lines.append(f"[italic]slice = {idx}/{n_slices - 1}, shape = {slice_2d.shape}[/]")
            lines.append("\u25C4 a |q quit| d \u25BA")
        return "\n".join(lines)

    with Live(get_render(idx), console=console, screen=False, auto_refresh=False) as live:
        while True:
            key = readchar.readkey()
            new_idx = idx
            if key == "q":
                break
            elif key in ("d", readchar.key.RIGHT):
                new_idx = (idx + stride) % n_slices
            elif key in ("a", readchar.key.LEFT):
                new_idx = (idx - stride) % n_slices
            if new_idx != idx:
                idx = new_idx
                live.update(get_render(idx), refresh=True)


def viz_batch_volume(tensor, axis=0, pause=0.1, colorscheme: ColorScheme = None, scale=1, legend=True, stride=1, **kwargs):
    """
    Animated viewer for a batch of 3D tensors.

    Args:
        tensor (Tensor or ndarray): 4D tensor of shape [B, D, H, W].
        axis (int): Axis within each volume to slice along.
        pause (float): Time to wait between frames (in seconds).
        colorscheme: Optional color scheme.
        scale (int): Visual scaling factor.
        legend (bool): Show slice index.
        stride (int): Step size between slices.
    """
    np_array = tensor_to_numpy(tensor)

    if np_array.ndim != 4:
        raise ValueError("Tensor must be 4D [B, D, H, W]")

    B = np_array.shape[0]
    volumes = [np.moveaxis(np_array[b], axis, 0) for b in range(B)]
    n_slices = volumes[0].shape[0]

    console = Console()

    grid_rows = int(math.floor(math.sqrt(B)))
    grid_cols = int(math.ceil(B / grid_rows))

    def get_render(idx):
        rendered = []

        for b in range(B):
            slice_2d = volumes[b][idx]
            scaled = np.repeat(np.repeat(slice_2d, scale, axis=1), scale, axis=0)
            lines = _viz_2d(scaled, colorscheme, **kwargs)
            rendered.append(lines)

        max_height = max(len(r) for r in rendered)
        for r in rendered:
            while len(r) < max_height:
                r.append("")

        lines = []
        for row in range(grid_rows):
            line_blocks = []
            for col in range(grid_cols):
                idx_in_batch = row * grid_cols + col
                if idx_in_batch < B:
                    line_blocks.append(rendered[idx_in_batch])
            for i in range(max_height):
                lines.append("".join(block[i] for block in line_blocks if i < len(block)))

        if legend:
            lines.append(f"[italic]slice = {idx}/{n_slices - 1}[/]")

        return "\n".join(lines)

    with Live(console=console, screen=False, refresh_per_second=1 / pause) as live:
        for idx in range(0, n_slices, stride):
            live.update(get_render(idx))
            time.sleep(pause)


def viz_batch_volume_manual(tensor, axis=0, colorscheme: ColorScheme = None, scale=1, legend=True, stride=1, **kwargs):
    """
    Interactive viewer for a batch of 3D tensors.

    Args:
        tensor (Tensor or ndarray): 4D tensor of shape [B, D, H, W].
        axis (int): Slice axis within each volume.
        colorscheme: Optional color scheme.
        scale (int): Visual scaling factor.
        stride (int): Step size when navigating.
    """
    np_array = tensor_to_numpy(tensor) # TODO tensor

    if np_array.ndim != 4:
        raise ValueError("Tensor must be 4D [B, D, H, W]")

    B = np_array.shape[0]
    volumes = [np.moveaxis(np_array[b], axis, 0) for b in range(B)]
    n_slices = volumes[0].shape[0]

    idx = 0
    console = Console()

    grid_rows = int(math.floor(math.sqrt(B)))
    grid_cols = int(math.ceil(B / grid_rows))

    def get_render(idx):
        rendered = []

        for b in range(B):
            slice_2d = volumes[b][idx]
            scaled = np.repeat(np.repeat(slice_2d, scale, axis=1), scale, axis=0)
            lines = _viz_2d(scaled, colorscheme, **kwargs)
            rendered.append(lines)

        # Pad all rendered blocks to same height
        max_height = max(len(r) for r in rendered)
        for r in rendered:
            while len(r) < max_height:
                r.append("")

        # Arrange in grid
        lines = []
        for row in range(grid_rows):
            line_blocks = []
            for col in range(grid_cols):
                idx_in_batch = row * grid_cols + col
                if idx_in_batch < B:
                    line_blocks.append(rendered[idx_in_batch])
            for i in range(max_height):
                lines.append("".join(block[i] for block in line_blocks if i < len(block)))

        if legend:
            lines.append(f"[italic]slice = {idx}/{n_slices - 1}[/]")
            lines.append("[dim]\u25C4 a |q quit| d \u25BA[/]")

        return "\n".join(lines)

    with Live(get_render(idx), console=console, screen=False, auto_refresh=False) as live:
        while True:
            key = readchar.readkey()
            new_idx = idx
            if key == "q":
                break
            elif key in ("d", readchar.key.RIGHT):
                new_idx = (idx + stride) % n_slices
            elif key in ("a", readchar.key.LEFT):
                new_idx = (idx - stride) % n_slices

            if new_idx != idx:
                idx = new_idx
                live.update(get_render(idx), refresh=True)
