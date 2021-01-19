import colorsys
import random
from enum import Enum
from typing import Dict, List, Tuple, Union

import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from PIL import Image
from scipy import special
from skimage import measure


def change_color_brightness(color: Tuple, brightness_factor: float):
    """
    Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
    less or more saturation than the original color.
    Args:
        color: color of the polygon. Refer to `matplotlib.colors` for a full list of
            formats that are accepted.
        brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
            0 will correspond to no change, a factor in [-1.0, 0) range will result in
            a darker color and a factor in (0, 1.0] range will result in a lighter color.
    Returns:
        modified_color (tuple[double]): a tuple containing the RGB values of the
            modified color. Each value in the tuple is in the [0.0, 1.0] range.
    """
    assert brightness_factor >= -1.0 and brightness_factor <= 1.0
    color = mplc.to_rgb(color)
    polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
    modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
    modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
    modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
    modified_color = colorsys.hls_to_rgb(
        polygon_color[0], modified_lightness, polygon_color[2]
    )
    return modified_color


def draw_text(
    ax: plt.Axes,
    text: str,
    position: Tuple,
    font_size: float,
    color: str = "g",
    horizontal_alignment: str = "center",
    rotation: int = 0,
):
    """
    Args:
        text (str): class label
        position (tuple): a tuple of the x and y coordinates to place text on image.
        font_size (int, optional): font of the text. If not provided, a font size
            proportional to the image width is calculated and used.
        color: color of the text. Refer to `matplotlib.colors` for full list
            of formats that are accepted.
        horizontal_alignment (str): see `matplotlib.text.Text`
        rotation: rotation angle in degrees CCW
    Returns:
        output (VisImage): image object with text drawn.
    """
    # since the text background is dark, we don't want the text to be dark
    color = np.maximum(list(mplc.to_rgb(color)), 0.2)
    color[np.argmax(color)] = max(0.8, np.max(color))

    x, y = position
    ax.text(
        x,
        y,
        text,
        size=font_size * 1,
        family="sans-serif",
        bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
        verticalalignment="top",
        horizontalalignment=horizontal_alignment,
        color=color,
        zorder=10,
        rotation=rotation,
    )


class BoxType(Enum):
    xyxy = 1
    xywh = 2


def draw_instances(
    img: Union[Image.Image, np.ndarray],
    boxes: Union[np.ndarray, List],
    labels: Union[np.ndarray, List],
    scores: Union[np.ndarray, List],
    masks: Union[np.ndarray, List],
    idx_class_dict: Dict[int, str],
    title: str = "",
    figsize: Tuple = (16, 8),
    show_boxes: bool = False,
    show_masks: bool = True,
    min_score: float = 0.5,
    min_area: int = 0,
    colors: List = None,
    ax: plt.Axes = None,
    box_type: BoxType = BoxType.xyxy,
    only_class_idxs: List[int] = None,
):
    """draw the instances from a object detector or an instance segmentation model

    Args:
        img (np.ndarray): an image with shape (width, height, channels)
        boxes (np.ndarray): an array of shape (nboxes, 4)
        labels (np.ndarray): an array of shape (nlabels,)
        scores (np.ndarray): an array of shape (nscores,)
        masks (np.ndarray): an array of shape [nmasks, 1, width, height ]
        idx_class_dict (Dict[int, str]): a dictionary that maps class id to class name
        title (str, optional): [description]. Defaults to ''.
        figsize (Tuple, optional): [description]. Defaults to (16, 8).
        show_boxes (bool, optional): [description]. Defaults to False.
        show_masks (bool, optional): [description]. Defaults to True.
        min_score (float, optional): [description]. Defaults to 0.5.
        colors (List, optional): [description]. Defaults to None.
        ax (plt.Axes, optional): [description]. Defaults to None.
        box_type (BoxType, optional): [description]. Defaults to BoxType.xyxy.
        only_class_idxs (List[int], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    if boxes is not None and len(boxes) > 0 and len(np.array(boxes).shape) != 2:
        raise ValueError(
            f"the shape of the boxes should be (N_BOXES, 4) while shape is {np.array(boxes).shape}"
        )

    if masks is not None and len(masks) > 0 and len(np.array(masks).shape) < 3:
        raise ValueError(
            f"the shape of the masks should be (N_MASKS, HEIGHT, WIDTH) while shape is {np.array(masks).shape}"
        )

    labels_names = create_text_labels(labels, scores, idx_class_dict)

    colors = generate_colormap(len(idx_class_dict) + 1) if colors is None else colors

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    only_class_idxs = (
        list(idx_class_dict.keys()) if only_class_idxs is None else only_class_idxs
    )

    if isinstance(img, Image.Image):
        width, height = img.width, img.height
    else:
        height, width = img.shape[:2]

    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis("off")
    ax.set_title(title)

    out_image = np.array(img).astype(np.uint8)

    for idx in range(len(labels)):
        label_id = labels[idx]

        if label_id not in only_class_idxs:
            continue

        label_name = labels_names[idx]
        score = scores[idx]
        if score < min_score:
            continue

        if show_masks:
            mask = np.squeeze(masks[idx, ...])
        color = colors[label_id]

        if show_boxes:
            box = boxes[idx]
            if box_type.value == BoxType.xyxy.value:
                x0, y0, x1, y1 = box
                x, y, w, h = x0, y0, x1 - x0, y1 - y0
            else:
                x, y, w, h = box

            area = w * h
            if area < min_area:
                continue

            p = Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                alpha=0.7,
                linestyle="dashed",
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(p)

        # add the caption
        # draw text in the center (defined by median) when box is not drawn
        # median is less sensitive to outliers.
        if show_masks:
            text_pos = np.median(mask.nonzero(), axis=1)[::-1]
        else:
            text_pos = (x + 5, y + 5)

        horiz_align = "left"

        lighter_color = change_color_brightness(color, brightness_factor=0.7)
        font_size = 10
        draw_text(ax, label_name, text_pos, font_size, lighter_color, horiz_align)
        if show_masks:
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.float32
            )
            padded_mask[1:-1, 1:-1] = mask
            contours = measure.find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(
                    verts, facecolor=color, edgecolor=color, fill=True, alpha=0.5
                )
                ax.add_patch(p)
    ax.imshow(out_image)
    return ax


def generate_colormap(nelems: int, scaled: bool = False, bright: bool = True):
    # Generate colors for drawing bounding boxes.
    brightness = 1.0 if bright else 0.7
    hsv_tuples = [(x / nelems, 1.0, brightness) for x in range(nelems)]
    colors = [colorsys.hsv_to_rgb(*x) for x in hsv_tuples]
    if scaled:
        colors = [(int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)) for x in colors]
    random.shuffle(colors)
    return colors


def create_text_labels(
    classes: List[int], scores: List[float], idx_class_dict: Dict[int, str]
):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        idx_class_dict (Dict[int, str] or None):
    Returns:
        list[str]
    """
    labels = [idx_class_dict[i] for i in classes]
    labels = [
        "{} {:.0f}%".format(label, score * 100) for label, score in zip(labels, scores)
    ]
    return labels


# def draw_segmentation_map(mask: np.ndarray, colors: List[str]):
#     """draw a segmentation map based on a class mask.
#     The mask have to contains number that represent the class

#     Args:
#         mask (np.ndarray): [description]
#         colors (List[str], optional): [description]. Defaults to None.

#     Returns:
#         [type]: [description]
#     """
#     colors_rgb = np.array([matplotlib.colors.to_rgb(c) for c in colors]) * 255

#     r = np.zeros_like(mask).astype(np.uint8)
#     g = np.zeros_like(mask).astype(np.uint8)
#     b = np.zeros_like(mask).astype(np.uint8)

#     for c in sorted(np.unique(mask))[1:]:
#         idx = mask == c
#         r[idx] = colors_rgb[c, 0]
#         g[idx] = colors_rgb[c, 1]
#         b[idx] = colors_rgb[c, 2]

#     rgb = np.stack([r, g, b], axis=2)
#     return rgb


def draw_segmentation(
    img: Union[np.ndarray, Image.Image],
    logits_or_mask: np.ndarray,
    idx_name_dict: Dict[int, str],
    min_conf: float,
    colors: List = None,
    title: str = "",
    ax: plt.Axes = None,
    figsize: Tuple[int, int] = (16, 8),
):
    """draw the result from a segmentation model

    Args:
        img (Union[np.ndarray, Image.Image]): an PIL image or a numpy array
        logits_or_mask (np.ndarray): it accepts:
            - the logits coming from the model with shape (n_classes, H, W), or
            - the mask coming from true annotations with shape (H,W) and containing pixel classification
        idx_name_dict (Dict[int, str]):
        min_conf (float): the min confidence of the mask given as output
        colors (List, optional): the colors to diplay categories. Defaults to None.
        title (str, optional): [description]. Defaults to ''.
        ax (plt.Axes, optional): [description]. Defaults to None.
        figsize (Tuple[int, int], optional): [description]. Defaults to (16, 8).

    Returns:
        [plt.Axes]: the ax of the given plot
    """
    if colors is None:
        colors = generate_colormap(len(idx_name_dict) + 1)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    width, height = img.size
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis("off")
    ax.set_title(title)

    out_image = np.array(img).astype(np.uint8)

    if len(logits_or_mask.shape) == 2:
        masks = logits_or_mask
        probs = np.ones((len(idx_name_dict), *masks.shape))
    else:
        probs = special.softmax(logits_or_mask, axis=0)
        masks = probs.argmax(0)

    for cat in np.unique(masks)[1:]:
        mask = masks == cat

        conf = np.round(np.nan_to_num(probs[cat, mask].mean()), 2)
        if conf < min_conf:
            continue
        # remove all the pixels with score lower
        filt_mask = np.copy(mask)
        filt_mask[probs[cat, ...] < min_conf] = 0

        name = f"{idx_name_dict[cat]} {int(conf * 100)}%"
        color = colors[cat]

        # draw text in the center (defined by median) when box is not drawn
        # median is less sensitive to outliers.
        text_pos = np.median(filt_mask.nonzero(), axis=1)[::-1] - 20
        # horiz_align = "left"

        lighter_color = change_color_brightness(color, brightness_factor=0.7)
        font_size = 10
        draw_text(ax, name, text_pos, font_size, horizontal_alignment="left")

        padded_mask = np.zeros(
            (filt_mask.shape[0] + 2, filt_mask.shape[1] + 2), dtype=np.uint8
        )
        padded_mask[1:-1, 1:-1] = filt_mask
        contours = measure.find_contours(padded_mask, 0.5)
        for verts in contours:
            verts = np.fliplr(verts) - 1
            p = Polygon(
                verts,
                facecolor=color,
                edgecolor=lighter_color,  # 'black',
                fill=True,
                alpha=0.5,
            )
            ax.add_patch(p)
    ax.imshow(out_image)
    return ax
