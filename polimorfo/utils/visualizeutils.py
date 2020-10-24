import colorsys
import random
import matplotlib.colors as mplc
import numpy as np
from skimage.measure import find_contours
from matplotlib.patches import Polygon, Rectangle
import matplotlib
import matplotlib.pyplot as plt
from typing import List
from enum import Enum


def change_color_brightness(color, brightness_factor):
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
    modified_lightness = polygon_color[1] + (brightness_factor *
                                             polygon_color[1])
    modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
    modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
    modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness,
                                         polygon_color[2])
    return modified_color


def draw_text(ax,
              text,
              position,
              font_size,
              color="g",
              horizontal_alignment="center",
              rotation=0):
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
        bbox={
            "facecolor": "black",
            "alpha": 0.8,
            "pad": 0.7,
            "edgecolor": "none"
        },
        verticalalignment="top",
        horizontalalignment=horizontal_alignment,
        color=color,
        zorder=10,
        rotation=rotation,
    )


class BoxType(Enum):
    xyxy = 1
    xywh = 2


def draw_instances(img,
                   boxes,
                   labels,
                   scores,
                   masks,
                   class_name_dict,
                   title='',
                   figsize=(16, 8),
                   show_boxes=False,
                   show_masks=True,
                   min_score=0.5,
                   colors=None,
                   ax=None,
                   box_type: BoxType = BoxType.xyxy):
    labels_names = create_text_labels(labels, scores, class_name_dict)

    if colors is None:
        colors = generate_colormap(len(class_name_dict) + 1)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    width, height = img.size
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    out_image = np.array(img).astype(np.uint8)

    for idx in range(len(labels)):
        label_id = labels[idx]
        label_name = labels_names[idx]
        score = scores[idx]
        if score < min_score:
            continue
        if show_masks:
            mask = np.squeeze(masks[..., idx])
        color = colors[label_id]
        box = boxes[idx]

        if box_type == BoxType.xyxy:
            x0, y0, x1, y1 = box
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
        else:
            x, y, w, h = box

        if show_boxes:
            p = Rectangle((x, y),
                          w,
                          h,
                          linewidth=2,
                          alpha=0.7,
                          linestyle="dashed",
                          edgecolor=color,
                          facecolor='none')
            ax.add_patch(p)

        # add the caption
        # draw text in the center (defined by median) when box is not drawn
        # median is less sensitive to outliers.
        if show_masks:
            text_pos = np.median(mask.nonzero(), axis=1)[::-1]
        else:
            text_pos = (x + w / 2, y + h / 2)

        horiz_align = "left"

        lighter_color = change_color_brightness(color, brightness_factor=0.7)
        font_size = 10
        draw_text(ax, label_name, text_pos, font_size, lighter_color,
                  horiz_align)
        if show_masks:
            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2),
                                   dtype=np.float32)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts,
                            facecolor=color,
                            edgecolor=color,
                            fill=True,
                            alpha=.5)
                ax.add_patch(p)
    ax.imshow(out_image)
    return ax


def generate_colormap(nelems, scaled=False, bright=True):
    # Generate colors for drawing bounding boxes.
    brightness = 1. if bright else .7
    hsv_tuples = [(x / nelems, 1., brightness) for x in range(nelems)]
    colors = [colorsys.hsv_to_rgb(*x) for x in hsv_tuples]
    if scaled:
        colors = [
            (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)) for x in colors
        ]
    random.shuffle(colors)
    return colors


def create_text_labels(classes, scores, class_name_dict):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
    Returns:
        list[str] or None
    """
    labels = [class_name_dict[i] for i in classes]
    labels = [
        "{} {:.0f}%".format(label, score * 100)
        for label, score in zip(labels, scores)
    ]
    return labels


def draw_segmentation_map(mask: np.ndarray, colors: List[str]):
    """draw a segmentation map based on a class mask. 
    The mask have to contains number that represent the class

    Args:
        mask (np.ndarray): [description]
        colors (List[str], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    colors_rgb = np.array([matplotlib.colors.to_rgb(c) for c in colors]) * 255

    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)

    for c in sorted(np.unique(mask))[1:]:
        idx = mask == c
        r[idx] = colors_rgb[c, 0]
        g[idx] = colors_rgb[c, 1]
        b[idx] = colors_rgb[c, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def draw_segmentation(
        img,
        mask,
        idx_name_dict,
        colors=None,
        title='',
        ax=None,
        figsize=(16, 8),
):
    if colors is None:
        colors = generate_colormap(len(idx_name_dict) + 1)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    width, height = img.size
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    out_image = np.array(img).astype(np.uint8)

    for cat in np.unique(mask)[1:]:
        mask_cat = (mask == cat)
        cat_name = idx_name_dict[cat]
        color = colors[cat]

        # draw text in the center (defined by median) when box is not drawn
        # median is less sensitive to outliers.
        text_pos = np.median(mask_cat.nonzero(), axis=1)[::-1] - 20
        # horiz_align = "left"

        lighter_color = change_color_brightness(color, brightness_factor=0.7)
        font_size = 10
        draw_text(ax,
                  cat_name,
                  text_pos,
                  font_size,
                  horizontal_alignment='left')

        padded_mask = np.zeros((mask_cat.shape[0] + 2, mask_cat.shape[1] + 2),
                               dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask_cat
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            verts = np.fliplr(verts) - 1
            p = Polygon(
                verts,
                facecolor=color,
                edgecolor=lighter_color,    # 'black',
                fill=True,
                alpha=.5)
            ax.add_patch(p)
    ax.imshow(out_image)
    return ax