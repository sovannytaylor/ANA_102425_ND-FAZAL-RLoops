import os
import math
import numpy as np
from loguru import logger
from aicsimageio import AICSImage
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageOps

logger.info("Import ok")

# ----------------------- CONFIG -----------------------
input_path = "raw_data"
output_root = "python_results/czi_png_output/"

CHANNEL_CONFIG = {
    0: {"name": "NUCLEOLUS",  "color": "#FF00BB"},
    1: {"name": "SPERMIDINE",       "color": "#31C731"},
    2: {"name": "NUCLEUS",   "color": "#6D776D"},
}

INCLUDE_CHANNELS_IN_MERGE = [0, 1,2]
DEFAULT_MICRONS_PER_PIXEL = 0.325
SCALEBAR_UM_FIXED = 10
scalebar_margin_px = 20
scalebar_height_px = 6

base_font_size = 24
font_path = None
label_position = "top_left"

MAKE_PANEL = True
PANEL_ORDER = ["NUCLEOLUS", "SPERMIDINE", "NUCLEUS"]
#this will also include the merged after 
PANEL_LAYOUT = "horizontal"
PANEL_SPACING = 15
PANEL_BG_COLOR = (10, 10, 10)

# ----------------------- HELPERS -----------------------

os.makedirs(output_root, exist_ok=True)

# ----------------------- HELPERS -----------------------
def get_xy_um_per_px(img: AICSImage):
    try:
        pps = getattr(img, "physical_pixel_sizes", None)
        if not pps:
            raise ValueError("No physical_pixel_sizes")
        x_um = getattr(pps, "X", None) or getattr(pps, "x", None)
        y_um = getattr(pps, "Y", None) or getattr(pps, "y", None)
        if x_um is None or y_um is None:
            raise ValueError("Missing dimensions")
        return float(x_um), float(y_um), "metadata"
    except Exception:
        return float(DEFAULT_MICRONS_PER_PIXEL), float(DEFAULT_MICRONS_PER_PIXEL), "fallback"


def normalize_channel(channel):
    ch = channel.astype(np.float32)
    ch -= ch.min()
    if ch.max() > 0:
        ch /= ch.max()
    return (ch * 255).astype(np.uint8)


def create_MIP(image_data):
    return np.max(image_data, axis=0) if image_data.ndim == 3 else image_data


def color_to_rgb(color_spec):
    try:
        return ImageColor.getrgb(color_spec)
    except Exception:
        return (255, 255, 255)


def pick_text_color_for_region(image: Image.Image, box):
    x0, y0, x1, y1 = [int(v) for v in box]
    gray = np.array(image.convert("L"))[y0:y1, x0:x1]
    mean_val = float(gray.mean()) if gray.size > 0 else 0
    return (0, 0, 0) if mean_val > 128 else (255, 255, 255)


def nice_scalebar_um_for_width(width_px, x_um_per_px, target_fraction=0.2):
    target_um = width_px * x_um_per_px * target_fraction
    exp = int(math.floor(math.log10(target_um)))
    candidates = [b * (10 ** e) for e in range(exp - 2, exp + 3) for b in (1, 2, 5)]
    return max(min(candidates, key=lambda u: abs(u - target_um)), 0.001)


def draw_label_and_scalebar(image: Image.Image, label_text: str, x_um_per_px: float):
    draw = ImageDraw.Draw(image)
    W, H = image.size

    font_size = int(min(W * 0.04, H * 0.04))
    font_size = max(min(font_size, 48), 18)
    try:
        font = ImageFont.truetype(font_path or "arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    tw, th = draw.textbbox((0, 0), label_text, font=font)[2:]
    while tw > W * 0.8 and font_size > 14:
        font_size -= 2
        try:
            font = ImageFont.truetype(font_path or "arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        tw, th = draw.textbbox((0, 0), label_text, font=font)[2:]

    pad = 10
    if label_position == "top_left":
        label_xy = (pad, pad)
        probe_box = (0, 0, min(200, W), min(60, H))
    else:
        label_xy = (pad, H - font_size - 2 * pad)
        probe_box = (0, max(0, H - 80), min(200, W), H)

    label_color = pick_text_color_for_region(image, probe_box)
    try:
        draw.text(
            label_xy, label_text,
            fill=label_color, font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0) if label_color == (255, 255, 255) else (255, 255, 255)
        )
    except TypeError:
        draw.text(label_xy, label_text, fill=label_color, font=font)

    bar_um = SCALEBAR_UM_FIXED or nice_scalebar_um_for_width(W, x_um_per_px, 0.2)
    px_per_um = 1.0 / x_um_per_px
    bar_len_px = int(round(bar_um * px_per_um))
    x1, y1 = W - scalebar_margin_px, H - scalebar_margin_px
    x0, y0 = x1 - bar_len_px, y1 - scalebar_height_px

    bar_color = pick_text_color_for_region(image, (x0, y0, x1, y1))
    draw.rectangle([x0, y0, x1, y1], fill=bar_color)

    label = f"{bar_um:g} µm"
    tw, th = draw.textbbox((0, 0), label, font=font)[2:]
    tx = x0 + (bar_len_px - tw) / 2
    ty = y0 - th - 5 if y0 - th - 5 > 0 else y1 + 5
    draw.text(
        (tx, ty), label,
        fill=(0, 0, 0) if bar_color == (255, 255, 255) else (255, 255, 255),
        font=font
    )
    return image


def colorize_channel(grayscale_arr, color_spec):
    ch = normalize_channel(grayscale_arr)
    rgb = np.array(color_to_rgb(color_spec)) / 255.0
    colorized = np.zeros((*ch.shape, 3), dtype=np.float32)
    for i in range(3):
        colorized[..., i] = ch / 255.0 * rgb[i]
    return (colorized * 255).astype(np.uint8)
# -------------------------------------------------------


def process_czi_file(image_name, input_folder, output_root):
    try:
        input_file = os.path.join(input_folder, f"{image_name}.czi")
        output_folder = os.path.join(output_root, image_name)
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Processing {image_name} ...")

        img = AICSImage(input_file)
        data = img.get_image_data("CZYX", S=0, T=0)
        np.save(os.path.join(output_folder, f"{image_name}.npy"), data)

        x_um_per_px, y_um_per_px, src = get_xy_um_per_px(img)
        logger.info(f"Pixel size source: {src} | X={x_um_per_px:.4f} µm/px")

        # ---- Save individual colorized channels ----
        mips = []
        for c in range(data.shape[0]):
            mip = create_MIP(data[c])
            mips.append(mip)

            ch_info = CHANNEL_CONFIG.get(c, {"name": f"T{c}", "color": "white"})
            ch_name, ch_color = ch_info["name"], ch_info["color"]

            colorized = colorize_channel(mip, ch_color)
            ch_img = Image.fromarray(colorized)
            ch_img = draw_label_and_scalebar(ch_img, f"{image_name} - {ch_name}", x_um_per_px)

            ch_out = os.path.join(output_folder, f"{image_name}_{ch_name}.png")
            os.makedirs(os.path.dirname(ch_out), exist_ok=True)
            ch_img.save(ch_out)
            logger.info(f"Saved {ch_out}")

        # ---- Merge selected channels ----
        include = INCLUDE_CHANNELS_IN_MERGE or list(range(len(mips)))
        include = [c for c in include if c < len(mips)]

        H, W = mips[0].shape
        merged = np.zeros((H, W, 3), dtype=np.float32)
        for c in include:
            ch_info = CHANNEL_CONFIG.get(c, {"name": f"T{c}", "color": "white"})
            rgb = color_to_rgb(ch_info["color"])
            norm = normalize_channel(mips[c]).astype(np.float32) / 255.0
            merged[..., 0] += norm * (rgb[0] / 255.0)
            merged[..., 1] += norm * (rgb[1] / 255.0)
            merged[..., 2] += norm * (rgb[2] / 255.0)

        merged = np.clip(merged, 0, 1)
        merged_img = Image.fromarray((merged * 255).astype(np.uint8))
        included_names = [CHANNEL_CONFIG.get(c, {"name": f"T{c}"})["name"] for c in include]
        merged_label = f"{image_name} - merged ({'+'.join(included_names)})"
        merged_img = draw_label_and_scalebar(merged_img, merged_label, x_um_per_px)
        merged_path = os.path.join(output_folder, f"{image_name}_merged.png")
        merged_img.save(merged_path)
        logger.info(f"Saved merged image with {included_names}")

        # ---- Create panel (optional) ----
        if MAKE_PANEL:
            panel_output = os.path.join(output_folder, f"{image_name}_panel.png")
            create_panel(
                image_folder=output_folder,
                output_path=panel_output,
                order=PANEL_ORDER + ["merged"],  # ensure merged is last
                layout=PANEL_LAYOUT,
                spacing=PANEL_SPACING,
                bg_color=PANEL_BG_COLOR,
                include_only=True,               # <-- key: don't append others
            )
    except Exception as e:
        logger.error(f"❌ Failed on {image_name}: {e}")
# -------------------------------------------------------

def create_panel(
    image_folder,
    output_path,
    order=None,
    layout="horizontal",
    spacing=10,
    bg_color=(0, 0, 0),
    include_only=False,   # NEW
):
    pngs = [f for f in os.listdir(image_folder) if f.lower().endswith(".png")]
    if not pngs:
        logger.warning(f"No PNGs found in {image_folder}")
        return

    if order:
        wanted = [name.lower() for name in order]
        sorted_pngs = []
        for name in wanted:
            # pick first match by substring; keep stable order; avoid dups
            for f in pngs:
                fl = f.lower()
                if name in fl and f not in sorted_pngs:
                    sorted_pngs.append(f)
                    break  # only one per name
        if not include_only:
            # append any leftover files if desired
            for f in pngs:
                if f not in sorted_pngs:
                    sorted_pngs.append(f)
    else:
        sorted_pngs = sorted(pngs)

    imgs = []
    for f in sorted_pngs:
        try:
            imgs.append(Image.open(os.path.join(image_folder, f)).convert("RGB"))
        except Exception as e:
            logger.error(f"Could not open {f}: {e}")

    if not imgs:
        return

    if layout == "horizontal":
        min_h = min(im.height for im in imgs)
        imgs = [ImageOps.contain(im, (im.width, min_h)) for im in imgs]
        total_w = sum(im.width for im in imgs) + spacing * (len(imgs) - 1)
        total_h = min_h
    else:
        min_w = min(im.width for im in imgs)
        imgs = [ImageOps.contain(im, (min_w, im.height)) for im in imgs]
        total_h = sum(im.height for im in imgs) + spacing * (len(imgs) - 1)
        total_w = min_w

    panel = Image.new("RGB", (total_w, total_h), bg_color)
    offset = 0
    for im in imgs:
        if layout == "horizontal":
            panel.paste(im, (offset, 0))
            offset += im.width + spacing
        else:
            panel.paste(im, (0, offset))
            offset += im.height + spacing

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    panel.save(output_path)
    logger.info(f"Saved panel → {output_path}")

# -------------------------------------------------------


def main():
    file_list = []
    for root, dirs, files in os.walk(input_path):
        for f in files:
            if f.lower().endswith(".czi"):
                full_path = os.path.join(root, f)
                name_only = os.path.splitext(os.path.basename(f))[0]
                file_list.append((name_only, full_path))

    if not file_list:
        logger.warning(f"No .czi files found in {input_path}")
        return

    logger.info(f"Found {len(file_list)} CZI files")

    for i, (name, full_path) in enumerate(file_list, start=1):
        logger.info(f"[{i}/{len(file_list)}] {name}")
        process_czi_file(name, os.path.dirname(full_path), output_root)

    logger.info("✅ All images processed successfully 🎉")


if __name__ == "__main__":
    main()
