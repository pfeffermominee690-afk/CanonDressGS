from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
PUBLICATION = FIGURES / "publication"
SOURCE = PUBLICATION / "source" / "pure_endpoint"

BLUE = "#2F73BF"
GREEN = "#2E9F74"
AMBER = "#D99D28"
RED = "#C94B3C"
DARK = "#18232D"
MID = "#667788"
LIGHT = "#E7EDF2"
PALE_BLUE = "#EEF5FB"
PALE_GREEN = "#EDF8F3"
PALE_AMBER = "#FFF7E8"
WHITE = "#FFFFFF"


def font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size=size)


def text_center(draw, xy, text, face, fill=DARK):
    x, y = xy
    box = draw.textbbox((0, 0), text, font=face)
    draw.text((x - (box[2] - box[0]) / 2, y), text, font=face, fill=fill)


def multiline_center(draw, box, text, face, fill=DARK, spacing=8):
    x0, y0, x1, y1 = box
    bbox = draw.multiline_textbbox((0, 0), text, font=face, spacing=spacing, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.multiline_text(
        ((x0 + x1 - tw) / 2, (y0 + y1 - th) / 2),
        text,
        font=face,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def panel_tag(draw, x, y, label):
    draw.rectangle((x, y, x + 62, y + 62), fill=DARK)
    text_center(draw, (x + 31, y + 7), label, font(42, bold=True), WHITE)


def arrow(draw, start, end, fill=GREEN, width=12, head=24):
    x0, y0 = start
    x1, y1 = end
    draw.line((x0, y0, x1 - head, y1), fill=fill, width=width)
    draw.polygon(((x1, y1), (x1 - head, y1 - head), (x1 - head, y1 + head)), fill=fill)


def paste_contained(canvas_img, image_path, box, border=LIGHT, border_width=4):
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    with Image.open(image_path) as src:
        src = src.convert("RGB")
        src.thumbnail((width - 12, height - 12), Image.Resampling.LANCZOS)
        px = x0 + (width - src.width) // 2
        py = y0 + (height - src.height) // 2
        canvas_img.paste(src, (px, py))
    draw = ImageDraw.Draw(canvas_img)
    draw.rectangle(box, outline=border, width=border_width)


def save_png_pdf(image, stem, dpi=300):
    png_path = FIGURES / f"{stem}.png"
    pdf_path = FIGURES / f"{stem}.pdf"
    image.save(png_path, dpi=(dpi, dpi), optimize=True)
    page_w = image.width / dpi * 72.0
    page_h = image.height / dpi * 72.0
    pdf = canvas.Canvas(str(pdf_path), pagesize=(page_w, page_h))
    pdf.drawImage(ImageReader(image), 0, 0, width=page_w, height=page_h)
    pdf.showPage()
    pdf.save()


def build_figure1():
    w, h = 5000, 1220
    img = Image.new("RGB", (w, h), WHITE)
    draw = ImageDraw.Draw(img)
    panel_tag(draw, 45, 35, "A")
    draw.text((130, 38), "REFERENCE-CONTROLLED ENDPOINT REALIZATION", font=font(58, True), fill=DARK)
    draw.text(
        (130, 108),
        "Five seen garments, one frozen animatable identity, matched target pose and camera",
        font=font(34),
        fill=MID,
    )

    garments = ["O01", "O02", "O03", "O04", "O08"]
    names = ["hoodie / trousers", "shirt / slacks", "suit", "jacket / jeans", "sweater / trousers"]
    margin = 75
    pair_gap = 30
    pair_w = (w - 2 * margin - 4 * pair_gap) // 5
    box_w = 405
    box_h = 730
    y0 = 285
    for idx, (garment, name) in enumerate(zip(garments, names)):
        gx = margin + idx * (pair_w + pair_gap)
        center = gx + pair_w // 2
        text_center(draw, (center, 170), f"{garment}  {name}", font(34, True), DARK)

        ref_x0 = gx + 20
        out_x0 = gx + pair_w - box_w - 20
        ref_box = (ref_x0, y0, ref_x0 + box_w, y0 + box_h)
        out_box = (out_x0, y0, out_x0 + box_w, y0 + box_h)
        text_center(draw, (ref_x0 + box_w / 2, 225), "Reference", font(29, True), MID)
        text_center(draw, (out_x0 + box_w / 2, 225), "CanonDressGS", font(29, True), GREEN)

        ref_path = SOURCE / "references" / garment / "cond_000000.png"
        out_path = SOURCE / "renders" / f"{garment}_snapped.png"
        paste_contained(img, ref_path, ref_box, border=LIGHT, border_width=4)
        paste_contained(img, out_path, out_box, border=GREEN, border_width=8)
        arrow(draw, (ref_x0 + box_w + 16, y0 + box_h // 2), (out_x0 - 14, y0 + box_h // 2), GREEN, 10, 20)

    draw.line((75, 1060, w - 75, 1060), fill=LIGHT, width=4)
    metrics = [
        ("60 / 60", "clean endpoint selections"),
        ("1.000", "exact endpoint match"),
        ("0", "identity contamination"),
    ]
    centers = [1050, 2500, 3950]
    for center, (value, label) in zip(centers, metrics):
        text_center(draw, (center, 1080), value, font(44, True), GREEN)
        text_center(draw, (center, 1132), label, font(28), MID)
    save_png_pdf(img, "figure1_canondressgs_teaser")


def build_figure7():
    w, h = 5000, 1080
    img = Image.new("RGB", (w, h), WHITE)
    draw = ImageDraw.Draw(img)
    panel_tag(draw, 45, 35, "A")
    draw.text((130, 38), "SUBJECT00 SECOND-IDENTITY REPLICATION PROTOCOL", font=font(56, True), fill=DARK)
    draw.text(
        (130, 108),
        "Same construction, representation, controller architecture, and safety audit",
        font=font(34),
        fill=MID,
    )

    stages = [
        ("Frozen subject00\nbase avatar", PALE_BLUE, BLUE),
        ("Teacher Endpoints\nO01 / O03 / O04", PALE_BLUE, BLUE),
        ("Rank <= 2 endpoint\ncoordinate system", PALE_BLUE, BLUE),
        ("Same 3,076-parameter\nreference controller", PALE_GREEN, GREEN),
        ("Endpoint realization +\nidentity-safety audit", PALE_GREEN, GREEN),
    ]
    x = 95
    y0 = 245
    box_w = 820
    box_h = 250
    gap = 155
    for idx, (label, fill, stroke) in enumerate(stages):
        draw.rectangle((x, y0, x + box_w, y0 + box_h), fill=fill, outline=stroke, width=8)
        multiline_center(draw, (x + 20, y0 + 20, x + box_w - 20, y0 + box_h - 20), label, font(34, True), DARK)
        if idx < len(stages) - 1:
            arrow(draw, (x + box_w + 26, y0 + box_h // 2), (x + box_w + gap - 22, y0 + box_h // 2), GREEN, 11, 23)
        x += box_w + gap

    panel_tag(draw, 45, 605, "B")
    draw.text((130, 612), "SEALED RESULT SLOTS", font=font(46, True), fill=DARK)
    draw.text((860, 620), "Intentionally blank until the formal subject00 experiment is complete", font=font(31), fill=RED)

    labels = ["Clean top-1", "Macro precision", "Macro recall", "Exact endpoint", "Identity contamination", "LPIPS"]
    left = 110
    top = 740
    slot_w = 755
    slot_h = 190
    gap = 60
    for i, label in enumerate(labels):
        sx = left + i * (slot_w + gap)
        draw.rectangle((sx, top, sx + slot_w, top + slot_h), fill="#FAFBFC", outline=LIGHT, width=5)
        text_center(draw, (sx + slot_w / 2, top + 34), label, font(28, True), MID)
        text_center(draw, (sx + slot_w / 2, top + 88), "--", font(54, True), DARK)
        text_center(draw, (sx + slot_w / 2, top + 150), "pending sealed result", font(24), RED)

    draw.text(
        (110, 1002),
        "No subject00 quantitative or qualitative result is asserted by this draft panel.",
        font=font(27, True),
        fill=RED,
    )
    save_png_pdf(img, "figure7_subject00_replication")


def copy_existing_figures():
    mapping = {
        "figure2_method": "figure2_canondressgs_pipeline",
        "figure3_geometry_causal": "figure3_geometry_causal_analysis",
        "figure4_dual_support": "figure4_dual_support_all_pair",
        "figure5_hard_lookup_relation": "figure5_hard_lookup_relation",
        "figure6_endpoint_limits": "figure6_scope_analysis",
    }
    for source_stem, target_stem in mapping.items():
        for suffix in (".png", ".pdf"):
            source = PUBLICATION / f"{source_stem}{suffix}"
            target = FIGURES / f"{target_stem}{suffix}"
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)


def build_preview():
    stems = [
        "figure1_canondressgs_teaser",
        "figure2_canondressgs_pipeline",
        "figure3_geometry_causal_analysis",
        "figure4_dual_support_all_pair",
        "figure5_hard_lookup_relation",
        "figure6_scope_analysis",
        "figure7_subject00_replication",
    ]
    target_w = 2250
    margin = 70
    label_h = 70
    rows = []
    for i, stem in enumerate(stems, start=1):
        with Image.open(FIGURES / f"{stem}.png") as source:
            source = source.convert("RGB")
            scale = target_w / source.width
            resized = source.resize((target_w, int(source.height * scale)), Image.Resampling.LANCZOS)
        row = Image.new("RGB", (target_w + 2 * margin, resized.height + label_h + margin), WHITE)
        row_draw = ImageDraw.Draw(row)
        row_draw.text((margin, 16), f"Figure {i}", font=font(34, True), fill=DARK)
        row.paste(resized, (margin, label_h))
        row_draw.line((margin, row.height - 2, row.width - margin, row.height - 2), fill=LIGHT, width=3)
        rows.append(row)
    preview = Image.new("RGB", (target_w + 2 * margin, sum(row.height for row in rows)), WHITE)
    y = 0
    for row in rows:
        preview.paste(row, (0, y))
        y += row.height
    preview.save(FIGURES / "current_v2_all_seven_figures_preview.png", dpi=(180, 180), optimize=True)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    build_figure1()
    copy_existing_figures()
    build_figure7()
    build_preview()
    print("Generated the seven current-v2 figure slots in", FIGURES)


if __name__ == "__main__":
    main()
