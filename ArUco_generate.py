import cv2
import numpy as np
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from PIL import Image


# ==============================
# Settings
# ==============================
marker_ids = list(range(10))   # ID 0 to 9
marker_size_cm = 4.0           # each tag is 4 cm x 4 cm
gap_cm = 0.7                   # white gap between tags
show_id_text = True            # show ID under each tag

# Save in the same folder as this Python file
current_folder = Path(__file__).parent
output_pdf = current_folder / "aruco_tags_0_to_9_one_page.pdf"


# ==============================
# ArUco dictionary
# ==============================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


# ==============================
# PDF page setup
# ==============================
page_width, page_height = landscape(A4)

c = canvas.Canvas(str(output_pdf), pagesize=(page_width, page_height))

rows = 2
cols = 5

marker_size_pt = marker_size_cm * cm
gap_pt = gap_cm * cm

grid_width = cols * marker_size_pt + (cols - 1) * gap_pt
grid_height = rows * marker_size_pt + (rows - 1) * gap_pt

start_x = (page_width - grid_width) / 2
start_y = (page_height - grid_height) / 2


# ==============================
# Generate and draw 10 tags
# ==============================
for idx, marker_id in enumerate(marker_ids):
    row = idx // cols
    col = idx % cols

    # Generate marker as numpy image
    marker_img = cv2.aruco.generateImageMarker(
        aruco_dict,
        marker_id,
        800
    )

    # Convert numpy image to PIL image, no need to save PNG
    pil_img = Image.fromarray(marker_img)

    # Convert PIL image to ReportLab image object
    img_reader = ImageReader(pil_img)

    x = start_x + col * (marker_size_pt + gap_pt)
    y = start_y + (rows - 1 - row) * (marker_size_pt + gap_pt)

    c.drawImage(
        img_reader,
        x,
        y,
        width=marker_size_pt,
        height=marker_size_pt
    )

    if show_id_text:
        c.setFont("Helvetica", 10)
        c.drawCentredString(
            x + marker_size_pt / 2,
            y - 12,
            f"ID {marker_id}"
        )


# ==============================
# Save PDF
# ==============================
c.save()

print(f"Saved PDF to: {output_pdf}")
print("Generated one PDF page with 10 ArUco tags, IDs 0 to 9.")
print("Each tag is 4 cm x 4 cm.")