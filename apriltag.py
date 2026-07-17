import os
import cv2
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader


# ============================================================
# SETTINGS
# ============================================================

TAG_SIZE_CM = 5.0
GAP_CM = 1.0

COLUMNS = 2
ROWS = 5

TAG_IMAGE_PIXELS = 1000

PDF_GROUPS = [
    (1, 10, "AprilTags_ID_1_to_10_A4.pdf"),
    (11, 20, "AprilTags_ID_11_to_20_A4.pdf"),
]


# ============================================================
# SAVE LOCATION
# ============================================================

# Save the PDFs in the same folder as this Python script
OUTPUT_FOLDER = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# CHECK REQUIRED OPENCV MODULE
# ============================================================

if not hasattr(cv2, "aruco"):
    raise ImportError(
        "cv2.aruco is not available.\n"
        "Install the required packages with:\n"
        "pip install opencv-contrib-python reportlab pillow"
    )


# AprilTag 36h11 family
APRILTAG_DICTIONARY = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_APRILTAG_36h11
)


# ============================================================
# UNIT CONVERSION
# ============================================================

CM_TO_POINTS = 72.0 / 2.54

tag_size_pt = TAG_SIZE_CM * CM_TO_POINTS
gap_pt = GAP_CM * CM_TO_POINTS


# ============================================================
# A4 PAGE DIMENSIONS
# ============================================================

page_width, page_height = A4

grid_width = (
    COLUMNS * tag_size_pt
    + (COLUMNS - 1) * gap_pt
)

grid_height = (
    ROWS * tag_size_pt
    + (ROWS - 1) * gap_pt
)

if grid_width > page_width or grid_height > page_height:
    raise ValueError(
        "The selected AprilTag layout does not fit on A4 paper."
    )

# Center the tag grid on the page
left_margin = (page_width - grid_width) / 2
bottom_margin = (page_height - grid_height) / 2


# ============================================================
# GENERATE ONE APRILTAG
# ============================================================

def generate_tag_image(tag_id):
    """
    Generate one AprilTag image and return it as a NumPy array.
    """

    if hasattr(cv2.aruco, "generateImageMarker"):
        tag_image = cv2.aruco.generateImageMarker(
            APRILTAG_DICTIONARY,
            tag_id,
            TAG_IMAGE_PIXELS
        )

    else:
        # For older OpenCV versions
        tag_image = np.zeros(
            (TAG_IMAGE_PIXELS, TAG_IMAGE_PIXELS),
            dtype=np.uint8
        )

        cv2.aruco.drawMarker(
            APRILTAG_DICTIONARY,
            tag_id,
            TAG_IMAGE_PIXELS,
            tag_image,
            1
        )

    return tag_image


# ============================================================
# CREATE ONE PDF
# ============================================================

def create_pdf(first_id, last_id, filename):
    """
    Create one A4 PDF containing ten AprilTags.
    """

    output_path = os.path.join(OUTPUT_FOLDER, filename)

    pdf = canvas.Canvas(output_path, pagesize=A4)

    pdf.setTitle(
        f"AprilTags IDs {first_id} to {last_id}"
    )

    temporary_files = []

    try:
        for index, tag_id in enumerate(
            range(first_id, last_id + 1)
        ):
            row = index // COLUMNS
            column = index % COLUMNS

            # Generate AprilTag image
            tag_image = generate_tag_image(tag_id)

            # Temporarily save the tag as a PNG
            temporary_path = os.path.join(
                OUTPUT_FOLDER,
                f"_temporary_apriltag_{tag_id}.png"
            )

            Image.fromarray(tag_image).save(temporary_path)
            temporary_files.append(temporary_path)

            # Horizontal position
            x = left_margin + column * (
                tag_size_pt + gap_pt
            )

            # Vertical position
            # PDF coordinates start at the bottom-left corner
            y = (
                page_height
                - bottom_margin
                - tag_size_pt
                - row * (tag_size_pt + gap_pt)
            )

            # Draw the 5 cm × 5 cm AprilTag
            pdf.drawImage(
                ImageReader(temporary_path),
                x,
                y,
                width=tag_size_pt,
                height=tag_size_pt,
                preserveAspectRatio=True,
                mask="auto"
            )

            # Draw a dashed cutting box around the tag
            pdf.setLineWidth(0.3)
            pdf.setDash(2, 2)

            pdf.rect(
                x,
                y,
                tag_size_pt,
                tag_size_pt,
                stroke=1,
                fill=0
            )

            pdf.setDash()

        pdf.showPage()
        pdf.save()

    finally:
        # Remove temporary PNG files
        for temporary_file in temporary_files:
            if os.path.exists(temporary_file):
                os.remove(temporary_file)

    print(f"Saved PDF: {output_path}")


# ============================================================
# GENERATE BOTH PDF FILES
# ============================================================

for first_id, last_id, filename in PDF_GROUPS:
    create_pdf(first_id, last_id, filename)


print("\nFinished generating both AprilTag sheets.")
print(f"Files were saved in:\n{OUTPUT_FOLDER}")
print("\nPrint using 'Actual size' or '100%' scaling.")