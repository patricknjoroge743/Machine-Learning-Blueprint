from PIL import Image, ImageDraw, ImageFont
import textwrap


def text_to_image(
    text,
    output_path="output.png",
    font_size=14,
    bg_color="#1e1e1e",
    text_color="#00ff00",
    padding=20,
):
    """
    Convert ASCII text block to image.

    Args:
        text: The ASCII art text
        output_path: Where to save the image
        font_size: Font size in pixels
        bg_color: Background color (hex)
        text_color: Text color (hex)
        padding: Padding around text in pixels
    """

    # Use monospace font for proper alignment
    try:
        # Try to use a common monospace font
        font = ImageFont.truetype("Courier New.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
        except:
            # Fallback to default font
            font = ImageFont.load_default()

    # Split text into lines
    lines = text.split("\n")

    # Calculate image dimensions
    # Get max line length
    max_width = max(len(line) for line in lines)

    # Create dummy image to measure text
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)

    # Measure single character to calculate dimensions
    char_bbox = dummy_draw.textbbox((0, 0), "M", font=font)
    char_width = char_bbox[2] - char_bbox[0]
    char_height = char_bbox[3] - char_bbox[1]

    # Calculate final image size
    img_width = (max_width * char_width) + (2 * padding)
    img_height = (len(lines) * char_height) + (2 * padding)

    # Create actual image
    img = Image.new("RGB", (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw each line
    y_position = padding
    for line in lines:
        draw.text((padding, y_position), line, fill=text_color, font=font)
        y_position += char_height

    # Save image
    img.save(output_path)
    print(f"Image saved to: {output_path}")
    return output_path


# # Your ASCII block
# ascii_block = """┌─────────────────────────────────────────────────────────────────┐
# │                    WITH AFML CACHING                            │
# ├─────────────────────────────────────────────────────────────────┤
# │                                                                 │
# │  Load Ticks ────────> [CACHED] 0.5s [OK]                        │
# │       │                                                         │
# │       ▼                                                         │
# │  Volume Bars ───────> [CACHED] 0.3s [OK]                        │
# │       │                                                         │
# │       ▼                                                         │
# │  Feature Eng ───────> [CACHED] 0.8s [OK]                        │
# │       │                                                         │
# │       ▼                                                         │
# │  Labeling ──────────> [CACHED] 0.5s [OK]                        │
# │       │                                                         │
# │       ▼                                                         │
# │  Training ──────────> 60s (still computed)                      │
# │       │                                                         │
# │       ▼                                                         │
# │  Backtest ──────────> 45s (still computed)                      │
# │                                                                 │
# │  TOTAL: 107s (first run: 390s) × 50 params = 1.5 hours          │
# │  SPEEDUP: 3.6x overall, 200x on cached steps                    │
# └─────────────────────────────────────────────────────────────────┘"""

# # Generate image
# text_to_image(
#     ascii_block,
#     output_path='afml_caching_diagram.png',
#     font_size=16,
#     bg_color='#0d1117',  # GitHub dark background
#     text_color='#58a6ff',  # GitHub blue
#     padding=30
# )
