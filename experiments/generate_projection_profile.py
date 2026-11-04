#!/usr/bin/env python3
import os
import argparse
from PIL import Image, ImageDraw, ImageFont

def main():
    parser = argparse.ArgumentParser(description="Generate text vertical projection profile visualization.")
    parser.add_argument("--text", type=str, default="ແຕ່ເໜືອ", help="Lao text to render")
    parser.add_argument("--output", type=str, default="plots/lao_projection.png", help="Path to save the output image")
    parser.add_argument("--font_size", type=int, default=72, help="Font size for Lao text")
    parser.add_argument("--trans_size", type=int, default=24, help="Font size for translation text")
    args = parser.parse_args()

    # Create output dir if not exist
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 1. Load fonts
    # Find Lao font
    lao_font_path = "tesstrain_core/font/phetsarath_ot.ttf"
    if not os.path.exists(lao_font_path):
        lao_font_path = "../tesstrain_core/font/phetsarath_ot.ttf"
    if not os.path.exists(lao_font_path):
        lao_font_path = "font/phetsarath_ot.ttf"
    if not os.path.exists(lao_font_path):
        raise FileNotFoundError(f"Lao font not found at {lao_font_path}")

    # Find translation font (supporting Chinese/Unicode CJK on Mac/Linux/Windows)
    trans_font_path = None
    system_cjk_fonts = [
        "/System/Library/Fonts/PingFang.ttc",                # macOS
        "/System/Library/Fonts/STHeiti Light.ttc",          # macOS
        "/Library/Fonts/Arial Unicode.ttf",                 # macOS CJK
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",   # Ubuntu CJK
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", # Linux CJK
        "C:\\Windows\\Fonts\\msyh.ttc",                      # Windows YaHei
        "C:\\Windows\\Fonts\\simsun.ttc",                    # Windows SimSun
    ]
    for fp in system_cjk_fonts:
        if os.path.exists(fp):
            trans_font_path = fp
            break

    lao_font = ImageFont.truetype(lao_font_path, args.font_size)
    if trans_font_path:
        trans_font = ImageFont.truetype(trans_font_path, args.trans_size)
    else:
        # Fallback to default PIL font
        trans_font = ImageFont.load_default()

    # 2. Render Lao text to a temporary canvas to get tight bounding box
    temp_width = args.font_size * len(args.text) * 2
    temp_height = args.font_size * 3
    temp_img = Image.new("L", (temp_width, temp_height), color=255) # white background
    draw = ImageDraw.Draw(temp_img)
    
    # Draw text in black (0)
    text_x = args.font_size
    text_y = args.font_size
    draw.text((text_x, text_y), args.text, font=lao_font, fill=0)
    
    # Crop to bounding box of black pixels
    inverted_temp = Image.eval(temp_img, lambda x: 255 - x)
    bbox = inverted_temp.getbbox()
    if not bbox:
        raise ValueError("Rendered text image is empty.")
    
    # Add a bit of padding around the cropped text
    padding = 10
    left = max(0, bbox[0] - padding)
    top = max(0, bbox[1] - padding)
    right = min(temp_width, bbox[2] + padding)
    bottom = min(temp_height, bbox[3] + padding)
    
    cropped_text_img = temp_img.crop((left, top, right, bottom))
    w, h = cropped_text_img.size

    # 3. Calculate Vertical Projection Profile
    # Count black pixels (value < 128) in each column
    pixels = cropped_text_img.load()
    projection = []
    for x in range(w):
        black_count = 0
        for y in range(h):
            if pixels[x, y] < 128:
                black_count += 1
        projection.append(black_count)

    # 4. Create the final image
    # Height of projection area matches the text image height
    proj_h = h
    divider_h = 2
    spacing_h = 15
    label_h = args.trans_size + 15
    
    final_h = h + divider_h + proj_h + spacing_h + label_h
    final_img = Image.new("L", (w, final_h), color=255)
    final_draw = ImageDraw.Draw(final_img)

    # Paste original Lao text at the top
    final_img.paste(cropped_text_img, (0, 0))

    # Draw divider line (black horizontal line)
    divider_y = h
    final_draw.line([(0, divider_y), (w, divider_y)], fill=0, width=divider_h)

    # Draw vertical projection profile lines downwards
    start_y = divider_y + divider_h
    for x in range(w):
        length = projection[x]
        if length > 0:
            final_draw.line([(x, start_y), (x, start_y + length)], fill=0, width=1)

    # Draw centered translation text at the bottom
    try:
        t_bbox = final_draw.textbbox((0, 0), args.translation, font=trans_font)
        t_w = t_bbox[2] - t_bbox[0]
    except AttributeError:
        # Fallback for older Pillow versions
        t_w, _ = final_draw.textsize(args.translation, font=trans_font)

    label_y = start_y + proj_h + spacing_h
    label_x = (w - t_w) // 2
    final_draw.text((label_x, label_y), args.translation, font=trans_font, fill=0)

    # Save to output
    final_img.save(args.output)
    print(f"✓ Saved vertical projection profile visualization to: {args.output}")

if __name__ == "__main__":
    main()
