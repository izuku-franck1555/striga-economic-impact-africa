#!/usr/bin/env python3
"""
Concatenate v2 panel images into final 4-panel Figure 2.
Uses PIL direct concatenation (proven approach from v1).
"""

from PIL import Image
import os

# Paths to v2 panel images
REVISED_DIR = '/Users/francktonle/Downloads/DOCUMENTS/PAPER_DRAFTS/DR_HUGO_PAPER/NEW-ORGANIZATION/sn-article-template/REVISION-NC/04-FIGURES/REVISED'

img_ab_path = os.path.join(REVISED_DIR, 'figure_ssa_nature_quality_calibrated_v2.png')
img_cd_path = os.path.join(REVISED_DIR, 'figure_economic_priority_ssa_calibrated_v2.png')
output_path = os.path.join(REVISED_DIR, 'figure_2_revised.png')

print("=" * 60)
print("Creating 4-Panel Combined Figure v2")
print("=" * 60)

# Load v2 panel images
img_ab = Image.open(img_ab_path)
img_cd = Image.open(img_cd_path)

width_ab, height_ab = img_ab.size
width_cd, height_cd = img_cd.size
print(f"Panel a,b dimensions: {width_ab} x {height_ab}")
print(f"Panel c,d dimensions: {width_cd} x {height_cd}")

# Match widths
target_width = max(width_ab, width_cd)

if width_ab != target_width:
    scale = target_width / width_ab
    new_height_ab = int(height_ab * scale)
    img_ab = img_ab.resize((target_width, new_height_ab), Image.Resampling.LANCZOS)
    height_ab = new_height_ab

if width_cd != target_width:
    scale = target_width / width_cd
    new_height_cd = int(height_cd * scale)
    img_cd = img_cd.resize((target_width, new_height_cd), Image.Resampling.LANCZOS)
    height_cd = new_height_cd

# Stack vertically: a,b on top, c,d below
combined_height = height_ab + height_cd
combined = Image.new('RGB', (target_width, combined_height), 'white')
combined.paste(img_ab, (0, 0))
combined.paste(img_cd, (0, height_ab))

# Save
combined.save(output_path, dpi=(600, 600))
print(f"\n4-panel figure saved to: {output_path}")
print(f"Dimensions: {target_width} x {combined_height} pixels")
print(f"At 600 DPI: {target_width/600:.2f} x {combined_height/600:.2f} inches")

# Also save PDF version
output_pdf = os.path.join(REVISED_DIR, 'figure_2_revised.pdf')
combined.save(output_pdf, dpi=(600, 600))
print(f"PDF saved to: {output_pdf}")
