"""
Reconstruct lines 20-22 by grouping words into y-coordinate bands.
"""
import pdfplumber

with pdfplumber.open('docs/references/Malliavin-Mirafzali.pdf') as pdf:
    page = pdf.pages[7]
    words = page.extract_words(x_tolerance=2, y_tolerance=2)

# Lines 20-22 are around y=340-420 based on previous run
# Group words into bands: use 8pt tolerance
bands = {}
for w in words:
    y = w['top']
    if not (340 <= y <= 430):
        continue
    # Find which band this word belongs to
    placed = False
    for by in list(bands.keys()):
        if abs(y - by) < 8:
            bands[by].append(w)
            placed = True
            break
    if not placed:
        bands[y] = [w]

# Sort bands by y, and within each band sort by x
for by in sorted(bands.keys()):
    line_words = sorted(bands[by], key=lambda w: w['x0'])
    line_text = ' '.join(w['text'] for w in line_words)
    print(f"y≈{by:.1f}: {line_text}")
