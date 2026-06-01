"""
Extract words with coordinates from Algorithm 5 page (page 8, 0-indexed: 7).
Sort by y then x to reconstruct lines 20-22.
"""
import pdfplumber

with pdfplumber.open('docs/references/Malliavin-Mirafzali.pdf') as pdf:
    page = pdf.pages[7]
    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    # Print all words with their top coordinate
    prev_y = None
    for w in words:
        y = round(w['top'], 1)
        if y != prev_y:
            print(f'\n--- y={y} ---')
            prev_y = y
        print(f"  [{w['x0']:.1f},{w['x1']:.1f}] {w['text']!r}", end='')
