"""
Extract Algorithm 5 lines 20-22 with word-level coordinates.
"""
import pdfplumber

with pdfplumber.open('docs/references/Malliavin-Mirafzali.pdf') as pdf:
    # Algorithm 5 is on page 8 (0-indexed: 7)
    for page_idx in [7, 8]:
        page = pdf.pages[page_idx]
        print(f'\n=== PAGE {page_idx+1} full text ===')
        txt = page.extract_text()
        print(txt)
