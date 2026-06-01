import pdfplumber

with pdfplumber.open('docs/references/Malliavin-Mirafzali.pdf') as pdf:
    total = len(pdf.pages)
    print(f'Total pages: {total}')
    for i, page in enumerate(pdf.pages):
        txt = page.extract_text() or ''
        if 'Algorithm 7' in txt or 'Algorithm7' in txt:
            print(f'--- Page {i+1} ---')
            print(txt)
            print()
