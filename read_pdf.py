import fitz
import sys

sys.stdout.reconfigure(encoding='utf-8')

doc = fitz.open(r'd:\ULSC-main\Semantic_Importance-Aware_Communications_With_Semantic_Correction_Using_Large_Language_Models.pdf')
text = ''.join([doc[i].get_text() for i in range(min(20, len(doc)))])
print(text[:15000])
