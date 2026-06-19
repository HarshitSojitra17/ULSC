import fitz
import sys
sys.stdout.reconfigure(encoding='utf-8')

doc = fitz.open(r'd:\ULSC-main\Semantic_Importance-Aware_Communications_With_Semantic_Correction_Using_Large_Language_Models.pdf')
all_text = ''.join([doc[i].get_text() for i in range(len(doc))])

# Get all text around DeepJSCC mentions
idx = 0
count = 0
while count < 10:
    idx = all_text.find('DeepJSCC', idx)
    if idx == -1:
        break
    print(f"\n===== Occurrence {count+1} at pos {idx} =====")
    print(all_text[max(0,idx-400):idx+800])
    print("---")
    idx += len('DeepJSCC')
    count += 1

# Also get references section
ref_idx = all_text.rfind('REFERENCES')
if ref_idx != -1:
    print("\n===== REFERENCES =====")
    print(all_text[ref_idx:ref_idx+3000])
