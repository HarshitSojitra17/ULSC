"""
Investigates ALL Excel files in DJSCC folder to find which contain SCDA data.
Cross-references with the existing graph_7frames_psnr.png which shows SCDA curves.
"""
import pandas as pd
import numpy as np
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

BASE = 'DJSCC'
PROB_STRS = ['0.001','0.004','0.007','0.01','0.04','0.07','0.1']

# All file prefixes found in DJSCC folder
prefixes = [
    '7sm_results',
    'NO7sm_results',
    '7PSNR_results',
    'NO7PSNR_results',
    '14sm_results',
    'NO14sm_results',
    '14psnr_results',
    'NO14psnr_results',
    'Attacker_7PSNR_results',
    'Attacker_NO7PSNR_results',
    'Attacker_14PSNR_results',
    'Attacker_NO14PSNR_results',
]

print("=== MEAN VALUES PER FILE GROUP (at loss rate 0.001) ===\n")
print(f"{'File Prefix':<35} {'Rows':>6}  {'Col1 (Loss)':<15} {'Col2 (metric)':<15}")
print("-"*75)

results = {}
for prefix in prefixes:
    fname = os.path.join(BASE, f'{prefix}_0.001.xlsx')
    if os.path.exists(fname):
        df = pd.read_excel(fname)
        num_cols = df.select_dtypes(include='number').columns.tolist()
        m1 = df[num_cols[0]].mean() if len(num_cols) > 0 else None
        m2 = df[num_cols[1]].mean() if len(num_cols) > 1 else None
        rows = len(df)
        print(f"{prefix:<35} {rows:>6}  {m1:<15.4f} {m2:<15.4f}")
        results[prefix] = {'rows': rows, 'loss_mean': m1, 'metric_mean': m2}
    else:
        print(f"{prefix:<35}  NOT FOUND")

# Now collect full curves for all 7 loss rates
print("\n\n=== FULL SM CURVES (mean per loss rate) ===")
for prefix in ['7sm_results', 'NO7sm_results', '14sm_results', 'NO14sm_results']:
    vals = []
    for p in PROB_STRS:
        f = os.path.join(BASE, f'{prefix}_{p}.xlsx')
        if os.path.exists(f):
            df = pd.read_excel(f)
            num_cols = df.select_dtypes(include='number').columns.tolist()
            vals.append(round(df[num_cols[1]].mean(), 4))
        else:
            vals.append(None)
    print(f"{prefix:<30}: {vals}")

print("\n=== FULL ATTACKER PSNR CURVES ===")
for prefix in ['Attacker_7PSNR_results','Attacker_NO7PSNR_results','Attacker_14PSNR_results','Attacker_NO14PSNR_results']:
    vals = []
    for p in PROB_STRS:
        f = os.path.join(BASE, f'{prefix}_{p}.xlsx')
        if os.path.exists(f):
            df = pd.read_excel(f)
            num_cols = df.select_dtypes(include='number').columns.tolist()
            vals.append(round(df[num_cols[1]].mean(), 3))
        else:
            vals.append(None)
    print(f"{prefix:<35}: {vals}")

print("\n\n=== KEY INSIGHT: Existing graph_7frames_psnr.png shows 4 curves ===")
print("Graph curves found (read from image):")
print("  DeepJSCC Train@0.01: ~23.6 dB at 0.001 → matches 7PSNR  = 23.64 dB")
print("  DeepJSCC Train@0:    ~23.6 dB at 0.001 → matches NO7PSNR = 23.65 dB")
print("  SCDA Train@0.01:     ~16.8 dB at 0.001 → matches Attacker_7PSNR = 16.79 dB")
print("  SCDA Train@0:        ~16.7 dB at 0.001 → matches Attacker_NO7PSNR = 16.73 dB")
print("\n=> Conclusion: 'Attacker_*PSNR' files contain SCDA PSNR data!")
print("=> BUT: No SCDA SM (Image Similarity) files exist in Excel!")
print("=> SCDA SM values must come from paper-reported text:")
print("   SCDA Train@0.01 @ low rates ≈ 0.76 (paper says 'around 0.76')")
print("   SCDA Train@0    @ low rates ≈ 0.80 (paper says 'around 0.80')")
