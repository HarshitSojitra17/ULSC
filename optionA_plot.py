"""
Option A: Plot Fig. 7 & Fig. 8 using EXISTING Excel data in DJSCC folder
1000 images per loss rate, fully computed by original authors.
NO SCDA, NO ULSC — only what we have.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

BASE = 'DJSCC'
LOSS_PROBS = [0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1]
PROB_STRS  = ['0.001','0.004','0.007','0.01','0.04','0.07','0.1']

# ── Helper: read mean value from one Excel file ────────────────────────────
def read_mean(prefix, p_str, metric_col):
    path = os.path.join(BASE, f'{prefix}_{p_str}.xlsx')
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return None
    df = pd.read_excel(path)
    # Find numeric column that is NOT the frame loss rate
    for col in df.columns:
        if col == metric_col or str(col).upper() == metric_col.upper():
            return float(df[col].mean())
    # fallback: find 2nd numeric column
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if len(num_cols) >= 2:
        return float(df[num_cols[1]].mean())
    return None

# ── Collect data for all curves ────────────────────────────────────────────
def collect_curve(prefix, metric_col):
    vals = []
    for p_str in PROB_STRS:
        v = read_mean(prefix, p_str, metric_col)
        vals.append(v)
    return vals

print("=== Collecting 7-frame Similarity (Image Similarity) data ===")
sm_7_NO   = collect_curve('NO7sm_results', 'SM')    # Train@0,   7 frames
sm_7_MI   = collect_curve('7sm_results', 'SM')      # Train@0.01, 7 frames
print(f"NO7sm  (Train@0):    {[f'{v:.4f}' for v in sm_7_NO]}")
print(f"7sm    (Train@0.01): {[f'{v:.4f}' for v in sm_7_MI]}")

print("\n=== Collecting 14-frame Similarity data ===")
sm_14_NO  = collect_curve('NO14sm_results', 'SM')   # Train@0,   14 frames
sm_14_MI  = collect_curve('14sm_results', 'SM')     # Train@0.01, 14 frames
print(f"NO14sm (Train@0):    {[f'{v:.4f}' for v in sm_14_NO]}")
print(f"14sm   (Train@0.01): {[f'{v:.4f}' for v in sm_14_MI]}")

print("\n=== Collecting Attacker-PSNR data (Fig. 8) ===")
att_7_MI  = collect_curve('Attacker_7PSNR_results', 'PSNR')    # Train@0.01, 7f
att_7_NO  = collect_curve('Attacker_NO7PSNR_results', 'PSNR')  # Train@0,    7f
att_14_MI = collect_curve('Attacker_14PSNR_results', 'PSNR')   # Train@0.01, 14f
att_14_NO = collect_curve('Attacker_NO14PSNR_results', 'PSNR') # Train@0,    14f
print(f"Att_7  (Train@0.01): {[f'{v:.2f}' for v in att_7_MI]}")
print(f"Att_7  (Train@0):    {[f'{v:.2f}' for v in att_7_NO]}")
print(f"Att_14 (Train@0.01): {[f'{v:.2f}' for v in att_14_MI]}")
print(f"Att_14 (Train@0):    {[f'{v:.2f}' for v in att_14_NO]}")

# Save aggregated data
agg = {
    'loss_probs': LOSS_PROBS,
    'sm_7_NO': sm_7_NO, 'sm_7_MI': sm_7_MI,
    'sm_14_NO': sm_14_NO, 'sm_14_MI': sm_14_MI,
    'att_7_MI': att_7_MI, 'att_7_NO': att_7_NO,
    'att_14_MI': att_14_MI, 'att_14_NO': att_14_NO,
}
with open('optionA_data.json', 'w') as f:
    json.dump(agg, f, indent=2)
print("\n  Saved: optionA_data.json")

# ── Styling ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'lines.linewidth': 2,
    'lines.markersize': 7, 'figure.dpi': 150,
})

# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — Image Similarity vs Frame Erase Rate
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('Fig. 7 — Image Similarity vs Frame Erase Rate (DeepJSCC only)', fontsize=14, fontweight='bold')

# (a) 7 frames
ax1.plot(LOSS_PROBS, sm_7_NO,  color='#1565C0', marker='s', linestyle='--', label='DeepJSCC 7f (Training at 0)', markerfacecolor='white', markeredgewidth=2)
ax1.plot(LOSS_PROBS, sm_7_MI,  color='#1565C0', marker='*', linestyle='-',  label='DeepJSCC 7f (Training at 0.01)')
ax1.set_xscale('log'); ax1.set_xlabel('Frame Erase Rate'); ax1.set_ylabel('Image Similarity')
ax1.set_title('(a) 7 Frames'); ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(); ax1.set_ylim([0.50, 0.90])

# (b) 14 frames
ax2.plot(LOSS_PROBS, sm_14_NO, color='#2E7D32', marker='s', linestyle='--', label='DeepJSCC 14f (Training at 0)', markerfacecolor='white', markeredgewidth=2)
ax2.plot(LOSS_PROBS, sm_14_MI, color='#2E7D32', marker='*', linestyle='-',  label='DeepJSCC 14f (Training at 0.01)')
ax2.set_xscale('log'); ax2.set_xlabel('Frame Erase Rate'); ax2.set_ylabel('Image Similarity')
ax2.set_title('(b) 14 Frames'); ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(); ax2.set_ylim([0.50, 0.90])

plt.tight_layout()
fig.savefig('optionA_fig7_similarity.png', dpi=200, bbox_inches='tight', facecolor='white')
print("  Saved: optionA_fig7_similarity.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — Attacker-PSNR vs Frame Erase Rate
# ─────────────────────────────────────────────────────────────────────────────
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5.5))
fig2.suptitle('Fig. 8 — Attacker-PSNR vs Frame Erase Rate (DeepJSCC only)', fontsize=14, fontweight='bold')

ax3.plot(LOSS_PROBS, att_7_NO,  color='#1565C0', marker='s', linestyle='--', label='DeepJSCC 7f (Training at 0)', markerfacecolor='white', markeredgewidth=2)
ax3.plot(LOSS_PROBS, att_7_MI,  color='#1565C0', marker='*', linestyle='-',  label='DeepJSCC 7f (Training at 0.01)')
ax3.set_xscale('log'); ax3.set_xlabel('Frame Erase Rate'); ax3.set_ylabel('Attacker-PSNR (dB)')
ax3.set_title('(a) 7 Frames'); ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend()

ax4.plot(LOSS_PROBS, att_14_NO, color='#2E7D32', marker='s', linestyle='--', label='DeepJSCC 14f (Training at 0)', markerfacecolor='white', markeredgewidth=2)
ax4.plot(LOSS_PROBS, att_14_MI, color='#2E7D32', marker='*', linestyle='-',  label='DeepJSCC 14f (Training at 0.01)')
ax4.set_xscale('log'); ax4.set_xlabel('Frame Erase Rate'); ax4.set_ylabel('Attacker-PSNR (dB)')
ax4.set_title('(b) 14 Frames'); ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend()

plt.tight_layout()
fig2.savefig('optionA_fig8_attacker_psnr.png', dpi=200, bbox_inches='tight', facecolor='white')
print("  Saved: optionA_fig8_attacker_psnr.png")

# ─────────────────────────────────────────────────────────────────────────────
# Print full summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("FULL DATA TABLE — Option A (existing 1000-image Excel data)")
print("="*75)
print(f"{'Loss Rate':<12} {'7f-Train@0':>12} {'7f-Train@.01':>13} {'14f-Train@0':>13} {'14f-Train@.01':>14}")
print("-"*65)
for i, p in enumerate(LOSS_PROBS):
    print(f"{p:<12} {sm_7_NO[i]:>12.4f} {sm_7_MI[i]:>13.4f} {sm_14_NO[i]:>13.4f} {sm_14_MI[i]:>14.4f}")

print(f"\n{'Loss Rate':<12} {'Att7-T@0':>10} {'Att7-T@.01':>12} {'Att14-T@0':>11} {'Att14-T@.01':>13}")
print("-"*60)
for i, p in enumerate(LOSS_PROBS):
    print(f"{p:<12} {att_7_NO[i]:>10.2f} {att_7_MI[i]:>12.2f} {att_14_NO[i]:>11.2f} {att_14_MI[i]:>13.2f}")
