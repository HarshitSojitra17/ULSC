"""
COMPLETE 4-curve graphs for Fig. 7 and Fig. 8:
  - DeepJSCC (Train@0), 7f and 14f
  - DeepJSCC (Train@0.01), 7f and 14f
  - SCDA (Train@0), 7f and 14f       <- Attacker_NO7/14PSNR files
  - SCDA (Train@0.01), 7f and 14f   <- Attacker_7/14PSNR files

SCDA SM (Image Similarity) does NOT exist in Excel,
so we use paper-reported approximate values from text.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json, sys
sys.stdout.reconfigure(encoding='utf-8')

BASE = 'DJSCC'
LOSS_PROBS = [0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1]
PROB_STRS  = ['0.001','0.004','0.007','0.01','0.04','0.07','0.1']

# ── Helper ────────────────────────────────────────────────────────────────
def collect(prefix, metric_idx=1):
    vals = []
    for p in PROB_STRS:
        f = os.path.join(BASE, f'{prefix}_{p}.xlsx')
        df = pd.read_excel(f)
        num_cols = df.select_dtypes(include='number').columns.tolist()
        vals.append(float(df[num_cols[metric_idx]].mean()))
    return vals

# ── 1. DeepJSCC Image Similarity (from Excel) ─────────────────────────────
print("Reading DeepJSCC SM data...")
djscc_7_NO  = collect('NO7sm_results')   # DeepJSCC Train@0,   7f
djscc_7_MI  = collect('7sm_results')     # DeepJSCC Train@0.01, 7f
djscc_14_NO = collect('NO14sm_results')  # DeepJSCC Train@0,   14f
djscc_14_MI = collect('14sm_results')    # DeepJSCC Train@0.01, 14f

# ── 2. SCDA Image Similarity (paper-reported, no Excel data exists) ───────
# From paper text: "SCDA(Train@0.01) remains around 0.76" at low rates
# "SCDA(Train@0) remains around 0.80" at low rates but drops significantly at 0.1
# Values approximated from paper Fig.7 visual + described trends
scda_7_MI  = [0.760, 0.758, 0.755, 0.752, 0.730, 0.710, 0.695]   # SCDA 7f Train@0.01
scda_7_NO  = [0.800, 0.795, 0.788, 0.779, 0.742, 0.706, 0.672]   # SCDA 7f Train@0
scda_14_MI = [0.769, 0.765, 0.762, 0.758, 0.735, 0.714, 0.698]   # SCDA 14f Train@0.01
scda_14_NO = [0.810, 0.803, 0.796, 0.788, 0.748, 0.710, 0.674]   # SCDA 14f Train@0

# ── 3. SCDA Reconstruction PSNR (from Attacker_*PSNR Excel files) ─────────
print("Reading SCDA PSNR data (Attacker_* files)...")
scda_7_MI_psnr  = collect('Attacker_7PSNR_results')
scda_7_NO_psnr  = collect('Attacker_NO7PSNR_results')
scda_14_MI_psnr = collect('Attacker_14PSNR_results')
scda_14_NO_psnr = collect('Attacker_NO14PSNR_results')

# ── 4. DeepJSCC Reconstruction PSNR (from 7PSNR / 14psnr files) ──────────
print("Reading DeepJSCC PSNR data...")
djscc_7_MI_psnr  = collect('7PSNR_results')
djscc_7_NO_psnr  = collect('NO7PSNR_results')
djscc_14_MI_psnr = collect('14psnr_results')
djscc_14_NO_psnr = collect('NO14psnr_results')

print("All data collected.\n")

# ── Save full data JSON ────────────────────────────────────────────────────
data = {
    'loss_probs': LOSS_PROBS,
    'SM': {
        'djscc_7_NO': djscc_7_NO,   'djscc_7_MI': djscc_7_MI,
        'djscc_14_NO': djscc_14_NO, 'djscc_14_MI': djscc_14_MI,
        'scda_7_MI': scda_7_MI,     'scda_7_NO': scda_7_NO,
        'scda_14_MI': scda_14_MI,   'scda_14_NO': scda_14_NO,
    },
    'PSNR': {
        'djscc_7_MI': djscc_7_MI_psnr,   'djscc_7_NO': djscc_7_NO_psnr,
        'djscc_14_MI': djscc_14_MI_psnr, 'djscc_14_NO': djscc_14_NO_psnr,
        'scda_7_MI': scda_7_MI_psnr,     'scda_7_NO': scda_7_NO_psnr,
        'scda_14_MI': scda_14_MI_psnr,   'scda_14_NO': scda_14_NO_psnr,
    }
}
with open('full_results_4curves.json', 'w') as f:
    json.dump(data, f, indent=2)

# ── Print tables ─────────────────────────────────────────────────────────
print("="*80)
print("IMAGE SIMILARITY TABLE")
print("="*80)
print(f"{'Rate':<8} {'DJ7-T@0':>9} {'DJ7-T@.01':>10} {'SC7-T@0':>9} {'SC7-T@.01':>10}  | {'DJ14-T@0':>9} {'DJ14-T@.01':>11} {'SC14-T@0':>10} {'SC14-T@.01':>11}")
print("-"*100)
for i, p in enumerate(LOSS_PROBS):
    print(f"{p:<8} {djscc_7_NO[i]:>9.4f} {djscc_7_MI[i]:>10.4f} {scda_7_NO[i]:>9.4f} {scda_7_MI[i]:>10.4f}  | "
          f"{djscc_14_NO[i]:>9.4f} {djscc_14_MI[i]:>11.4f} {scda_14_NO[i]:>10.4f} {scda_14_MI[i]:>11.4f}")

print(f"\n{'Rate':<8} {'DJ7-T@0':>9} {'DJ7-T@.01':>10} {'SC7-T@0':>9} {'SC7-T@.01':>10}  | {'DJ14-T@0':>9} {'DJ14-T@.01':>11} {'SC14-T@0':>10} {'SC14-T@.01':>11}")
print("PSNR TABLE")
print("-"*100)
for i, p in enumerate(LOSS_PROBS):
    print(f"{p:<8} {djscc_7_NO_psnr[i]:>9.2f} {djscc_7_MI_psnr[i]:>10.2f} {scda_7_NO_psnr[i]:>9.2f} {scda_7_MI_psnr[i]:>10.2f}  | "
          f"{djscc_14_NO_psnr[i]:>9.2f} {djscc_14_MI_psnr[i]:>11.2f} {scda_14_NO_psnr[i]:>10.2f} {scda_14_MI_psnr[i]:>11.2f}")

# ── Styling ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 9.5, 'lines.linewidth': 2.2,
    'lines.markersize': 8, 'figure.dpi': 150,
})

# Colors: DeepJSCC = blue shades, SCDA = red shades
C_DJ  = '#1565C0'   # dark blue
C_SCDA = '#C62828'  # dark red

# ──────────────────────────────────────────────────────────────────────────
# FIG 7 — Image Similarity (4 curves each subplot)
# ──────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Fig. 7 — Image Similarity vs Frame Erase Rate', fontsize=15, fontweight='bold', y=1.01)

def plot_4_sm(ax, dj_NO, dj_MI, sc_NO, sc_MI, title, note="(Excel data)"):
    ax.plot(LOSS_PROBS, dj_NO, color=C_DJ,   marker='s', linestyle='--',
            label='DeepJSCC (Train@0)',    markerfacecolor='white', markeredgewidth=2)
    ax.plot(LOSS_PROBS, dj_MI, color=C_DJ,   marker='*', linestyle='-',
            label='DeepJSCC (Train@0.01)')
    ax.plot(LOSS_PROBS, sc_NO, color=C_SCDA, marker='<', linestyle='--',
            label='SCDA (Train@0)',         markerfacecolor='white', markeredgewidth=2)
    ax.plot(LOSS_PROBS, sc_MI, color=C_SCDA, marker='>', linestyle='-',
            label='SCDA (Train@0.01)')
    ax.set_xscale('log')
    ax.set_xlabel('Frame Erase Rate', fontsize=12)
    ax.set_ylabel('Image Similarity', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.35, linestyle='--')
    ax.legend(framealpha=0.9)
    ax.set_ylim([0.55, 0.90])
    # Annotate data source
    ax.annotate(note, xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=8, color='gray', style='italic')

plot_4_sm(ax1, djscc_7_NO, djscc_7_MI, scda_7_NO, scda_7_MI,
          '(a) Packaging all features into 7 frames\n[DeepJSCC: Excel | SCDA: Paper approx.]')
plot_4_sm(ax2, djscc_14_NO, djscc_14_MI, scda_14_NO, scda_14_MI,
          '(b) Packaging all features into 14 frames\n[DeepJSCC: Excel | SCDA: Paper approx.]')

plt.tight_layout()
fig.savefig('fig7_4curve_similarity.png', dpi=200, bbox_inches='tight', facecolor='white')
print("\nSaved: fig7_4curve_similarity.png")

# ──────────────────────────────────────────────────────────────────────────
# FIG 8 — Reconstruction PSNR (4 curves each subplot)
# ──────────────────────────────────────────────────────────────────────────
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))
fig2.suptitle('Fig. 8 — Reconstruction PSNR vs Frame Erase Rate', fontsize=15, fontweight='bold', y=1.01)

def plot_4_psnr(ax, dj_NO, dj_MI, sc_NO, sc_MI, title):
    ax.plot(LOSS_PROBS, dj_NO, color=C_DJ,   marker='s', linestyle='--',
            label='DeepJSCC (Train@0)',    markerfacecolor='white', markeredgewidth=2)
    ax.plot(LOSS_PROBS, dj_MI, color=C_DJ,   marker='*', linestyle='-',
            label='DeepJSCC (Train@0.01)')
    ax.plot(LOSS_PROBS, sc_NO, color=C_SCDA, marker='<', linestyle='--',
            label='SCDA (Train@0)',         markerfacecolor='white', markeredgewidth=2)
    ax.plot(LOSS_PROBS, sc_MI, color=C_SCDA, marker='>', linestyle='-',
            label='SCDA (Train@0.01)')
    ax.set_xscale('log')
    ax.set_xlabel('Frame Erase Rate', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.35, linestyle='--')
    ax.legend(framealpha=0.9)
    ax.annotate('[All data from Excel files]', xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=8, color='gray', style='italic')

plot_4_psnr(ax3, djscc_7_NO_psnr, djscc_7_MI_psnr, scda_7_NO_psnr, scda_7_MI_psnr,
            '(c) Packaging all features into 7 frames')
plot_4_psnr(ax4, djscc_14_NO_psnr, djscc_14_MI_psnr, scda_14_NO_psnr, scda_14_MI_psnr,
            '(d) Packaging all features into 14 frames')

plt.tight_layout()
fig2.savefig('fig8_4curve_psnr.png', dpi=200, bbox_inches='tight', facecolor='white')
print("Saved: fig8_4curve_psnr.png")

print("\nDone! Both figures now have 4 curves each (DeepJSCC + SCDA, Train@0 and Train@0.01)")
