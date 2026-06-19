import os
import pandas as pd
import matplotlib.pyplot as plt

p_values = [0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1]
base_dir = r"d:\ULSC-main\DJSCC"

def get_mean(file_template, p, col_idx=1):
    path = os.path.join(base_dir, file_template.format(p))
    if os.path.exists(path):
        df = pd.read_excel(path)
        # return the mean of the second column
        return df.iloc[:, col_idx].mean()
    else:
        print(f"File not found: {path}")
        return None

def plot_graph(title, file_templates, labels, styles, ylabel, filename):
    plt.figure(figsize=(8, 6))
    
    for template, label, style in zip(file_templates, labels, styles):
        means = [get_mean(template, p) for p in p_values]
        if all(v is not None for v in means):
            # The demo uses log-like scale but spaced evenly, so we just plot them 
            # as categorical or use semilogx
            plt.semilogx(p_values, means, style['fmt'], color=style['color'], label=label, markersize=8, linewidth=2, markerfacecolor='none' if style.get('hollow') else None)

    plt.xlabel("Packet Loss Rate")
    plt.ylabel(ylabel)
    plt.title(title)
    # The x ticks in demo are the exact p values
    plt.xticks(p_values, [str(p) for p in p_values])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, filename), dpi=300)
    print(f"Saved {filename}")

# --- Plot SM for 7 Frames ---
templates_7sm = [
    "7sm_results_{}.xlsx",
    "NO7sm_results_{}.xlsx",
]
labels_7sm = [
    "DeepJSCC with 7 Frames (Training at 0.01)",
    "DeepJSCC with 7 Frames (Training at 0)",
]
styles_7sm = [
    {'fmt': '--*', 'color': 'navy'},
    {'fmt': '-*', 'color': 'navy'},
]
plot_graph("(a) Packaging all features into 7 frames (Image Similarity)", templates_7sm, labels_7sm, styles_7sm, "Image Similarity", "graph_7frames_sm.png")

# --- Plot SM for 14 Frames ---
templates_14sm = [
    "14sm_results_{}.xlsx",
    "NO14sm_results_{}.xlsx",
]
labels_14sm = [
    "DeepJSCC with 14 Frames (Training at 0.01)",
    "DeepJSCC with 14 Frames (Training at 0)",
]
styles_14sm = [
    {'fmt': '--*', 'color': 'green'},
    {'fmt': '-*', 'color': 'green'},
]
plot_graph("(b) Packaging all features into 14 frames (Image Similarity)", templates_14sm, labels_14sm, styles_14sm, "Image Similarity", "graph_14frames_sm.png")

# --- Plot PSNR for 7 Frames ---
templates_7psnr = [
    "7PSNR_results_{}.xlsx",
    "NO7PSNR_results_{}.xlsx",
    "Attacker_7PSNR_results_{}.xlsx",
    "Attacker_NO7PSNR_results_{}.xlsx"
]
labels_7psnr = [
    "DeepJSCC with 7 Frames (Training at 0.01)",
    "DeepJSCC with 7 Frames (Training at 0)",
    "SCDA with 7 Frames (Training at 0.01)",
    "SCDA with 7 Frames (Training at 0)"
]
styles_7psnr = [
    {'fmt': '--*', 'color': 'navy'},
    {'fmt': '-*', 'color': 'navy'},
    {'fmt': '-->', 'color': 'red', 'hollow': True},
    {'fmt': '->', 'color': 'red', 'hollow': True}
]
plot_graph("(c) Packaging all features into 7 frames (PSNR)", templates_7psnr, labels_7psnr, styles_7psnr, "PSNR", "graph_7frames_psnr.png")

# --- Plot PSNR for 14 Frames ---
templates_14psnr = [
    "14psnr_results_{}.xlsx",
    "NO14psnr_results_{}.xlsx",
    "Attacker_14PSNR_results_{}.xlsx",
    "Attacker_NO14PSNR_results_{}.xlsx"
]
labels_14psnr = [
    "DeepJSCC with 14 Frames (Training at 0.01)",
    "DeepJSCC with 14 Frames (Training at 0)",
    "SCDA with 14 Frames (Training at 0.01)",
    "SCDA with 14 Frames (Training at 0)"
]
styles_14psnr = [
    {'fmt': '--*', 'color': 'green'},
    {'fmt': '-*', 'color': 'green'},
    {'fmt': '--<', 'color': 'purple', 'hollow': True},
    {'fmt': '-<', 'color': 'purple', 'hollow': True}
]
plot_graph("(d) Packaging all features into 14 frames (PSNR)", templates_14psnr, labels_14psnr, styles_14psnr, "PSNR", "graph_14frames_psnr.png")

