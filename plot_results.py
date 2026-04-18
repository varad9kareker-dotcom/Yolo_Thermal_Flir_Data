import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('D:/YOLO_Custom/runs/detect/flir_4class_v13/results.csv')
df.columns = df.columns.str.strip()

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('FLIR Training Results - flir_4class_v13', fontsize=14, fontweight='bold')

# ── Row 1: Losses ─────────────────────────────────────────────────────
df['train/box_loss'].plot(ax=axes[0,0], title='Box Loss (train)',   color='red')
df['train/cls_loss'].plot(ax=axes[0,1], title='Class Loss (train)', color='orange')
df['train/dfl_loss'].plot(ax=axes[0,2], title='DFL Loss (train)',   color='purple')

# ── Row 2: Metrics ────────────────────────────────────────────────────
df['metrics/mAP50(B)'].plot(      ax=axes[1,0], title='mAP@50',     color='green')
df['metrics/mAP50-95(B)'].plot(   ax=axes[1,1], title='mAP@50-95',  color='blue')
df['metrics/precision(B)'].plot(  ax=axes[1,2], title='Precision',   color='teal')

for ax in axes.flat:
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/YOLO_Custom/training_curves.png', dpi=150)
plt.show()
print("Saved: training_curves.png")

# print best epoch stats
best_idx = df['metrics/mAP50(B)'].idxmax()
best     = df.loc[best_idx]

print(f"\nBest Epoch:  {int(best['epoch'])}")
print(f"mAP@50:      {best['metrics/mAP50(B)']:.4f}")
print(f"mAP@50-95:   {best['metrics/mAP50-95(B)']:.4f}")
print(f"Precision:   {best['metrics/precision(B)']:.4f}")
print(f"Recall:      {best['metrics/recall(B)']:.4f}")
print(f"Box Loss:    {best['train/box_loss']:.4f}")