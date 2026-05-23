"""
WiFi-Sensing v3: Time-based Cross-Validation
Splittet innerhalb der großen Sessions, um Inter-Session-Drift zu eliminieren.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import joblib

DATA_DIR = 'data'
WINDOW_SIZE = 180
WINDOW_STEP = 90
N_SUBCARRIERS = 64
MIN_PACKETS = 5000

LABEL_MAP = {'empty': 'absent', 'motion': 'present'}

def parse_csi_line(line):
    match = re.search(r'\[([\d\s,\-]+)\]', line)
    if not match:
        return None
    try:
        v = np.array([int(x) for x in match.group(1).split(',')], dtype=np.float32)
        if len(v) % 2 != 0:
            return None
        return np.sqrt(v[0::2]**2 + v[1::2]**2)
    except Exception:
        return None

def load_csv(filepath):
    print(f"  Lade {os.path.basename(filepath)} ...", end=' ')
    df = pd.read_csv(filepath)
    label = LABEL_MAP.get(df['label'].iloc[0], df['label'].iloc[0])
    amps = []
    for raw in df['raw_csi_line']:
        a = parse_csi_line(raw)
        if a is not None and len(a) >= N_SUBCARRIERS:
            amps.append(a[:N_SUBCARRIERS])
    print(f"{len(amps)} Pakete → '{label}'")
    return np.array(amps, dtype=np.float32), label

def make_windows(amps, label):
    """Robuste, kalibrationsunabhängige Features."""
    X, y, idx = [], [], []
    for start in range(0, len(amps) - WINDOW_SIZE + 1, WINDOW_STEP):
        w = amps[start:start + WINDOW_SIZE]
        diff = np.diff(w, axis=0)
        features = np.concatenate([
            w.std(axis=0),
            np.abs(diff).mean(axis=0),
            diff.std(axis=0),
            np.percentile(w, 90, axis=0) - np.percentile(w, 10, axis=0),
        ])
        X.append(features)
        y.append(label)
        idx.append(start)
    return np.array(X), np.array(y), np.array(idx)

# ========== NUR DIE GROSSEN DATEIEN ==========
files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
big_files = []
for f in files:
    n = len(pd.read_csv(f, usecols=['label']))
    if n >= MIN_PACKETS:
        big_files.append(f)

# Wir wollen pro Klasse die größte Datei
by_class = {}
for f in big_files:
    cls = os.path.basename(f).split('_')[0]
    by_class.setdefault(cls, []).append(f)

selected = []
for cls, flist in by_class.items():
    flist_sorted = sorted(flist, key=lambda x: os.path.getsize(x), reverse=True)
    selected.append(flist_sorted[0])

print("📁 Verwendete Dateien (je größte pro Klasse):")
for f in selected:
    print(f"   • {os.path.basename(f)}")

# ========== LADEN ==========
print("\n🔬 Lade Daten:")
all_X, all_y, all_session = [], [], []
for sid, f in enumerate(selected):
    amps, lbl = load_csv(f)
    X, y, _ = make_windows(amps, lbl)
    all_X.append(X)
    all_y.append(y)
    all_session.append(np.full(len(X), sid))   # Session-ID

X = np.vstack(all_X)
y = np.concatenate(all_y)
session = np.concatenate(all_session)
print(f"\n   → {X.shape[0]} Windows gesamt")
unique, counts = np.unique(y, return_counts=True)
print(f"   → Klassen: {dict(zip(unique, counts))}")

# ========== TIME-BASED SPLITS innerhalb jeder Session ==========
print("\n🔬 5-Fold Time-Based Cross-Validation:")
print("   Jede Session wird in 5 zeitliche Blöcke geteilt.")
print("   Pro Fold: 1 Block Test, 4 Blöcke Training (pro Session).\n")

n_folds = 5
fold_accs = []
all_y_true = []
all_y_pred = []

for fold in range(n_folds):
    train_mask = np.zeros(len(X), dtype=bool)
    test_mask = np.zeros(len(X), dtype=bool)
    
    for sid in range(len(selected)):
        sess_idx = np.where(session == sid)[0]
        n = len(sess_idx)
        block_size = n // n_folds
        test_start = fold * block_size
        test_end = test_start + block_size if fold < n_folds - 1 else n
        test_mask[sess_idx[test_start:test_end]] = True
        train_mask[sess_idx] = ~test_mask[sess_idx]

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=20,
        random_state=42, n_jobs=-1, class_weight='balanced'
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    fold_accs.append(acc)
    all_y_true.extend(y_te)
    all_y_pred.extend(y_pred)
    print(f"   Fold {fold+1}: Train {len(X_tr)} | Test {len(X_te)} | Acc {acc*100:.2f}%")

mean_acc = np.mean(fold_accs)
std_acc = np.std(fold_accs)
print(f"\n🎯 Mean CV Accuracy: {mean_acc*100:.2f}% (±{std_acc*100:.2f}%)\n")

# ========== AGGREGIERTE CONFUSION MATRIX ==========
labels = sorted(set(all_y_true) | set(all_y_pred))
cm = confusion_matrix(all_y_true, all_y_pred, labels=labels)
print("📋 Classification Report (über alle Folds):")
print(classification_report(all_y_true, all_y_pred, zero_division=0))
print("📊 Confusion Matrix:")
print(f"   {'':10s} " + " ".join(f"{l:>10s}" for l in labels))
for i, l in enumerate(labels):
    print(f"   {l:10s} " + " ".join(f"{cm[i,j]:>10d}" for j in range(len(labels))))

# Plot
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels); ax.set_yticklabels(labels)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'CV Confusion Matrix — Mean Acc: {mean_acc*100:.2f}% (±{std_acc*100:.2f}%)')
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im); plt.tight_layout()
plt.savefig('confusion_matrix_cv.png', dpi=120)
print("\n📁 Saved: confusion_matrix_cv.png")
plt.show()