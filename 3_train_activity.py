"""
WiFi-Sensing: Aktivitäts-Erkennung (still / typing / waving)
Time-based 5-Fold CV auf den 3 neuesten Aufnahmen.
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
WINDOW_SIZE = 90          # ~1 Sek bei 88 Hz
WINDOW_STEP = 30
N_SUBCARRIERS = 64
CLASSES = ['still', 'typing', 'waving']

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

def features_from_window(w):
    """Robuste, kalibrationsunabhängige Features."""
    diff = np.diff(w, axis=0)
    diff2 = np.diff(diff, axis=0)  # Beschleunigung
    return np.concatenate([
        w.std(axis=0),                                  # zeitl. Variation
        np.abs(diff).mean(axis=0),                      # mittlere Änderung
        diff.std(axis=0),                               # Std der Änderungen
        np.percentile(w, 90, axis=0) - np.percentile(w, 10, axis=0),  # Range
        np.abs(diff2).mean(axis=0),                     # Beschleunigung (NEU!)
    ])

# ========== NEUESTE DATEIEN PRO KLASSE ==========
all_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
selected = []
for cls in CLASSES:
    cls_files = [f for f in all_files if os.path.basename(f).startswith(cls + '_')]
    if not cls_files:
        print(f"❌ Keine Datei für Klasse '{cls}' gefunden!")
        exit(1)
    selected.append(sorted(cls_files)[-1])  # neueste

print("📁 Dateien:")
for f in selected:
    print(f"   • {os.path.basename(f)}")

# ========== LADEN + WINDOWS ==========
print("\n🔬 Lade Daten:")
all_X, all_y, all_session = [], [], []
for sid, f in enumerate(selected):
    df = pd.read_csv(f)
    label = df['label'].iloc[0]
    amps = []
    for raw in df['raw_csi_line']:
        a = parse_csi_line(raw)
        if a is not None and len(a) >= N_SUBCARRIERS:
            amps.append(a[:N_SUBCARRIERS])
    amps = np.array(amps, dtype=np.float32)
    print(f"  {os.path.basename(f)}: {len(amps)} Pakete → '{label}'")

    n_windows = 0
    for start in range(0, len(amps) - WINDOW_SIZE + 1, WINDOW_STEP):
        all_X.append(features_from_window(amps[start:start+WINDOW_SIZE]))
        all_y.append(label)
        all_session.append(sid)
        n_windows += 1
    print(f"  → {n_windows} Windows")

X = np.array(all_X)
y = np.array(all_y)
session = np.array(all_session)
print(f"\n✅ Gesamt: {X.shape[0]} Windows, {X.shape[1]} Features")
unique, counts = np.unique(y, return_counts=True)
print(f"   Klassen: {dict(zip(unique, counts))}")

# ========== 5-FOLD TIME-BASED CV ==========
print("\n🔬 5-Fold Time-Based CV:\n")
n_folds = 5
fold_accs = []
all_true, all_pred = [], []

for fold in range(n_folds):
    train_mask = np.zeros(len(X), dtype=bool)
    test_mask = np.zeros(len(X), dtype=bool)
    for sid in range(len(selected)):
        idx = np.where(session == sid)[0]
        n = len(idx)
        block = n // n_folds
        ts = fold * block
        te = ts + block if fold < n_folds - 1 else n
        test_mask[idx[ts:te]] = True
        train_mask[idx] = ~test_mask[idx]

    clf = RandomForestClassifier(n_estimators=300, max_depth=25,
                                  random_state=42, n_jobs=-1,
                                  class_weight='balanced')
    clf.fit(X[train_mask], y[train_mask])
    y_pred = clf.predict(X[test_mask])
    acc = accuracy_score(y[test_mask], y_pred)
    fold_accs.append(acc)
    all_true.extend(y[test_mask]); all_pred.extend(y_pred)
    print(f"   Fold {fold+1}: Train {train_mask.sum()} | Test {test_mask.sum()} | Acc {acc*100:.2f}%")

mean_acc = np.mean(fold_accs)
std_acc = np.std(fold_accs)
print(f"\n🎯 Mean CV Accuracy: {mean_acc*100:.2f}% (±{std_acc*100:.2f}%)\n")
print(classification_report(all_true, all_pred, zero_division=0))

labels_sorted = sorted(set(all_true))
cm = confusion_matrix(all_true, all_pred, labels=labels_sorted)
print("📊 Confusion Matrix (True \\ Predicted):")
print(f"   {'':10s} " + " ".join(f"{l:>10s}" for l in labels_sorted))
for i, l in enumerate(labels_sorted):
    print(f"   {l:10s} " + " ".join(f"{cm[i,j]:>10d}" for j in range(len(labels_sorted))))

# ========== FINALES MODELL SPEICHERN ==========
print("\n💾 Trainiere finales Modell auf ALLEN Daten ...")
clf_final = RandomForestClassifier(n_estimators=300, max_depth=25,
                                    random_state=42, n_jobs=-1,
                                    class_weight='balanced')
clf_final.fit(X, y)
joblib.dump(clf_final, 'rf_model_activity.pkl')
print("✅ Gespeichert: rf_model_activity.pkl")

# ========== PLOT ==========
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(labels_sorted))); ax.set_yticks(range(len(labels_sorted)))
ax.set_xticklabels(labels_sorted); ax.set_yticklabels(labels_sorted)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Aktivitäts-Erkennung CV — Acc: {mean_acc*100:.2f}% (±{std_acc*100:.2f}%)')
for i in range(len(labels_sorted)):
    for j in range(len(labels_sorted)):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im); plt.tight_layout()
plt.savefig('confusion_matrix_activity.png', dpi=120)
plt.show()