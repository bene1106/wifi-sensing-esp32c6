"""
WiFi-Sensing: Test mit FRISCHEN Daten
Nimmt die neuesten 2 Aufnahmen (eine pro Klasse) und macht time-based CV.
Wenn dieses Modell auch live versagt → Klassen sind physikalisch nicht trennbar.
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
WINDOW_SIZE = 90           # ~1 Sek bei 90 Hz (kleineres Window, da nur 60s Daten)
WINDOW_STEP = 30
N_SUBCARRIERS = 64

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

def features_from_window(w):
    diff = np.diff(w, axis=0)
    return np.concatenate([
        w.std(axis=0),
        np.abs(diff).mean(axis=0),
        diff.std(axis=0),
        np.percentile(w, 90, axis=0) - np.percentile(w, 10, axis=0),
    ])

def load_amps(filepath):
    df = pd.read_csv(filepath)
    label = LABEL_MAP.get(df['label'].iloc[0], df['label'].iloc[0])
    amps = []
    for raw in df['raw_csi_line']:
        a = parse_csi_line(raw)
        if a is not None and len(a) >= N_SUBCARRIERS:
            amps.append(a[:N_SUBCARRIERS])
    return np.array(amps, dtype=np.float32), label

# ========== NEUESTE DATEIEN PRO KLASSE ==========
files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))

by_class = {}
for f in files:
    cls = os.path.basename(f).split('_')[0]
    by_class.setdefault(cls, []).append(f)

# Pro Klasse die NEUSTE Datei nehmen (sortiert nach Timestamp im Namen)
selected = []
for cls, flist in by_class.items():
    newest = sorted(flist)[-1]   # letzte alphabetisch = neueste
    selected.append(newest)

print("📁 Verwendete (NEUESTE) Dateien pro Klasse:")
for f in selected:
    print(f"   • {os.path.basename(f)}")

# ========== LADEN + WINDOWS ==========
all_X, all_y, all_session = [], [], []
for sid, f in enumerate(selected):
    amps, lbl = load_amps(f)
    print(f"\n  {os.path.basename(f)}: {len(amps)} Pakete → '{lbl}'")
    
    n_windows = 0
    for start in range(0, len(amps) - WINDOW_SIZE + 1, WINDOW_STEP):
        all_X.append(features_from_window(amps[start:start+WINDOW_SIZE]))
        all_y.append(lbl)
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

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    clf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                  random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    fold_accs.append(acc)
    all_true.extend(y_te); all_pred.extend(y_pred)
    print(f"   Fold {fold+1}: Train {len(X_tr)} | Test {len(X_te)} | Acc {acc*100:.2f}%")

mean_acc = np.mean(fold_accs)
std_acc = np.std(fold_accs)
print(f"\n🎯 Mean CV Accuracy: {mean_acc*100:.2f}% (±{std_acc*100:.2f}%)")
print(classification_report(all_true, all_pred, zero_division=0))

labels = sorted(set(all_true) | set(all_pred))
cm = confusion_matrix(all_true, all_pred, labels=labels)
print("📊 Confusion Matrix:")
print(f"   {'':10s} " + " ".join(f"{l:>10s}" for l in labels))
for i, l in enumerate(labels):
    print(f"   {l:10s} " + " ".join(f"{cm[i,j]:>10d}" for j in range(len(labels))))

# ========== MODELL AUF ALLES TRAINIEREN UND SPEICHERN ==========
print("\n💾 Trainiere finales Modell auf ALLEN frischen Daten ...")
clf_final = RandomForestClassifier(n_estimators=200, max_depth=20,
                                    random_state=42, n_jobs=-1, class_weight='balanced')
clf_final.fit(X, y)
joblib.dump(clf_final, 'rf_model_fresh.pkl')
print("✅ Gespeichert: rf_model_fresh.pkl")

# Plot
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels); ax.set_yticklabels(labels)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Fresh Data CV — Acc: {mean_acc*100:.2f}%')
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im); plt.tight_layout()
plt.savefig('confusion_matrix_fresh.png', dpi=120)
plt.show()