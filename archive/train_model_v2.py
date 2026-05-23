"""
WiFi-Sensing v2: Robustere Features
Nutzt nur kalibrations-unabhängige Features (Variabilität, nicht absolute Pegel).
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
MIN_PACKETS_FOR_FILE = 5000   # Ignoriere zu kleine Testdateien

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
    file_label = df['label'].iloc[0]
    label = LABEL_MAP.get(file_label, file_label)
    amps = []
    for raw in df['raw_csi_line']:
        a = parse_csi_line(raw)
        if a is not None and len(a) >= N_SUBCARRIERS:
            amps.append(a[:N_SUBCARRIERS])
    arr = np.array(amps, dtype=np.float32)
    print(f"{len(arr)} Pakete → '{label}'")
    return arr, label

def make_windows(amps, label):
    """ROBUSTE FEATURES: nur Variabilität, keine absoluten Pegel."""
    X, y = [], []
    for start in range(0, len(amps) - WINDOW_SIZE + 1, WINDOW_STEP):
        w = amps[start:start + WINDOW_SIZE]   # (T, n_sub)

        # Normalisierung: pro Subcarrier auf Mean=0, Std=1
        w_norm = (w - w.mean(axis=0, keepdims=True)) / (w.std(axis=0, keepdims=True) + 1e-6)

        # Differenzen zwischen aufeinanderfolgenden Paketen
        diff = np.diff(w, axis=0)  # (T-1, n_sub)

        features = np.concatenate([
            w.std(axis=0),                       # zeitliche Variation (key feature!)
            np.abs(diff).mean(axis=0),           # mittlere Änderungsrate
            diff.std(axis=0),                    # Std der Differenzen
            np.percentile(w, 90, axis=0) - np.percentile(w, 10, axis=0),  # robuster Range
        ])
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

# ========== LADEN ==========
files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
print("\n📊 Verfügbare Dateien (kleine ignoriert):")
valid_files = []
for f in files:
    df_quick = pd.read_csv(f, usecols=['label'])
    n = len(df_quick)
    print(f"  {os.path.basename(f):<40s} {n:>6d} Pakete", 
          "✅" if n >= MIN_PACKETS_FOR_FILE else "⏭️ ignoriert (Testdatei)")
    if n >= MIN_PACKETS_FOR_FILE:
        valid_files.append(f)

# Train/Val Split: pro Klasse größte Datei = Training, andere = Validation
by_class = {}
for f in valid_files:
    cls = os.path.basename(f).split('_')[0]
    by_class.setdefault(cls, []).append(f)

train_files, val_files = [], []
for cls, flist in by_class.items():
    # Größte Datei nach Pakete als Training
    flist_sorted = sorted(flist, key=lambda x: os.path.getsize(x), reverse=True)
    train_files.append(flist_sorted[0])
    val_files.extend(flist_sorted[1:])

print(f"\n📁 Training: {[os.path.basename(f) for f in train_files]}")
print(f"📁 Validation: {[os.path.basename(f) for f in val_files]}")

print("\n🔬 Training-Daten:")
X_train, y_train = [], []
for f in train_files:
    a, l = load_csv(f)
    X, y = make_windows(a, l)
    X_train.append(X); y_train.append(y)
X_train = np.vstack(X_train); y_train = np.concatenate(y_train)
print(f"   → {X_train.shape[0]} Windows, {X_train.shape[1]} Features")

print("\n🔬 Validation-Daten:")
X_val, y_val = [], []
for f in val_files:
    a, l = load_csv(f)
    X, y = make_windows(a, l)
    X_val.append(X); y_val.append(y)
X_val = np.vstack(X_val); y_val = np.concatenate(y_val)
print(f"   → {X_val.shape[0]} Windows")

# ========== TRAINING ==========
print("\n🤖 Training Random Forest ...")
clf = RandomForestClassifier(n_estimators=200, max_depth=20,
                              random_state=42, n_jobs=-1, class_weight='balanced')
clf.fit(X_train, y_train)

# ========== EVAL ==========
y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"\n🎯 Validation Accuracy: {acc*100:.2f} %\n")
print(classification_report(y_val, y_pred, zero_division=0))

labels = sorted(set(y_val) | set(y_pred))
cm = confusion_matrix(y_val, y_pred, labels=labels)
print("📊 Confusion Matrix:")
print(f"   {'':10s} " + " ".join(f"{l:>10s}" for l in labels))
for i, l in enumerate(labels):
    print(f"   {l:10s} " + " ".join(f"{cm[i,j]:>10d}" for j in range(len(labels))))

# Auch Training-Accuracy zeigen (Sanity)
train_acc = clf.score(X_train, y_train)
print(f"\n📊 Sanity: Training Accuracy = {train_acc*100:.2f}% (sollte hoch sein)")

# Plot
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels); ax.set_yticklabels(labels)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix v2 — Acc: {acc*100:.2f}%')
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im); plt.tight_layout()
plt.savefig('confusion_matrix_v2.png', dpi=120)
joblib.dump(clf, 'rf_model_v2.pkl')
print("\n📁 Saved: confusion_matrix_v2.png, rf_model_v2.pkl")
plt.show()