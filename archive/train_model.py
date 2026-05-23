"""
WiFi-Sensing: Präsenz-Erkennung — Modell-Training
Verwendet CSI-Daten aus den CSV-Aufnahmen, trainiert einen Random Forest
zur Klassifikation absent vs. present.
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

# ============ KONFIGURATION ============
DATA_DIR = 'data'
WINDOW_SIZE = 180        # ~2 Sek bei 90 Hz
WINDOW_STEP = 90         # 50% Overlap
N_SUBCARRIERS = 64       # Wir nutzen ersten halben Sub-Carrier-Block

# Label-Mapping: Dateiname-Label → Use-Case-Klasse
LABEL_MAP = {
    'empty':  'absent',
    'motion': 'present',
}

# Welche Dateien sind Training, welche Validation?
# Heuristik: chronologisch sortieren, JÜNGSTE pro Klasse → Validation
def split_train_val(files):
    """Pro Klasse: älteste = Training, neueste = Validation"""
    by_class = {}
    for f in files:
        cls = os.path.basename(f).split('_')[0]
        by_class.setdefault(cls, []).append(f)
    train, val = [], []
    for cls, flist in by_class.items():
        flist.sort()  # ältere zuerst (Timestamp im Namen)
        train.extend(flist[:-1])  # alle außer letzte
        val.append(flist[-1])     # letzte = Validation
    return train, val

# ============ CSI-PARSER ============
def parse_csi_line(line):
    """Extrahiert Amplituden-Vektor aus einer CSI_DATA-Zeile."""
    match = re.search(r'\[([\d\s,\-]+)\]', line)
    if not match:
        return None
    try:
        values = np.array([int(x) for x in match.group(1).split(',')],
                          dtype=np.float32)
        if len(values) % 2 != 0:
            return None
        real = values[0::2]
        imag = values[1::2]
        return np.sqrt(real**2 + imag**2)
    except Exception:
        return None

# ============ DATEN LADEN ============
def load_csv(filepath):
    """Lädt eine CSV-Datei und gibt (amplitude_array, label) zurück."""
    print(f"  Lade {os.path.basename(filepath)} ...", end=' ')
    df = pd.read_csv(filepath)
    file_label = df['label'].iloc[0]
    use_case_label = LABEL_MAP.get(file_label, file_label)

    amplitudes = []
    for raw in df['raw_csi_line']:
        amp = parse_csi_line(raw)
        if amp is not None and len(amp) >= N_SUBCARRIERS:
            amplitudes.append(amp[:N_SUBCARRIERS])
    arr = np.array(amplitudes, dtype=np.float32)
    print(f"{len(arr)} Pakete → Klasse '{use_case_label}'")
    return arr, use_case_label

# ============ FEATURE-EXTRAKTION ============
def make_windows(amplitudes, label):
    """Sliding-Windows + Feature-Extraktion."""
    X, y = [], []
    for start in range(0, len(amplitudes) - WINDOW_SIZE + 1, WINDOW_STEP):
        window = amplitudes[start:start + WINDOW_SIZE]  # (WINDOW_SIZE, N_SUBCARRIERS)

        # Statistische Features pro Subcarrier
        features = np.concatenate([
            window.mean(axis=0),       # Mittelwert pro Subcarrier
            window.std(axis=0),        # Standardabweichung (Bewegungs-Indikator!)
            window.max(axis=0) - window.min(axis=0),  # Range
            np.median(window, axis=0), # Median
        ])
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

# ============ MAIN ============
print("="*60)
print("📊 WiFi-Sensing — Präsenz-Erkennung Training")
print("="*60)

files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
if not files:
    print(f"❌ Keine CSV-Dateien in {DATA_DIR}/ gefunden!")
    exit(1)

train_files, val_files = split_train_val(files)
print(f"\n📁 Training-Dateien:")
for f in train_files:
    print(f"   • {os.path.basename(f)}")
print(f"\n📁 Validation-Dateien:")
for f in val_files:
    print(f"   • {os.path.basename(f)}")

# Training-Daten
print("\n🔬 Lade Training-Daten:")
X_train_list, y_train_list = [], []
for f in train_files:
    amps, lbl = load_csv(f)
    X, y = make_windows(amps, lbl)
    X_train_list.append(X)
    y_train_list.append(y)
X_train = np.vstack(X_train_list)
y_train = np.concatenate(y_train_list)
print(f"\n   → Training-Set: {X_train.shape[0]} Windows, {X_train.shape[1]} Features")

# Validation-Daten
print("\n🔬 Lade Validation-Daten:")
X_val_list, y_val_list = [], []
for f in val_files:
    amps, lbl = load_csv(f)
    X, y = make_windows(amps, lbl)
    X_val_list.append(X)
    y_val_list.append(y)
X_val = np.vstack(X_val_list)
y_val = np.concatenate(y_val_list)
print(f"\n   → Validation-Set: {X_val.shape[0]} Windows")

# ============ TRAINING ============
print("\n" + "="*60)
print("🤖 Training Random Forest ...")
print("="*60)
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',
)
clf.fit(X_train, y_train)
print("✅ Training abgeschlossen.")

# ============ EVALUATION ============
print("\n" + "="*60)
print("📈 Evaluation auf Validation-Set")
print("="*60)
y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print(f"\n🎯 Accuracy: {acc*100:.2f} %\n")
print("📋 Classification Report:")
print(classification_report(y_val, y_pred))

# Confusion Matrix
print("📊 Confusion Matrix:")
labels = sorted(set(y_val) | set(y_pred))
cm = confusion_matrix(y_val, y_pred, labels=labels)
print(f"   Reihen = True, Spalten = Predicted")
print(f"   {'':10s} " + " ".join(f"{l:>10s}" for l in labels))
for i, l in enumerate(labels):
    print(f"   {l:10s} " + " ".join(f"{cm[i,j]:>10d}" for j in range(len(labels))))

# Plot
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix — Accuracy: {acc*100:.2f}%')
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=120)
print("\n📁 Confusion Matrix gespeichert: confusion_matrix.png")

# ============ MODELL SPEICHERN ============
joblib.dump(clf, 'rf_model.pkl')
print("📁 Modell gespeichert: rf_model.pkl")

plt.show()