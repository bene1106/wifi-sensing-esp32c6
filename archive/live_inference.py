"""
WiFi-Sensing: Live Präsenz-Erkennung
Liest CSI vom RX-Board und zeigt in Echtzeit absent/present mit Confidence.
"""

import serial
import re
import time
import numpy as np
import joblib
from collections import deque

# ========== KONFIG ==========
SERIAL_PORT = 'COM5'
BAUDRATE = 921600
WINDOW_SIZE = 180          # ~2 Sek bei 90 Hz
WINDOW_STEP = 18           # Prediction alle 0.2 Sek
N_SUBCARRIERS = 64
MODEL_PATH = 'confusion_matrix_cv.png'  # Platzhalter, wir nutzen das echte Modell

# ========== MODELL LADEN ==========
# Wir trainieren on-the-fly nochmal, weil rf_model.pkl von der alten v1 ist
print("📂 Lade Modell ...")
print("   (Training fresh aus den Daten für Live-Inferenz)")

import os
import glob
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = 'data'
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

# Training-Daten laden (große Dateien)
big_files = sorted([
    f for f in glob.glob(os.path.join(DATA_DIR, '*.csv'))
    if len(pd.read_csv(f, usecols=['label'])) >= 5000
])
by_class = {}
for f in big_files:
    cls = os.path.basename(f).split('_')[0]
    by_class.setdefault(cls, []).append(f)
selected = [sorted(fl, key=lambda x: os.path.getsize(x), reverse=True)[0]
            for fl in by_class.values()]

print("   Training auf:", [os.path.basename(f) for f in selected])
X_train, y_train = [], []
for f in selected:
    df = pd.read_csv(f)
    label = LABEL_MAP.get(df['label'].iloc[0], df['label'].iloc[0])
    amps = []
    for raw in df['raw_csi_line']:
        a = parse_csi_line(raw)
        if a is not None and len(a) >= N_SUBCARRIERS:
            amps.append(a[:N_SUBCARRIERS])
    amps = np.array(amps, dtype=np.float32)
    for start in range(0, len(amps) - WINDOW_SIZE + 1, 90):
        X_train.append(features_from_window(amps[start:start+WINDOW_SIZE]))
        y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
clf = RandomForestClassifier(n_estimators=200, max_depth=20,
                              random_state=42, n_jobs=-1, class_weight='balanced')
clf.fit(X_train, y_train)
print(f"✅ Modell trainiert mit {len(X_train)} Windows.\n")

# ========== LIVE ==========
print("="*60)
print("🔴 LIVE PRÄSENZ-ERKENNUNG")
print("="*60)
print(f"Port: {SERIAL_PORT} @ {BAUDRATE} Baud")
print(f"Window: {WINDOW_SIZE} Pakete | Step: {WINDOW_STEP} Pakete")
print("Strg+C zum Beenden\n")

ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
buffer = deque(maxlen=WINDOW_SIZE)
packet_count = 0
last_pred = None

try:
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line.startswith('CSI_DATA'):
            continue
        amp = parse_csi_line(line)
        if amp is None or len(amp) < N_SUBCARRIERS:
            continue
        buffer.append(amp[:N_SUBCARRIERS])
        packet_count += 1

        if len(buffer) >= WINDOW_SIZE and packet_count % WINDOW_STEP == 0:
            w = np.array(buffer)
            feat = features_from_window(w).reshape(1, -1)
            pred = clf.predict(feat)[0]
            proba = clf.predict_proba(feat)[0]
            confidence = proba.max() * 100

            # Visualisierung
            if pred == 'present':
                emoji = '🪑'
                bar = '🟢' * int(confidence / 5)
            else:
                emoji = '🚪'
                bar = '🔵' * int(confidence / 5)

            # Linie neu schreiben (Carriage Return)
            print(f"\r{emoji} {pred.upper():<8s} | Confidence: {confidence:5.1f}% {bar:<20s}",
                  end='', flush=True)

except KeyboardInterrupt:
    print(f"\n\n✅ Beendet. {packet_count} Pakete verarbeitet.")
finally:
    ser.close()