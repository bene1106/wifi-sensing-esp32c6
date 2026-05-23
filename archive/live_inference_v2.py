"""
Live Präsenz-Erkennung mit dem FRISCH trainierten Modell.
Schreibt zusätzlich ein Log in CSV mit Zeitstempel + Vorhersage.
"""

import serial
import re
import time
import numpy as np
import joblib
from collections import deque
from datetime import datetime

SERIAL_PORT = 'COM5'
BAUDRATE = 921600
WINDOW_SIZE = 90        # gleich wie im Training
WINDOW_STEP = 10        # alle ~0.1 Sek eine Vorhersage
N_SUBCARRIERS = 64
MODEL_PATH = 'rf_model_fresh.pkl'
LOG_PATH = f'live_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

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

print("📂 Lade Modell ...")
clf = joblib.load(MODEL_PATH)
print(f"✅ Modell geladen: {MODEL_PATH}\n")

ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
buffer = deque(maxlen=WINDOW_SIZE)
packet_count = 0

print("="*60)
print("🔴 LIVE PRÄSENZ-ERKENNUNG")
print("="*60)
print(f"Log-Datei: {LOG_PATH}")
print("Strg+C zum Beenden\n")

log_file = open(LOG_PATH, 'w', encoding='utf-8')
log_file.write("timestamp,iso_time,prediction,confidence_absent,confidence_present\n")

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
            classes = clf.classes_
            conf_absent = proba[list(classes).index('absent')] * 100 if 'absent' in classes else 0
            conf_present = proba[list(classes).index('present')] * 100 if 'present' in classes else 0
            
            # Log
            ts = time.time()
            iso = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            log_file.write(f"{ts:.3f},{iso},{pred},{conf_absent:.2f},{conf_present:.2f}\n")
            log_file.flush()
            
            # Display
            if pred == 'present':
                emoji = '🪑'
                bar_color = '🟢'
            else:
                emoji = '🚪'
                bar_color = '🔵'
            confidence = max(conf_absent, conf_present)
            bar = bar_color * int(confidence / 5)
            print(f"\r{iso} | {emoji} {pred.upper():<8s} | "
                  f"abs:{conf_absent:5.1f}% pres:{conf_present:5.1f}% {bar:<20s}",
                  end='', flush=True)

except KeyboardInterrupt:
    print(f"\n\n✅ Beendet. {packet_count} Pakete verarbeitet.")
    print(f"📁 Log: {LOG_PATH}")
finally:
    ser.close()
    log_file.close()