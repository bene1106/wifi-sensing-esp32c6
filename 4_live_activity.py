"""
Live Aktivitäts-Erkennung mit dem trainierten 3-Klassen-Modell.
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
WINDOW_SIZE = 90
WINDOW_STEP = 10
N_SUBCARRIERS = 64
MODEL_PATH = 'rf_model_activity.pkl'

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
    diff2 = np.diff(diff, axis=0)
    return np.concatenate([
        w.std(axis=0),
        np.abs(diff).mean(axis=0),
        diff.std(axis=0),
        np.percentile(w, 90, axis=0) - np.percentile(w, 10, axis=0),
        np.abs(diff2).mean(axis=0),
    ])

print("📂 Lade Modell ...")
clf = joblib.load(MODEL_PATH)
classes = list(clf.classes_)
print(f"✅ Modell geladen. Klassen: {classes}\n")

LOG_PATH = f'live_activity_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

EMOJIS = {'still': '🧘', 'typing': '⌨️ ', 'waving': '👋'}
COLORS = {'still': '🟦', 'typing': '🟩', 'waving': '🟧'}

ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
buffer = deque(maxlen=WINDOW_SIZE)
packet_count = 0

print("="*70)
print("🔴 LIVE AKTIVITÄTS-ERKENNUNG")
print("="*70)
print(f"Log: {LOG_PATH}")
print("Strg+C zum Beenden\n")

log_file = open(LOG_PATH, 'w', encoding='utf-8')
log_file.write("timestamp,iso_time,prediction," + ",".join(f"conf_{c}" for c in classes) + "\n")

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
            confs = {c: proba[i]*100 for i, c in enumerate(classes)}

            ts = time.time()
            iso = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            log_file.write(f"{ts:.3f},{iso},{pred}," +
                           ",".join(f"{confs[c]:.2f}" for c in classes) + "\n")
            log_file.flush()

            emoji = EMOJIS.get(pred, '?')
            color = COLORS.get(pred, '⬜')
            bar = color * int(confs[pred] / 5)
            conf_str = " | ".join(f"{c}: {confs[c]:5.1f}%" for c in classes)
            print(f"\r{iso} {emoji} {pred.upper():<8s} | {conf_str} {bar:<20s}",
                  end='', flush=True)

except KeyboardInterrupt:
    print(f"\n\n✅ Beendet. {packet_count} Pakete verarbeitet.")
    print(f"📁 Log: {LOG_PATH}")
finally:
    ser.close()
    log_file.close()