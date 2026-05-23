"""
Dashboard Backend: Liest CSI vom RX-Board, macht Live-Inferenz,
stellt Daten als JSON-Endpoint bereit für das Browser-Frontend.
"""

import serial
import re
import time
import threading
import numpy as np
import joblib
from collections import deque
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

# ============ KONFIG ============
SERIAL_PORT = 'COM5'
BAUDRATE = 921600
WINDOW_SIZE = 90
WINDOW_STEP = 10
N_SUBCARRIERS = 64
MODEL_PATH = 'rf_model_activity.pkl'
SMOOTH_WINDOW = 5    # letzte 5 Predictions für Smoothing

# ============ HELPERS ============
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

# ============ MODELL LADEN ============
print("📂 Lade Modell ...")
clf = joblib.load(MODEL_PATH)
classes = list(clf.classes_)
print(f"✅ Modell geladen. Klassen: {classes}")

# ============ GLOBAL STATE ============
state = {
    'prediction': 'unknown',
    'smoothed_prediction': 'unknown',
    'confidences': {c: 0.0 for c in classes},
    'rssi': 0,
    'packet_count': 0,
    'rate': 0.0,
    'heatmap': [],         # letzte 50 Amplitudenvektoren
    'running': True,
}
state_lock = threading.Lock()
recent_predictions = deque(maxlen=SMOOTH_WINDOW)
recent_amps = deque(maxlen=50)

# ============ READER THREAD ============
def reader_thread():
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    buffer = deque(maxlen=WINDOW_SIZE)
    packet_count = 0
    start_time = time.time()
    last_rssi = 0

    while state['running']:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line.startswith('CSI_DATA'):
                continue
            
            # RSSI extrahieren (3. Feld nach Komma)
            try:
                parts = line.split(',')
                # CSI_DATA,seq,mac,rssi,... — RSSI ist meist Position 3
                rssi_val = int(parts[3])
                last_rssi = rssi_val
            except Exception:
                pass
            
            amp = parse_csi_line(line)
            if amp is None or len(amp) < N_SUBCARRIERS:
                continue
            amp = amp[:N_SUBCARRIERS]
            buffer.append(amp)
            recent_amps.append(amp.tolist())
            packet_count += 1

            if len(buffer) >= WINDOW_SIZE and packet_count % WINDOW_STEP == 0:
                w = np.array(buffer)
                feat = features_from_window(w).reshape(1, -1)
                pred = clf.predict(feat)[0]
                proba = clf.predict_proba(feat)[0]
                confs = {c: float(proba[i]*100) for i, c in enumerate(classes)}

                # Smoothing: häufigste Klasse in letzten N Predictions
                recent_predictions.append(pred)
                from collections import Counter
                smoothed = Counter(recent_predictions).most_common(1)[0][0]

                elapsed = time.time() - start_time
                rate = packet_count / elapsed if elapsed > 0 else 0

                with state_lock:
                    state['prediction'] = pred
                    state['smoothed_prediction'] = smoothed
                    state['confidences'] = confs
                    state['rssi'] = last_rssi
                    state['packet_count'] = packet_count
                    state['rate'] = rate
                    state['heatmap'] = list(recent_amps)
        except Exception as e:
            print(f"Reader error: {e}")
            time.sleep(0.5)

    ser.close()

# ============ FLASK APP ============
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory('.', '5_dashboard_frontend.html')

@app.route('/state')
def get_state():
    with state_lock:
        return jsonify(state)

# ============ START ============
if __name__ == '__main__':
    t = threading.Thread(target=reader_thread, daemon=True)
    t.start()
    print("\n" + "="*60)
    print("🌐 DASHBOARD LÄUFT")
    print("="*60)
    print("Öffne im Browser:  http://localhost:5000")
    print("Oder vom Handy:    http://<dein-PC-IP>:5000")
    print("Strg+C zum Beenden")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)