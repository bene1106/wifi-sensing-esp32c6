"""
Live CSI Heatmap Visualizer
Liest CSI-Daten vom ESP32-C6 RX-Board und zeigt sie als Heatmap.
"""

import serial
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ============ KONFIGURATION ============
SERIAL_PORT = 'COM5'       # RX-Board
BAUDRATE = 921600          # Wichtig: muss mit idf_monitor übereinstimmen!
WINDOW_SIZE = 200          # Anzahl Pakete in der Heatmap (~2 Sekunden bei 100Hz)
N_SUBCARRIERS = 128        # Wir nutzen nur die Amplitude pro Subcarrier

# ============ BUFFER ============
amplitude_buffer = deque(maxlen=WINDOW_SIZE)

# ============ CSI PARSER ============
def parse_csi_line(line):
    """Extrahiert CSI-Werte aus einer Zeile wie:
    CSI_DATA,8762,1a:00:..,−35,11,...,"[0,0,13,−12,16,...]"
    """
    try:
        # Finde die Werte in den eckigen Klammern
        match = re.search(r'\[([\d\s,\-]+)\]', line)
        if not match:
            return None
        raw = match.group(1)
        values = np.array([int(x) for x in raw.split(',')], dtype=np.float32)

        # CSI kommt als Paare (real, imag), wir berechnen Amplitude pro Paar
        if len(values) % 2 != 0:
            return None
        real = values[0::2]
        imag = values[1::2]
        amplitude = np.sqrt(real**2 + imag**2)
        return amplitude
    except Exception:
        return None

# ============ SERIAL VERBINDUNG ============
print(f"Öffne {SERIAL_PORT} mit {BAUDRATE} Baud ...")
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
print("Verbunden! Warte auf CSI-Daten ... (Strg+C zum Beenden)")

# ============ PLOT SETUP ============
plt.ion()
fig, ax = plt.subplots(figsize=(12, 5))
img = ax.imshow(
    np.zeros((N_SUBCARRIERS, WINDOW_SIZE)),
    aspect='auto', cmap='viridis',
    vmin=0, vmax=50, origin='lower'
)
ax.set_title('Live CSI Heatmap — Hand zwischen die Boards halten!')
ax.set_xlabel('Zeit (Pakete)')
ax.set_ylabel('Subcarrier')
plt.colorbar(img, ax=ax, label='Amplitude')

# ============ LIVE LOOP ============
packet_count = 0
try:
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line.startswith('CSI_DATA'):
            continue

        amplitude = parse_csi_line(line)
        if amplitude is None:
            continue

        # Auf feste Länge bringen (für Heatmap)
        if len(amplitude) >= N_SUBCARRIERS:
            amplitude = amplitude[:N_SUBCARRIERS]
        else:
            amplitude = np.pad(amplitude, (0, N_SUBCARRIERS - len(amplitude)))

        amplitude_buffer.append(amplitude)
        packet_count += 1

        # Heatmap nur alle 10 Pakete aktualisieren (sonst zu langsam)
        if packet_count % 10 == 0 and len(amplitude_buffer) > 5:
            matrix = np.array(amplitude_buffer).T   # (Subcarrier × Zeit)
            img.set_data(matrix)
            img.set_clim(vmin=0, vmax=max(50, matrix.max()))

            # Padding rechts, wenn Buffer noch nicht voll
            if matrix.shape[1] < WINDOW_SIZE:
                padded = np.zeros((N_SUBCARRIERS, WINDOW_SIZE))
                padded[:, -matrix.shape[1]:] = matrix
                img.set_data(padded)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

except KeyboardInterrupt:
    print(f"\nGesamt empfangen: {packet_count} CSI-Pakete")
finally:
    ser.close()
    plt.ioff()
    plt.show()