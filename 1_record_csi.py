"""
CSI Datensammlung
Sammelt CSI-Daten vom ESP32-C6 RX-Board und speichert sie als CSV mit Label.

Aufruf:
  python record_csi.py <label> <dauer_sekunden>
  
Beispiel:
  python record_csi.py empty 300   # 5 Min "leer"
  python record_csi.py motion 300  # 5 Min "Bewegung"
  python record_csi.py sitting 300 # 5 Min "sitzen"
"""

import serial
import sys
import time
import re
import os
from datetime import datetime

# ============ KONFIGURATION ============
SERIAL_PORT = 'COM5'
BAUDRATE = 921600
DATA_DIR = 'data'

# ============ CLI-ARGUMENTE ============
if len(sys.argv) != 3:
    print("Usage: python record_csi.py <label> <duration_seconds>")
    print("Labels: empty | motion | sitting")
    sys.exit(1)

label = sys.argv[1]
duration = int(sys.argv[2])

if label not in ['empty', 'motion', 'sitting', 'still', 'typing', 'waving']:
    print(f"⚠️  Warnung: unbekanntes Label '{label}'. Trotzdem fortfahren? (Strg+C zum Abbrechen)")
    time.sleep(2)

# ============ DATEN-ORDNER ============
os.makedirs(DATA_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"{DATA_DIR}/{label}_{timestamp}.csv"

# ============ COUNTDOWN ============
print(f"\n{'='*60}")
print(f"📊 Aufnahme-Konfiguration")
print(f"{'='*60}")
print(f"  Label:     {label}")
print(f"  Dauer:     {duration} Sekunden")
print(f"  Datei:     {filename}")
print(f"  Port:      {SERIAL_PORT} @ {BAUDRATE} Baud")
print(f"{'='*60}\n")

print("Aufnahme startet in:")
for i in range(5, 0, -1):
    print(f"  {i} ...")
    time.sleep(1)
print("🔴 AUFNAHME LÄUFT\n")

# ============ SERIAL VERBINDUNG ============
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)

# ============ AUFNAHME ============
start_time = time.time()
packet_count = 0
last_print = start_time

with open(filename, 'w', encoding='utf-8') as f:
    # Header schreiben
    f.write("timestamp,label,raw_csi_line\n")

    try:
        while time.time() - start_time < duration:
            line = ser.readline().decode('utf-8', errors='ignore').strip()

            if line.startswith('CSI_DATA'):
                ts = time.time()
                # CSV-sicher: doppelte Anführungszeichen escapen
                safe_line = line.replace('"', '""')
                f.write(f'{ts},{label},"{safe_line}"\n')
                packet_count += 1

            # Status alle 5 Sek
            now = time.time()
            if now - last_print > 5:
                elapsed = now - start_time
                remaining = duration - elapsed
                rate = packet_count / elapsed if elapsed > 0 else 0
                print(f"  ⏱️  {elapsed:.0f}s / {duration}s | {packet_count} Pakete | {rate:.1f} Pkt/s | noch {remaining:.0f}s")
                last_print = now
                f.flush()  # Wichtig: regelmäßig auf Disk schreiben

    except KeyboardInterrupt:
        print("\n⚠️  Abbruch durch Benutzer.")

    finally:
        ser.close()
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ Aufnahme abgeschlossen")
        print(f"{'='*60}")
        print(f"  Pakete gesamt:  {packet_count}")
        print(f"  Dauer:          {elapsed:.1f}s")
        print(f"  Datenrate:      {packet_count/elapsed:.1f} Pakete/s")
        print(f"  Dateigröße:     {os.path.getsize(filename)/1024:.1f} KB")
        print(f"  Datei:          {filename}")
        print(f"{'='*60}\n")