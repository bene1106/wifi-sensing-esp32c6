"""
Diagnose: Sind die Daten überhaupt diskriminativ?
"""

import os
import re
import glob
import numpy as np
import pandas as pd

DATA_DIR = 'data'
N_SUBCARRIERS = 64

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

files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
print(f"\n{'Datei':<45s} {'Label':<10s} {'Pakete':>8s} {'Mean':>8s} {'Std':>8s} {'TempVar':>8s}")
print("-" * 95)

for f in files:
    df = pd.read_csv(f)
    label = df['label'].iloc[0]
    amps = []
    for raw in df['raw_csi_line']:
        a = parse_csi_line(raw)
        if a is not None and len(a) >= N_SUBCARRIERS:
            amps.append(a[:N_SUBCARRIERS])
    arr = np.array(amps)
    
    mean_amp = arr.mean()
    std_amp = arr.std()
    temporal_var = arr.std(axis=0).mean()
    
    name = os.path.basename(f)
    print(f"{name:<45s} {label:<10s} {len(arr):>8d} {mean_amp:>8.2f} {std_amp:>8.2f} {temporal_var:>8.3f}")

print("\n💡 Wenn 'TempVar' bei motion-Files DEUTLICH höher als bei empty-Files:")
print("   → Klassen sind unterscheidbar.")
print("   Falls Werte ähnlich: Daten sind das Problem.")