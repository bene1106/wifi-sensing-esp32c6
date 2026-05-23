"""
Analysiert das Live-Log und zeigt die Vorhersagen über Zeit.
"""
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Neuestes Log nehmen
logs = sorted(glob.glob('live_log_*.csv'))
if not logs:
    print("❌ Kein Log gefunden!")
    exit(1)
log_path = logs[-1]
print(f"📁 Analysiere: {log_path}")

df = pd.read_csv(log_path)
print(f"   {len(df)} Vorhersagen über {(df['timestamp'].max() - df['timestamp'].min()):.1f} Sek")

# Statistik
counts = df['prediction'].value_counts()
print(f"\n📊 Verteilung:")
for cls, n in counts.items():
    print(f"   {cls}: {n} ({n/len(df)*100:.1f}%)")

print(f"\n📊 Confidence-Statistik:")
print(f"   conf_absent  - mean: {df['confidence_absent'].mean():.1f}%  max: {df['confidence_absent'].max():.1f}%")
print(f"   conf_present - mean: {df['confidence_present'].mean():.1f}%  min: {df['confidence_present'].min():.1f}%")

# Wo gab es Wechsel?
df['changed'] = df['prediction'] != df['prediction'].shift(1)
changes = df[df['changed']].copy()
print(f"\n📊 Klassen-Wechsel: {len(changes)-1}")
if len(changes) > 1:
    print("   Erste 10 Wechsel:")
    for _, row in changes.head(11).iterrows():
        print(f"   {row['iso_time']}: → {row['prediction']} (conf: {max(row['confidence_absent'], row['confidence_present']):.1f}%)")

# Plot
t = df['timestamp'] - df['timestamp'].iloc[0]
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax[0].plot(t, df['confidence_present'], label='conf present', color='green')
ax[0].plot(t, df['confidence_absent'], label='conf absent', color='blue')
ax[0].set_ylabel('Confidence %'); ax[0].legend(); ax[0].grid()
ax[0].set_title(f'Live Predictions — {os.path.basename(log_path)}')

# Klasse als Step
y_num = (df['prediction'] == 'present').astype(int)
ax[1].plot(t, y_num, drawstyle='steps-post', linewidth=2)
ax[1].set_yticks([0, 1])
ax[1].set_yticklabels(['absent', 'present'])
ax[1].set_xlabel('Zeit (Sek seit Start)'); ax[1].grid()

plt.tight_layout()
plt.savefig('live_log_plot.png', dpi=120)
plt.show()