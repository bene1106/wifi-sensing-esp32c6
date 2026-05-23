# Ã°Å¸â€œÂ¡ WiFi-Sensing mit ESP32-C6: AktivitÃƒÂ¤ts-Erkennung



> Passive Erkennung menschlicher AktivitÃƒÂ¤ten allein durch WiFi-Channel-State-Information (CSI) Ã¢â‚¬â€ ohne Kamera, ohne Mikrofon, ohne Wearables.



![Confusion Matrix](confusion_matrix_activity.png)



**Live-Demo erkennt 3 AktivitÃƒÂ¤ten am Schreibtisch:**

- Ã°Å¸Â§Ëœ **STILL** Ã¢â‚¬â€ HÃƒÂ¤nde im SchoÃƒÅ¸, nur Atmen

- Ã¢Å’Â¨Ã¯Â¸Â **TYPING** Ã¢â‚¬â€ Aktives Tippen auf der Tastatur

- Ã°Å¸â€˜â€¹ **WAVING** Ã¢â‚¬â€ Hand winkt ÃƒÂ¼ber der Tastatur



**Genauigkeit: 97.5 % (5-Fold Cross-Validation)**



---



## Ã°Å¸Å½Â¯ Was ist WiFi-Sensing?



Funkwellen werden von Menschen, MÃƒÂ¶beln und WÃƒÂ¤nden reflektiert. Die **Channel State Information (CSI)** beschreibt, wie sich das Signal pro WiFi-Subcarrier verÃƒÂ¤ndert Ã¢â‚¬â€ sie ist im Prinzip ein "Funk-Fingerabdruck" des Raums.



Wenn eine Person sich bewegt, ÃƒÂ¤ndert sich diese Signatur charakteristisch. Mit Machine Learning lassen sich verschiedene Bewegungsmuster trennen.



**Anwendungsbereiche (kommerziell):** Origin Wireless, Cognitive Systems, Aerial Technologies Ã¢â‚¬â€ PrÃƒÂ¤senz-Erkennung, Sturzdetektion, Atemfrequenz-Monitoring.



---



## Ã°Å¸â€Â§ Hardware



| Komponente | Modell | Funktion |

|------------|--------|----------|

| 2Ãƒâ€” ESP32-C6 | Waveshare ESP32-C6-WROOM-1-N8 | Sender (TX) und EmpfÃƒÂ¤nger (RX) |

| 2Ãƒâ€” USB-C-Kabel | Datenkabel | Power + Serial |

| 1Ãƒâ€” Laptop | Windows 10/11 | Daten verarbeiten + Modell |



**Gesamtkosten: \~25 Ã¢â€šÂ¬**



---



## Ã°Å¸Ââ€”Ã¯Â¸Â Architektur



```

Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â     ESP-NOW Pakete (100 Hz)      Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â

Ã¢â€â€š  TX-Board   Ã¢â€â€š  Ã¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€ÂÃ¢â€Â Ã¢â€ â€™   Ã¢â€â€š  RX-Board   Ã¢â€â€š

Ã¢â€â€š  (csi_send) Ã¢â€â€š       Channel 11, 2.4 GHz        Ã¢â€â€š  (csi_recv) Ã¢â€â€š

Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ                                  Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ

                                                        Ã¢â€â€š Serial @ 921600

                                                        Ã¢â€“Â¼

                                  Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â

                                  Ã¢â€â€š   Python (Laptop)            Ã¢â€â€š

                                  Ã¢â€â€š  Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â  Ã¢â€â€š

                                  Ã¢â€â€š  Ã¢â€â€š CSI-Parser Ã¢â€ â€™ Buffer   Ã¢â€â€š  Ã¢â€â€š

                                  Ã¢â€â€š  Ã¢â€â€š Feature Extraction    Ã¢â€â€š  Ã¢â€â€š

                                  Ã¢â€â€š  Ã¢â€â€š RandomForest Modell   Ã¢â€â€š  Ã¢â€â€š

                                  Ã¢â€â€š  Ã¢â€â€š Flask + Browser       Ã¢â€â€š  Ã¢â€â€š

                                  Ã¢â€â€š  Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ  Ã¢â€â€š

                                  Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ

```



---



## Ã°Å¸â€œâ€š Projektstruktur



```

.

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ 1_record_csi.py             # CSI-Daten aufzeichnen mit Label

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ 2_diagnose.py               # Statistik pro Aufnahme prÃƒÂ¼fen

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ 3_train_activity.py         # ML-Modell trainieren (Random Forest)

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ 4_live_activity.py          # Konsolen-Live-Inferenz

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ 5_dashboard_backend.py      # Flask-Server

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ 5_dashboard_frontend.html   # Browser-Dashboard

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ live_heatmap.py             # Live-Heatmap zum Debuggen

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ analyze_log.py              # Log-Files auswerten

Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ rf_model_activity.pkl       # Trainiertes Modell

Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ requirements.txt

```



---



## Ã°Å¸Å¡â‚¬ Setup (Windows)



### 1. ESP32-C6 vorbereiten



Diese Anleitung setzt voraus, dass [ESP-IDF v5.4](https://docs.espressif.com/projects/esp-idf/en/v5.4/esp32c6/get-started/) installiert ist.



**TX-Board flashen:**

```powershell

git clone https://github.com/espressif/esp-csi.git

cd esp-csi/examples/get-started/csi_send

idf.py set-target esp32c6

idf.py -p COM3 flash

```



**RX-Board flashen:**

```powershell

cd ../csi_recv

idf.py set-target esp32c6

idf.py -p COM5 flash

```



> COM-Ports unter Windows mit GerÃƒÂ¤te-Manager prÃƒÂ¼fen. Treiber: [CH343SER](https://www.wch-ic.com/downloads/CH343SER_EXE.html).



### 2. Python-Umgebung



```powershell

python -m venv venv

.\\venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

```



### 3. Daten aufnehmen



```powershell

python 1_record_csi.py still 120     # 2 Min still sitzen

python 1_record_csi.py typing 120    # 2 Min tippen

python 1_record_csi.py waving 120    # 2 Min winken

```



### 4. Modell trainieren



```powershell

python 3_train_activity.py

```



Erzeugt `rf_model_activity.pkl` und `confusion_matrix_activity.png`.



### 5. Live-Dashboard starten



```powershell

python 5_dashboard_backend.py

```



Im Browser ÃƒÂ¶ffnen: **http://localhost:5000**



Vom Handy (gleiches WLAN): `http://<PC-IP>:5000`



---



## Ã°Å¸â€œÅ  Methodik



### Feature-Extraktion



Pro Sliding-Window (90 Pakete = 1 Sek) werden 320 Features pro Sample extrahiert:



- **Std pro Subcarrier** Ã¢â‚¬â€ zeitliche Variation

- **Mean Absolute Difference** Ã¢â‚¬â€ Ãƒâ€žnderungsrate

- **Std der Differenzen** Ã¢â‚¬â€ VariabilitÃƒÂ¤t der Ãƒâ€žnderungen

- **Perzentil-Range (P90 Ã¢Ë†â€™ P10)** Ã¢â‚¬â€ robuster Range

- **Mean Absolute Beschleunigung** Ã¢â‚¬â€ 2. Ableitung



Diese Features sind **kalibrationsunabhÃƒÂ¤ngig** Ã¢â‚¬â€ absolute Amplitudenpegel werden bewusst nicht verwendet, da der ESP32 die AGC laufend anpasst.



### Validierung



**Time-Based 5-Fold Cross-Validation** statt zufÃƒÂ¤lligem Split. Innerhalb jeder Klasse werden die Windows zeitlich in 5 BlÃƒÂ¶cke geteilt Ã¢â‚¬â€ so wird verhindert, dass zeitnah aufeinanderfolgende Windows in beide Sets gelangen (Data Leakage).



### Ergebnisse



| Klasse | Precision | Recall | F1 |

|--------|-----------|--------|-----|

| still  | 1.00 | 1.00 | 1.00 |

| typing | 0.95 | 0.98 | 0.96 |

| waving | 0.98 | 0.95 | 0.96 |



**Gesamtgenauigkeit: 97.54 % (Ã‚Â± 1.86 %)**



---



## Ã¢Å¡Â Ã¯Â¸Â Bekannte Limitierungen



1\. **Inter-Session-Drift** Ã¢â‚¬â€ Der WiFi-Kanal ÃƒÂ¤ndert sich ÃƒÂ¼ber Stunden hinweg (AGC-Rekalibrierung, andere GerÃƒÂ¤te, Multipath). Ein Modell, das heute trainiert wurde, funktioniert morgen mÃƒÂ¶glicherweise nicht mehr. LÃƒÂ¶sung: **On-Site-Retraining** vor jedem Demo-Session.



2\. **Setup-SensitivitÃƒÂ¤t** Ã¢â‚¬â€ Werden die Boards verschoben, muss neu trainiert werden.



3\. **Klassen-Trennbarkeit** Ã¢â‚¬â€ `still` vs. `typing` haben sehr ÃƒÂ¤hnliche TempVar; das Modell unterscheidet sie durch subtile Subcarrier-spezifische Muster. In anderen Umgebungen evtl. schwieriger.



---



## Ã°Å¸â€Â¬ Wissenschaftlicher Kontext



Inspiriert durch die IEEE 802.11bf Standardisierung (WiFi-Sensing) und akademische Arbeiten zu CSI-basierter AktivitÃƒÂ¤tserkennung (z.B. *EI* von Microsoft Research, *FALL-Sense*, *WiCount*).



---



## Ã°Å¸â€œÂ Lizenz



MIT



---



## Ã°Å¸â„¢Â Credits



- **esp-csi** von [Espressif](https://github.com/espressif/esp-csi) Ã¢â‚¬â€ CSI-Firmware

- ESP32-C6-Hardware: [Waveshare](https://www.waveshare.com/wiki/ESP32-C6-WROOM-1)

