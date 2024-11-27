# Ödometerversuch mit KI-Unterstützung

Dieses Projekt untersucht die Anwendung von KI-Modellen zur Vorhersage von bodenmechanischen Parametern aus dem Ödometerversuch. Ziel ist es, Prozesse zu automatisieren und präzisere Vorhersagen über das Verhalten von Böden unter Belastung zu treffen.

## Projektübersicht

Der **Ödometerversuch** wird verwendet, um das Setzungs- und Konsolidationsverhalten von Böden zu analysieren. Dieses Projekt kombiniert traditionelle geotechnische Modelle mit einem neuronalen Netzwerk, das auf synthetischen Testdaten trainiert wird.

### Features
- **Automatische Generierung von Testdaten**: Werte basieren auf typischen geotechnischen Spannungs- und Deformationsbereichen.
- **Neural Network Training**: Konfigurierbare Modelle für die Vorhersage spezifischer geotechnischer Parameter.
- **Evaluation und Visualisierung**: Statistiken und Plots, um die Modellleistung zu bewerten.

---

## Verzeichnisstruktur

- **`main.py`**: Hauptskript für Datengenerierung und Modelltraining.
- **`handleTestValues.py`**: Generierung realistischer geotechnischer Parameter.
- **`handleNeuralNetwork.py`**: Definition, Training und Evaluation des neuronalen Netzwerks.
- **`handleDataframes.py`**: Datenmanipulation und Speicher-/Lade-Operationen.
- **`classProblemObjects.py`**: Definiert die Problemstruktur für Böden.

---

## Input- und Output-Parameter

### Input-Parameter:
1. **Compression Index (Cc):** Maß für Verdichtbarkeit (0.1 ≤ Cc ≤ 0.5).
2. **Swelling Index (Cs):** Maß für elastische Erholung (0.01 ≤ Cs ≤ 0.1).
3. **Initial Stress (σ₀):** Anfangsspannung im Boden (50 ≤ σ₀ ≤ 200 kN/m²).
4. **Strain Increment (Δε):** Deformationsinkrement (0.0001 ≤ Δε ≤ 0.01).

### Output-Parameter:
- **Additional Stress (Δσ):** Spannungsänderung, die aus dem Deformationsinkrement resultiert.

---

## Konfigurationsoptionen

Die Datei `config/config.ini` erlaubt die Anpassung der folgenden Parameter:
- **Netzwerkarchitektur**: Hidden Layer, Aktivierungsfunktionen und Dropout-Raten.
- **Training**: Lernrate, Epochen, Scheduler-Typ und Early Stopping.
- **Datenverarbeitung**: Normalisierung, Datenaugmentation und Split-Verhältnisse.

---

## Nächste Schritte
- **Integration realer Daten**: Validierung des Modells mit experimentellen Werten.
- **Optimierung des Modells**: Verbesserung der Vorhersagegenauigkeit durch Hyperparameter-Tuning.
- **Erweiterte Visualisierung**: Darstellung von Spannungs-Dehnungs-Kurven und Modellprognosen.

---

## Beispielausgabe

```plaintext
Generating new soil properties because the number of test-entries do not match...
Generating soil properties: 100%|██████████| 1000000/1000000 [00:30<00:00, 32689.75it/s]
Data saved to ./data/soil_properties.h5
Time taken: 35.83 seconds
Starting the neural network process...
Training Progress:  10%|█         | 10/100 [01:16<10:03,  6.71s/it]
Epoch [10/100], Loss: 0.0767
Training Progress:  20%|██        | 20/100 [02:22<08:49,  6.61s/it]
Epoch [20/100], Loss: 0.0479
Training Progress:  30%|███       | 30/100 [03:28<07:32,  6.47s/it]
Epoch [30/100], Loss: 0.0333
Training Progress:  40%|████      | 40/100 [04:33<06:30,  6.50s/it]
Epoch [40/100], Loss: 0.0250
Training Progress:  50%|█████     | 50/100 [05:40<05:35,  6.72s/it]
Epoch [50/100], Loss: 0.0203
Training Progress:  60%|██████    | 60/100 [06:47<04:32,  6.81s/it]
Epoch [60/100], Loss: 0.0176
Training Progress:  70%|███████   | 70/100 [07:55<03:26,  6.87s/it]
Epoch [70/100], Loss: 0.0161
Training Progress:  80%|████████  | 80/100 [09:00<02:11,  6.57s/it]
Epoch [80/100], Loss: 0.0154
Training Progress:  90%|█████████ | 90/100 [10:09<01:07,  6.75s/it]
Epoch [90/100], Loss: 0.0150
Training Progress: 100%|██████████| 100/100 [11:15<00:00,  6.76s/it]

Epoch [100/100], Loss: 0.0150
Test Loss: 0.0020
Additional Stress (delta_sigma):
  Tatsächlich: 0.00, Vorhergesagt: -0.06
MSE: 0.0020
MAE: 0.0330
R2: 0.0000
```
Output: ![Ödometerversuch](example_output.png)