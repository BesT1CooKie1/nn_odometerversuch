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
Starting the neural network process...
Training Progress:  10%|█         | 10/100 [01:10<10:13,  6.82s/it]
Epoch [10/100], Loss: 0.1215
Training Progress:  20%|██        | 20/100 [02:16<08:52,  6.65s/it]
Epoch [20/100], Loss: 0.0732
Training Progress:  30%|███       | 30/100 [03:24<07:53,  6.76s/it]
Epoch [30/100], Loss: 0.0525
Training Progress:  40%|████      | 40/100 [04:33<06:59,  6.99s/it]
Epoch [40/100], Loss: 0.0387
Training Progress:  50%|█████     | 50/100 [05:41<05:34,  6.68s/it]
Epoch [50/100], Loss: 0.0308
Training Progress:  60%|██████    | 60/100 [06:47<04:25,  6.65s/it]
Epoch [60/100], Loss: 0.0264
Training Progress:  70%|███████   | 70/100 [07:54<03:23,  6.78s/it]
Epoch [70/100], Loss: 0.0240
Training Progress:  80%|████████  | 80/100 [08:59<02:09,  6.46s/it]
Epoch [80/100], Loss: 0.0227
Training Progress:  90%|█████████ | 90/100 [10:05<01:05,  6.54s/it]
Epoch [90/100], Loss: 0.0222
Training Progress: 100%|██████████| 100/100 [11:09<00:00,  6.69s/it]

Epoch [100/100], Loss: 0.0222
Test Loss: 0.0007
Additional Stress (delta_sigma):
  Tatsächlich: 0.00, Vorhergesagt: 0.00
MSE: 0.0007
MAE: 0.0211
R2: 0.0000
```
Output: ![Ödometerversuch](example_output.png)