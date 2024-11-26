
# Ödometerversuch mit KI-Unterstützung

Dieses Projekt untersucht die Anwendung von KI-Modellen zur Vorhersage von bodenmechanischen Parametern aus dem Ödometerversuch. Es kombiniert klassische geotechnische Berechnungen mit maschinellem Lernen, um Prozesse zu automatisieren und Muster in Daten zu erkennen.

## Projektübersicht
Der **Ödometerversuch** wird genutzt, um das Setzungs- und Konsolidationsverhalten von Böden zu analysieren. Ziel dieses Projekts ist es, eine KI zu entwickeln, die auf Basis von Versuchsdaten Parameter wie Porenzahlen oder Steifemodule vorhersagen kann.

### Features:
- Generierung von Testdaten mit realistischen Wertebereichen für Input-Parameter.
- Berechnung klassischer bodenmechanischer Parameter aus dem Ödometerversuch.
- Vorbereitung für die Integration von KI-Modellen zur Vorhersage von Zielwerten.

---

## Verzeichnisstruktur
- **`main.py`**: Hauptskript zur Berechnung und Simulation von Testdaten.
- **`handleTestValues.py`**: Generierung realistischer Input-Parameter für den Ödometerversuch.
- **`image.png`**: Formeln und theoretische Grundlage des Projekts.

---

## Input- und Output-Parameter

### Input-Parameter (Einflussgrößen):
1. **Kompressionsbeiwert (Cc):** Verdichtungseigenschaften bei Belastung (0,1 ≤ Cc ≤ 0,5).
2. **Schwellbeiwert (Cs):** Elastizitätseigenschaften bei Entlastung (0,01 ≤ Cs ≤ 0,1).
3. **Anfangs-Porenzahl (e0):** Hohlraumdichte des Bodens (0,5 ≤ e0 ≤ 1,5).
4. **Effektive Anfangsspannung (σ0'):** Anfangsdruck im Boden (50 ≤ σ0' ≤ 200 kN/m²).
5. **Zusatzspannung (Δσ):** Aufgebrachte Spannung (10 ≤ Δσ ≤ 100 kN/m²).

### Output-Parameter (Berechnungen):
1. **Mittlere Spannung (σmittel):** σmittel = σ0' + Δσ/2.
2. **Steifemodul bei Belastung (Es, Belastung):** (1 + e0) / Cc * σmittel.
3. **Steifemodul bei Entlastung (Es, Entlastung):** Analog mit Cs.
4. **Porenzahlen (e, Belastung und e, Entlastung):** Berechnet über logarithmische Beziehungen.

---

## Installation

1. **Repository klonen:**
   ```bash
   git clone https://github.com/DeinBenutzername/Oedometer-KI.git
   cd Oedometer-KI
   ```

2. **Abhängigkeiten installieren:**
   Stelle sicher, dass Python 3.7+ installiert ist, und installiere die benötigten Bibliotheken:
   ```bash
   pip install -r requirements.txt
   ```

3. **Skripte ausführen:**
   Um Testwerte zu generieren und Berechnungen durchzuführen:
   ```bash
   python main.py
   ```

---

## Nächste Schritte
- Integration eines maschinellen Lernmodells zur Vorhersage von Output-Parametern basierend auf realen Versuchsdaten.
- Validierung des Algorithmus mit echten Testwerten aus Ödometerversuchen.
- Optimierung der Parameterbereiche und Verbesserung der Datenqualität.

---

## Lizenz
Dieses Projekt steht unter der [MIT-Lizenz](LICENSE).

---

## Kontakt
Wenn du Fragen oder Anmerkungen hast, melde dich gerne bei [deine E-Mail-Adresse] oder erstelle ein Issue im Repository.
