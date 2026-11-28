# Übergabe an die lokale Session

Hi, ich bin Kane. Ich arbeite am Projekt "Beyond Exposure" (Data Science for Linguists, Uni Tübingen, 6 ECTS). In einer Cloud-Session habe ich mit Claude den Stand geklärt. Hier die Zusammenfassung. Bitte sprich Deutsch mit mir, sachlich und normal, ohne Gedankenstriche.

## Wie ich arbeiten will

- Ich will das Projekt in 1 bis 2 Wochen abschließen.
- Ich will verstehen, was abgegeben wird. Erkläre mir neue Begriffe einfach, bevor du sie benutzt.
- Code nicht weiter aufblähen. Keine zusätzlichen Absicherungen oder Varianten, die nicht im Proposal stehen.
- Ich schreibe die Texte im Report selbst, du gibst Gliederung und Feedback.
- Bisher hat viel Codex gemacht. Ich will weg vom "fremden Code abnicken" hin zu "verstehen und selbst entscheiden".

## Entscheidungen (stehen fest, nicht neu diskutieren)

- **Weg B = das abgegebene Proposal** (`DatSciLingProjectProposal.pdf`): Zwei-Stufen-Verfahren. Stufe 1 pro Lemma ein binomiales GLM (s05) für Retentionsniveau und Decay-Steigung. Stufe 2 WLS mit Wortmerkmalen.
- Kein gemeinsames Ereignismodell (Weg C), keine Hopman-nahe Neuaggregation (Weg A). Beides widerspricht §4.2/§4.3 des Proposals.
- Die niedrige Zuverlässigkeit der Steigungen (ca. 12,5 %, s06) ist im Proposal schon vorgesehen (§5.3 Variance check, §7 "methodology becomes the finding"). H2 wird trotzdem gerechnet, aber mit diesem Hinweis berichtet.
- H1 im Report nicht als "Replikation, die das Verfahren bestätigt" darstellen, sondern als Vergleich mit Hopman et al. (andere Zielgröße, andere Analyseeinheit).
- H4 hängt vom Referenztag ab (Korrelation wechselt das Vorzeichen zwischen 2 und 14 Tagen). Als Grafik über mehrere Referenztage zeigen, deskriptiv, keine Hauptaussage.
- Mehrfachtests mit Benjamini-Hochberg, wie im Proposal.
- Alle optionalen Punkte aus §5.6 weglassen und als future work erwähnen.
- s04b (Übersetzungs-Prüfbogen) nicht als großes Script ausbauen. Stattdessen prüfe ich ca. 20 Übersetzungen von Hand und schreibe einen Satz dazu.
- s10 nicht um weitere Varianten erweitern.

## Was noch fehlt (Pflicht laut Proposal)

1. **s09 fertig:** Baseline (Übung + Wortart) vs. volles Modell (+ 12 Merkmale) für Niveau und Steigung. BIC, adjusted R², Cross-Validation nach Wort, linear (und logistisch, falls einfach machbar), WLS mit inverser Varianz. Welcher Merkmalsblock trägt am meisten bei (H3).
2. **Positive control** prüfen (Übersetzungsdistanz, semantische Dichte beim Niveau), falls noch nicht in s08.
3. **Bootstrap nach Lemma + Benjamini-Hochberg** für die Koeffizienten.
4. **Sensitivität:** s03 bis s05 mit 6 h statt 1 Tag Mindestpause (und ggf. anderer Mindestanzahl) laufen lassen, Kernzahlen vergleichen.
5. **Naiver vs. gefitteter Schätzer:** eine Grafik.
6. **H4-Grafik** über mehrere Referenztage.
7. Verteilung der Pausen berichten, Anzahl extrapolierter Halbwertszeiten, Korrelationsmatrix der Prädiktoren.
8. **Notebook als Report:** lädt die Script-Ausgaben, zeigt Grafiken, Text von mir. Gliederung: Einleitung, Daten, Methode, Ergebnisse, Diskussion/Grenzen, Abweichungen vom Proposal (Tabelle), Arbeitsteilung + KI-Nutzung.
9. README, requirements, pushen.

Offener Punkt zum Prüfen: Das Proposal (§5.4) verspricht als Kontrolle die Kursposition (Median von `history_seen` beim ersten Kontakt). In s07 sehe ich diese Variable nicht. Prüfen, ob s03 sie berechnet, sonst nachholen oder als Abweichung begründen.

## Der menschliche Teil

- Abschnitt "Lernt die Wörter kennen": 4 bis 6 echte Wörter (am besten/schlechtesten behalten, am schnellsten vergessen, eins gegen die Erwartung) mit selbstgezeichneten Steckbriefkarten.
- Eine gezeichnete Skizze "Niveau vs. Steigung".
- Ein ehrlicher Abschnitt in eigener Stimme: was nicht geklappt hat, was wir gelernt haben.
- Optional danach: eine Webseite mit den Ergebnissen, inspiriert von Duolingo (rund, freundlich), aber ohne deren Markenzeichen, mit eigenem gezeichneten Maskottchen.

## Erster Schritt in der lokalen Session

Lies `scripts/s08_h1.py`, `scripts/s09_other_analyses.py`, `scripts/s10_*.py` und `notes/`. Sag mir, welche Punkte der Liste oben schon erledigt sind. Dann machen wir s09 fertig, kompakt und mit einer Erklärung pro Block.
