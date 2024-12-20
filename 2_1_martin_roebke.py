"""Computational Physics Aufgabe 2.1,  Autor: Martin Roebke 24.04.16
Aufgabe Elementare numerische Methoden I.

Bestimmung der Ableitung einer Funktion f(x) an einer Stelle
numerisch mit verschiedener Praezision, 64 bit Gleitkomma-Arithmetik.
Drei Methoden:
  Vorwaertsdifferenz.
  Zentraldifferenz.
  Extrapolierte Differenz.
   AE(x) = 1/3h( 8 (f(x+h/4)-f(x-h/4)) - (f(x+h/2)-f(x-h/2)))
"""

from typing import Callable
import numpy as np  # Arrays, Mathe etc
import matplotlib.pyplot as plt  # Plotten


def relativ_Fehler(wert: float, erwartet: float):
    """Relativer Fehler von 'wert' bezueglich einer Referenz 'erwartet'."""
    return abs((wert - erwartet) / erwartet)


def func(x: float = 0, abl: int = 0, string: bool = False):
    """Reelle Funktion mit Argument x, Grundlage zu Berechnung.
    abl:
        0 Funktionswert.
        1 Erste Analytische Ableitung.
    string=True : Rueckgabe String f(x)
    """
    if string:
        # gibt Beschreibung f(x) als String zurueck
        return r"$arctan(x^3)$"
    if abl == 0:
        # Funktion f(x)
        return np.arctan(x**3)
    if abl == 1:
        # Erste analytische Ableitung der Funktion f(x)
        return 3.0 * x * x / (x**6 + 1)

    raise ValueError("'func' konnte Parameter nicht erkennen.")


def vorwaertsDiff(f: Callable[[float], float], x: float, h: np.ndarray[float]):
    """return Ableitung der Funktion f an Stelle x nach Vorwaertsdifferenz mit Parameter h"""
    return 1.0 / h * (f(x + h) - f(x))


def zentralDiff(f: Callable[[float], float], x: float, h: np.ndarray[float]):
    """return Ableitung der Funktion f an Stelle x nach Zentraldifferenz mit Parameter h"""
    return 1.0 / h * (f(x + h / 2) - f(x - h / 2))


def extrapolDiff(f: Callable[[float], float], x: float, h: np.ndarray[float]):
    """return Ableitung der Funktion f an Stelle x
    nach '1/3h *( 8*(f(x + h/4)-f(x - h/4)) - (f(x + h/2) - f(x - h/2)))'
    """
    return (
        1
        / (3 * h)
        * (8 * (f(x + h / 4) - f(x - h / 4)) - (f(x + h / 2) - f(x - h / 2)))
    )


def main():
    """Mainfunktion: Numerische Methoden I (Ableitung).
    Zur Eingabe der Parameter und Darstellung der Berechnung.
    """
    # Parameter
    f = func  # genutzte Funktion
    auswertung_x: float = 1 / 3  # x-Position
    h: np.ndarray[float] = 10.0 ** np.linspace(-10, 0, 100)  # Array h Log-10

    # Anfangstext Konsole
    print(__doc__)
    print(">>Keine Benutzer-Interaktion erforderlich.<<\n")
    print("Auswertung an: {}\n".format(auswertung_x))

    # Analytische Ableitung
    ableitung_wert = f(auswertung_x, abl=1)
    # Array VorwaertsDiff
    VorwaertsDiff = vorwaertsDiff(f, auswertung_x, h)
    # Array ZentralDiff
    ZentralDiff = zentralDiff(f, auswertung_x, h)
    # Array ExtrapolDiff
    ExtrapolDiff = extrapolDiff(f, auswertung_x, h)

    # Relative Fehler
    relFehlerVD = relativ_Fehler(VorwaertsDiff, ableitung_wert)
    relFehlerZD = relativ_Fehler(ZentralDiff, ableitung_wert)
    relFehlerED = relativ_Fehler(ExtrapolDiff, ableitung_wert)

    fig = plt.figure(figsize=(12, 10))  # Figure
    fig.subplots_adjust(0.08, 0.07, 0.95, 0.91)  # Begrenzung [l,b,r,t]

    # Erstelle Subplot Zeichenfenster
    plt.subplot(111, xscale="log", yscale="log")
    stitle = "Numerische Ableitung f(x)=" + f(string=True)
    # Plot-Titel
    plt.suptitle(stitle, fontsize="x-large")
    plt.title("Doppeltlogaritmisch")
    # Achsenbeschriftung
    plt.xlabel("Parameter h", fontsize="large")
    plt.ylabel("Relativer Fehler", fontsize="large")

    plt.plot(h, relFehlerVD, "bo", label="Vorwaertsdifferenz", ms=4.5, mew=0)
    plt.plot(h, relFehlerZD, "go", label="Zentraldifferenz", ms=4.5, mew=0)
    plt.plot(h, relFehlerED, "ro", label="Extrapol.-Differenz", ms=4.5, mew=0)

    # Erwartetes Skalierungsverhalten h^alpha
    # Linear
    h_l = h[6:].copy()
    plt.plot(h_l, h_l * 3.54 * 2, "b", label=r"$h^{\ 1}$", lw=1)
    # Quadratisch
    h_sq = h[37:].copy()
    plt.plot(h_sq, h_sq * h_sq * 0.66 * 2, "g", label=r"$h^{\ 2}$", lw=1)
    # h^4
    h_quad = h[65:].copy()
    plt.plot(h_quad, h_quad**4.0 * 0.032 * 2, "r", label=r"$h^{\ 4}$", lw=1)
    # Rundung invers zu h
    plt.plot(h, (h**-1) * (10**-17) * 3, "k", label=r"$h^{- 1}$", lw=3)

    legend = plt.legend(loc="upper left", shadow=True)

    plt.show()  # Zeigen des Fensters


# -------------Main Programm---------------
if __name__ == "__main__":
    main()

"""Kommentar:
Beispiel der Tiefe Gleitkomma-Diskretisierung, Mantisse 52 bit:
0.1 -> 0.10000000000000000555...
a) Bei kleinen Werten von h wird der Diskretisierungsfehler entsprechend
    der Skalierung klein, da jedes Verfahren OHNE Rundungsfehler
    gegen die Analytische Ableitung konvergiert.
   Am Beispiel der Vorwaertsdifferenz:
    Nimmt jedoch (f(x+h)-f(x)) sehr kleine Werte an (rund h*f'(x)),
    faellt in den letzten Stellen der Mantisse der Rundungsfehler
    staerker ins Gewicht. In der Approximation der Geraden f'(x)
    weicht (f(x+h)-f(x)) nach Rundung maximal um je ein X ab, also insgesamt
    um 2X. Dividiert durch h ergibt:
    Rundungsfehler entspricht 2X/h ~ 1/h.

b) Numerischer Fehler:
    Diskretisierungsfehler
    + Rundungsfehler -> Dieser liegt hier in Groessenordnung 10^-16
        bedingt durch 64 bit Gleitkomma-Arithmetik
    Sind beide in etwa gleich gross, ist die Summe beider
    - der Gesamtfehler - minimal, und h optimal:
    m ~ optimaler Parameter h
    10^-16 / m = m => Demnach ist m = 10^-8.
    Ein relativer Fehler von 10^-8 zu erwarten.

Beste Groessenordnungen abgelesen:
    Vorwaerts-Diff: h ~ 10^-8 ; relativer Fehler ~ 10^-8
    Zentral-Diff:   h ~ 10^-5 ; relativer Fehler ~ 10^-11
    Extrapol. Diff: h ~ 10^-3 ; relativer Fehler ~ 10^-13
"""
