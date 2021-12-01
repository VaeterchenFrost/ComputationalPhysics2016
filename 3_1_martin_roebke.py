"""Computational Physics Aufgabe 3.1,  Autor: Martin Roebke 01.05.16
"Elementare numerische Methoden II"

Numerische Integrale mit Mittelpunktsregel, Trapezmethode und Simpson-Methode.
Relativer Fehler zu analytischem Ergebnis in Plot.
Zahl N der Teilintervalle, dass Abstand Stuetzstellen [10**-4, 1] abdeckt.
Plot-Darstellung doppellog. mit Legende und analytischem Skalierungsverhalten.
Diskussion in End-Kommentar.
"""

from __future__ import division, print_function # problemlose Ganzzahl-Division
import numpy as np                              # Arrays, Mathe etc
import matplotlib.pyplot as plt                 # Plotten
from functools import partial                   # args Vorbelegung

def IntMPRegel(f, start, ende, n):
    """Berechnet Integral der reellen Funktion 'f' zwischen Stellen
    'start' und 'ende' nach Mittelpunktsregel mit 'n' Segmenten.
    """
    # Array mit Stuetzstellen
    array_stuetz, h = np.linspace(start, ende, n, endpoint=False, retstep=True)
    array_stuetz += 0.5*h
    # Integral berechnen
    ergebnis = h * sum(f(array_stuetz))
    return ergebnis

def IntTRRegel(f, start, ende, n):
    """Berechnet Integral der reellen Funktion 'f' zwischen Stellen
    'start' und 'ende' nach Trapezregel mit 'n' Segmenten.
    """
    # Array mit Stuetzstellen
    array_stuetz, h = np.linspace(start, ende, n+1, retstep=True)
    # Integral berechnen
    ergebnis = h * (sum(f(array_stuetz)) - 0.5*(f(start) + f(ende)))
    return ergebnis

def IntSMRegel(f, start, ende, n):
    """Berechnet Integral der reellen Funktion 'f' zwischen Stellen
    'start' und 'ende' nach Simpsonregel mit 'n' Segmenten.
    """
    # Zwei Arrays mit Stuetzstellen : Ganzzahlige N - Stuetzen
    stuetz_ganz, h = np.linspace(start, ende, n+1, retstep=True)
    stuetz_halb = stuetz_ganz[1::].copy() - 0.5*h     # Mittige Stuetzen
    # Integral berechnen
    summeganzN = sum(f(stuetz_ganz))
    summehalbN = sum(f(stuetz_halb))
    ergebnis = h * (4*summehalbN + 2*summeganzN - f(start) - f(ende)) / 6
    return ergebnis

def relativ_Fehler(wert, erwartet):
    """Relativer Fehler von 'wert' bezueglich einer Referenz 'erwartet'."""
    return abs((wert - erwartet) / erwartet)

def func(x=None, string=False, aufgabe='a'):
    """Reelle Funktion mit Argument x, Grundlage zu Berechnung.
    aufgabe: String Aufgabenstellungen
            'a', 'b', 'c'
    x: Reelle Zahlen (Array)
    string: Bool
        True : String Darstellung 'f'
    """
    if aufgabe == 'a':
        if string:
            # gibt Beschreibung f(x) als String zurueck
            return "np.sin(2x)"
        if x is None:
            # Gebe Wert bestimmtes Integral
            return -0.25
        # Funktion f(x) ueber Array
        return np.sin(2.0 * x)
    if aufgabe == 'b':
        if string:
            # gibt Beschreibung f(x) als String zurueck
            return "np.exp(-100 * x*x)"
        if x is None:
            # Gebe Wert bestimmtes Integral
            return 0.177245385090551602729816748334114518279
        # Funktion f(x) ueber Array
        return np.exp(-100*x*x)
    if aufgabe == 'c':
        if string:
            # gibt Beschreibung f(x) als String zurueck
            return "0.5 * (1.0 + np.sign(x))"
        if x is None:
            # Gebe Wert bestimmtes Integral
            return np.pi/3.
        # Funktion f(x) ueber Array
        return 0.5*(1.0+np.sign(x))

def main():
    """Mainfunktion: Numerische Methoden II (Integration).
    Eingabe der Parameter und Darstellung.
    """

    # --------Parameter--------
    f = partial(func, aufgabe='c')                      # genutzte Funktion
    a = -np.pi/2                                        # Start
    b = np.pi/3                                         # Ende
    analytisch_wert = f()                               # Gespeichertes Ergebnis
    h_wunsch = 10.0 ** np.linspace(-5.0, 0.0, 100)      # grobe Schrittweite

    # Array Anzahl ganzer Stuetzstellen
    aN = np.around((b-a)/h_wunsch)
    aN = np.unique(aN)
    # Array darstellbarer Abstaende ganzer Stuetzstellen
    h = (b-a)/aN

    # Anfangstext Konsole
    print(__doc__)
    print("f(x)=" + f(string=True))
    print("von", a, "bis", b, ".")
    print("\nKeine Benutzer-Interaktion erforderlich.")

    fig = plt.figure(figsize=(11,8))                    # Plot-Initialisierung
    fig.subplots_adjust(top=0.87)                       # Borders
    plt.subplot(111, xscale="log", yscale="log")        # Subplot
    plt.title("Integral numerisch")                     # Plot-Titel
    stitle = "f(x)= " + f(string=True)
    plt.suptitle(stitle, fontsize='x-large')
    plt.xlabel("Parameter h", fontsize='large')         # Achsenbeschriftung
    plt.ylabel("Relativer Fehler", fontsize='large')

    # Arrays fuer relativen Fehler jeder Rechnung aller h
    F_MP = np.zeros(len(h), dtype=np.float)
    F_TR = np.zeros(len(h), dtype=np.float)
    F_SM = np.zeros(len(h), dtype=np.float)

    for i, n in enumerate(aN):       # Schleife fuer Berechnung Relative Fehler
        F_MP[i] = relativ_Fehler(IntMPRegel(f, a, b, n), analytisch_wert)
        F_TR[i] = relativ_Fehler(IntTRRegel(f, a, b, n), analytisch_wert)
        F_SM[i] = relativ_Fehler(IntSMRegel(f, a, b, n), analytisch_wert)

    plt.plot(h, F_MP, "r.", label= "Mittelpunktsregel")
    plt.plot(h, F_TR, "b.", label= "Trapezregel")
    plt.plot(h, F_SM, "g.", label= "Simpsonregel")
    plt.plot(h, h*h*0.025, "k--", label= r"$\sim h^{2}$", lw=2.1)
    h_four = h[:50]
    plt.plot(h_four, h_four**4*10**-3, "k-.", label= r"$\sim h^{4}$", lw=1.6)
    legend = plt.legend(loc='best', numpoints=3)        # Legende
    plt.show()                                          # Zeigen des Fensters

#-------------Main Programm-------------------
if __name__ == "__main__":
    main()

"""Diskussion
->np.sin(2.0*x)
Als glatte Funktion ist MP&TR-Regel in h^2, Simpson in h^4 zu beobachten -
Solange Rundungsfehler nicht groesser sind.

->np.e**(-100*x*x)
Diese Funktion ist eine sehr scharfe Gauss-Glocke, und besitzt sehr steile
Stellen in der Naehe x=0 (wo die Funktion Wert 1 annimmt).
Ausserhalb von ca. [-0.4, 0.4] ist der Funktionswert rund Null.
Sehr gute Konvergenz ab ca h = 0.08.
Dann kann die um den Ursprung stark ansteigende Glocke gut approximiert
werden, u.a. weil die Ableitungen an a und b gleich (Null) sind.

->0.5*(1.0 + np.sign(x))
Diese unstetige Heaviside-Funktion zeigt bei allen Verfahren ungefaehr
lineare Skalierung. Diese Funktion besitzt keine endliche
Ableitung - so werden Konvergenzberechnungen der einzelnen Verfahren in den
Ableitungen hinfaellig. Ausreiss-Punkte bei exakter Berechnung
sind durch die parameterabhaengige Lage der Stuetzstellen um x=0 bedingt.

MP- und TR-Regel approximieren 'Lineare' Funktionen perfekt.
Die Simpson-Regel approximiert noch 'Kubische' Funktionen perfekt -
also einen Grad hoeher, als man nach Konstruktion erwarten wuerde.

Kommentar:
a) -0.25
b) 0.177245385090551602729816748334114518279
c) Pi/3 = 1.047197551196597746154214461093167628066
"""
