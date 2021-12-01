"""Berechnung von Eigenwerten und Eigenfunktionen der 1D Schroedingergleichung.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh


def diskretisierung(xmin, xmax, N, retstep=False):
    """Berechne die quantenmechanisch korrekte Ortsdiskretisierung.

    Parameter:
        xmin: unteres Ende des Bereiches
        xmax: oberes Ende des Bereiches
        N: Anzahl der Diskretisierungspunkte
        retstep: entscheidet, ob Schrittweite zurueckgegeben wird
    Rueckgabe:
        x: Array mit diskretisierten Ortspunkten
        delta_x (nur wenn `retstep` True ist): Ortsgitterabstand
    """
    delta_x = (xmax - xmin) / (N + 1.0)                    # Ortsgitterabstand
    x = np.linspace(xmin+delta_x, xmax-delta_x, N)         # Ortsgitterpunkte

    if retstep:
        return x, delta_x
    else:
        return x


def diagonalisierung(hquer, x, V):
    """Berechne sortierte Eigenwerte und zugehoerige Eigenfunktionen.

    Parameter:
        hquer: effektives hquer
        x: Ortspunkte
        V: Potentialwerte an x
    Rueckgabe:
        ew: sortierte Eigenwerte (Array der Laenge N)
        ef: entsprechende Eigenvektoren, ef[:, i] (Groesse N*N)
    """
    delta_x = x[1] - x[0]
    N = len(x)
    z = hquer**2 / (2.0*delta_x**2)                        # Nebendiagonalelem.
    h = (np.diag(V + 2.0*z) +
         np.diag(-z*np.ones(N-1), k=-1) +                  # Matrix-Darstellung
         np.diag(-z*np.ones(N-1), k=1))                    # Hamilton-Operat.

    ew, ef = eigh(h)                                       # Diagonalisierung
    ef = ef/np.sqrt(delta_x)                               # WS-Normierung
    return ew, ef


def plot_eigenfunktionen(ax, ew, ef, x, V, width=1, Emax=0.15, fak=0.01,
                         betragsquadrat=False, basislinie=True, alpha=1.0,
                         title=None):
    """Darstellung der Eigenfunktionen.

    Dargestellt werden die niedrigsten Eigenfunktionen 'ef' im Potential 'V'(x)
    auf Hoehe der Eigenwerte 'ew' in den Plotbereich 'ax'.
    Die Eigenwerte werden hierbei als sortiert angenommen.

    Optionale Parameter:
        width: (mit Default-Wert 1) gibt die Linienstaerke beim Plot der
            Eigenfunktionen an. width kann auch ein Array von Linienstaerken
            sein mit einem spezifischen Wert fuer jede Eigenfunktion.
        Emax: (mit Default-Wert 1/10) legt die Energieobergrenze
            fuer den Plot fest.
        fak: ist ein Skalierungsfaktor fuer die graphische Darstellung
            der Eigenfunktionen.
        betragsquadrat: gibt an, ob das Betragsquadrat der Eigenfunktion oder
            die (reelle!) Eigenfunktion selbst dargestellt wird.
        basislinie: gibt an, ob auf Hoehe der jeweiligen Eigenenergie eine
            gestrichelte graue Linie gezeichnet wird.
        alpha: gibt die Transparenz beim Plot der Eigenfunktionen an (siehe
            auch Matplotlib Dokumentation von plot()). alpha kann auch ein
            Array von Transparenzwerten sein mit einem spezifischen Wert
            fuer jede Eigenfunktion.
        title: Titel fuer den Plot.
    """
    if title is None:
        title = "Asymm. Doppelmuldenpotential"

    plt.axes(ax)                                      # Ortsraumplotfenster
    plt.setp(ax, autoscale_on=False)
    plt.axis([np.min(x), np.max(x), np.min(V), Emax])
    plt.xlabel(r'$x$')
    plt.title(title)

    plt.plot(x, V, linewidth=2, color='0.7')          # Potential plotten
    anz = np.sum(ew <= Emax)                          # Zahl zu plottenden Ef

    if basislinie:                                    # Plot Basislinie bei Ew
        for i in np.arange(anz):
            plt.plot(x, ew[i] + np.zeros(len(x)), ls='--', color='0.7')

    try:                                              # Verhaelt sich width
        iter(width)                                   # wie ein array?
    except TypeError:                                 # Falls `width` skalar:
        width = width * np.ones(anz)                  # konst. Linienstaerke

    try:                                              # entsprechend fuer
        iter(alpha)                                   # Transparenz alpha
    except TypeError:
        alpha = alpha * np.ones(anz)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']           # feste Farbreihenfolge
    if betragsquadrat:                                # Plot Betragsquadr. Efkt
        plt.ylabel(r'$V(x)\ \rm{,\ \|Efkt.\|^{2}\ bei\ EW}$')
        for i in np.arange(anz):
            plt.plot(x, ew[i] + fak*np.abs(ef[:, i])**2, linewidth=width[i],
                     color=colors[i % len(colors)], alpha=alpha[i])
    else:                                             # Plot Efkt
        plt.ylabel(r'$V(x)\ \rm{,\ Efkt.\ bei\ EW}$')
        for i in np.arange(anz):
            plt.plot(x, ew[i] + fak*ef[:, i], linewidth=width[i],
                     color=colors[i % len(colors)], alpha=alpha[i])
