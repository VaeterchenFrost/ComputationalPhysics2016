"""Computational Physics Aufgabe 7.1,  Autor: Martin Roebke 12.06.16
    Quantenmechanik von 1D-Potentialen III - Periodische Potentiale
Periodisches Potential, genutzt Cosinus und Tight-Binding Naeherung.
Ein Teilchen mit Masse m bewegt sich in diesem Potential der Periode=1.

Unter Variation der Blochphase in diskreten Schritten wird in erster
Brillouin-Zone das Eigenwertspektrum En(k) in linken Plotbereich gezeichnet.

Bei linkem Mausklick in En(k) wird zu dessen Position `k` in rechtem Plot das
Potential in Schwarz und Betragsquadrat der Eigenfunktion auf Hoehe jeweiliger
Eigenenergie gezeichnet.
"""

from __future__ import division, print_function # problemlose Division
import numpy as np                              # Arrays, Mathe etc
import matplotlib.pyplot as plt                 # Plotten
import matplotlib as mpl                        # Grundeinstellungen
from scipy.linalg import eigh                   # Matrix Eigenwerte
from functools import partial                   # Voreinstellung *args **kwargs

def per_pot(x, a=1.0):
    """Genutzte eindimensionale Potential-Funktion 'V(x)'
    Parameter:
    x : array_like
        Argument der Funktion.
    a : reelle Zahl
        Skalierung der Amplitude.
    return a * np.cos(2.0 * np.pi * np.array(x))
    """
    return a * np.cos(2.0 * np.pi * np.array(x))


class PeriodischesPotential(object):
    """Berechnung und Zeichnung des zeitunabhaengigen Hamiltonoperators eines
    Teilchens mit effektivem h-quer in einem periodischen 1D-Potential.
    Darstellung des Eigenwertspektrums in der linken Seite des Plot-Fensters.
    Per Mausklick an Position `k` folgt auf der rechten Seite die
    Darstellung des Betragsquadrates der Eigenfunktionen.
    Bietet:
        __init__(self, potential, xr_s, xr_e, emax, nr,
                 krange=100, n_per=1, heff=1, colors=None)
        start(self): Startet Berechnung und Darstellung.
        show(self): Zeige erstellten Figuren. Warte auf Benutzerinteraktion.
    Zusaetzlich:
        kcheck_rechnen_ewvonk(): Kurzpruefung der noetigen Variablen.
        kcheck_zeichnen(): Kurzpruefung der noetigen Variablen.
        rechnen_ewvonk(): Berechne aus Hamiltonian sortierte Eigenwerte
                          mit En(k)<=self.emax fuer alle `k` in self.k
    Privat genutzte Methoden:
        _zeichnen(self): Beginnt Plot. Verbindet _mausklick(self) mit Figur.
        _linien_ewvonk(self): Helferfunktion fuer _zeichnen(self).
        _mausklick(self) Verarbeitet Mausklick. Passt Plot an neue RandBdg. an.
    """
    def __init__(self, potential, xr_s, xr_e, emax, nr,
                 krange=100, n_per=1, heff=1, colors=None):
        """Initialisierung der genutzten Parameter.
        Funktionsparameter:
            potential   : function V(x), genutzte Potentialfunktion.
            xr_s, xr_e  : reell, Rechenbereich des Ortes x.
            emax        : reell, Maximale Eigenenergie genutzter Loesungen.
            nr          : int, Stuetzstellen in Ortsdiskretisierung.
            krange=100  : int, Diskretisierung k in Bereich ]-Pi, Pi]
            n_per=1,    : int, Anzahl Perioden Zeichnen.
            heff=1,     : reelle Zahl, effektives h_quer der Teilchen.
            colors=None : Array, matplotlib-colors fuer den internen Farbzyklus
        """
        self.potential = potential                  # Genutzte Potentialfunktion
        self.heff = heff                            # effektives h-quer
        assert self.heff > 0
        self.xr_s = float(xr_s)                     # Start Rechenbereich in Ort
        self.xr_e = float(xr_e)                     # Ende Rechenbereich in Ort
        assert self.xr_e - self.xr_s == 1.          # Periodenlaenge in Alg.= 1.

        self.emax = emax                            # Maximale E.-Auswertung
        self.krange = int(krange)                   # `k` in Schritten aendern
        assert self.krange > 0
        self.n_per = int(n_per)                     # Anzahl Perioden Zeichnen
        self.nr = int(nr)                           # x-Stuetzstellen
        assert self.nr > 0
        if colors is None: self.colors = ['b', 'g', 'r', 'c', 'm', 'y']
        else: self.colors = colors                  # feste Farbreihenfolge
        self.ef_skal = 0.1                          # Sichtbarkeit der Amplitude
        self.eigenw = ()                            # Tupel, immutable
        self.anzahl_ew = 0                          # Max. beobachtete Linien(k)

        self.plotlinks = []                         # Array k(E)
        self.k = []                                 # Array der k-Werte
        self.x = []                                 # Array der x-Werte
        self.V = []                                 # Array der Potential Werte
        self.k_line = None                          # Klickposition vline

    def start(self):
        """Startet Berechnung und Darstellung der Eigenwerte und Eigenfunktionen
        des Periodischen Potentials mit den vorgegebenen eigenen Parametern.
        """
        print("Start Berechnung.\n   x:[{}, {}[ , Zahl Stuetzpunkte={}, heff={}"
              "\n".format(self.xr_s, self.xr_e, self.nr, self.heff))
        self.rechnen_ewvonk()                       # Berechnung EW/EF in V.
        self._zeichnen()                            # Zeichnung + Interaktion

    def kcheck_zeichnen(self):
        """Kurze Pruefung der bereitgestellten Parameter."""
        s = ()
        if len(self.V)==0:
            s += "Potential self.V ist leer.",
        if len(self.x)==0:
            s += "Ortsarray self.x ist leer.",
        try:
            assert self.h.shape == (self.nr, self.nr)
        except:
            s += "Dimension von Hamiltonian self.h != (self.nr, self.nr)",
        return s

    def rechnen_ewvonk(self):
        """Berechne aus Hamiltonian sortierte Eigenwerte mit En(k) < self.emax
        Erstellt:
            Parameter: self.delta_x, self.z
            Arrays: self.k, self.x, self.V, self.nebendiag
            Hamiltonmatrix: self.h
            Tupel der Ew(k): self.eigenw
        Diese werden in PeriodischesPotential._zeichnen()
        zur Darstellung der Eigensysteme fuer verschiedene `k` genutzt.
        """
        # k-Werte erster Punkt ausgelassen ]-Pi, Pi].
        self.k = - np.linspace(-np.pi, np.pi, self.krange, endpoint=False)
        # Ortsgitterpunkte
        self.x, self.delta_x = np.linspace(self.xr_s, self.xr_e, self.nr,
                                           endpoint=False, retstep=True)
        # Potential
        self.V = self.potential(self.x)
        self.z = 0.5*(self.heff / self.delta_x)**2      # Nebendiagonalelement
        nebendiag = -self.z * np.ones(self.nr - 1)      # Nebendiagonale
        # Konstruiere Matrix-Darstellung - Hamilton-Operator:
        self.h = (np.diag(self.V + 2.0*self.z) +
                  np.diag(nebendiag, -1) +
                  np.diag(nebendiag, 1)
                  ) + 0j                                # Komplex

        for i in range(self.krange):
            # Eintrag unten links und oben rechts - Randbedingungen:
            self.h[-1, 0] = -self.z * np.exp(1j * self.k[i])
            self.h[0, -1] = -self.z * np.exp(-1j * self.k[i])
            # Diagonalisierung, nur Eigenwerte uebernehmen
            ew = eigh(self.h, eigvals_only=True)
            # Suche der weiter genutzten Indizes
            cut = ew[ew < self.emax]
            # Anhaengen der Ew an Auflistung
            self.eigenw += (cut, )

            # Aktualisieren Max Anzahl an n
            if len(cut) > self.anzahl_ew: self.anzahl_ew = len(cut)
        print("Anzahl genutzter Eigenenergien =", self.anzahl_ew)

    def _zeichnen(self):
        """>>Sinnvoll nach `self.rechnen_eigensys`.<<
        Prueft auf Konsistenz der genutzten Eingangs-Parameter.
        Zusammensetzen der genutzten En(k) mittels self._linien_ewvonk().
        Erstellt Figur und 2 Plotbereiche, sowie deren Beschriftung.
        Verbindet 'button_press_event' mit `self._mausklick`.
        """
        # Zusammensetzen der genutzten En(k) in self.plotlinks.
        self._linien_ewvonk()
        # Erstelle eine Figur inklusive Subplots.
        self.fig, (self.axlinks, self.axrechts) = plt.subplots(
                                                   1, 2, figsize=(16,10))
        self.fig.subplots_adjust(left=0.07, bottom=0.08, right=0.94,
                                 top=0.94, wspace=0.15)
        # Achse Variation k
        self.axlinks.axis([-np.pi, np.pi, min(self.V), self.emax*1.1])
        self.axlinks.set_title("$Eigenwertspektrum$")
        self.axlinks.set_xlabel("$Blochphase$"" k")
        self.axlinks.set_ylabel("$Energie$"" E(k)")
        self.axlinks.set_xticks(np.linspace(-np.pi, np.pi, 7))
        self.axlinks.grid(True)
        # Achse fuer Eigenfunktionen mit `self.n_per` Perioden
        self.axrechts.axis([self.xr_s, self.xr_s + self.n_per,
                            min(self.V), self.emax*1.1])
        self.axrechts.set_xlabel("x")
        self.axrechts.set_ylabel(r'$V(x)\ \rm{,\ \|Efkt.\|^{2}\ bei\ EW}$')
         #-------------------------------------------
        print(len(self.k), len(self.plotlinks[:,1]))
        for i in range(self.anzahl_ew):
            # [0]->k, [1]->E_i(k)
            self.axlinks.plot(self.k, self.plotlinks[:,i], 'o', ms=3,
            color=self.colors[i % len(self.colors)])
        # Potential Plot
        self.plot_x = np.concatenate([self.x + i for i in range(self.n_per)])
        self.plot_V = np.tile(self.V, self.n_per)
        self.axrechts.plot(self.plot_x, self.plot_V, color='k')
        # Verknuepfung des button_press_event mit Funktion
        self.fig.canvas.mpl_connect('button_press_event', self._mausklick)

    def _linien_ewvonk(self):
        """Helferfunktion. <<Sinnvoll nach self.rechnen_eigensys.>>
        Zusammensetzen der En(k) Punkte aus ersten Eigenwerten.
        Zuruecksetzen und Speichern der Linien in Array `self.plotlinks`.
        """
        # Array benoetigter Dimension k->, |E:
        self.plotlinks = np.empty((self.krange, self.anzahl_ew))
        for j in range(self.anzahl_ew):         # Fuer alle beob. Eigenenergien
            for i in range(self.krange):        # En an jedem k auffangen
                try:
                    # Versuche alle Energien im Bereich zu finden:
                    self.plotlinks[i, j] = self.eigenw[i][j]
                except IndexError:              # Wenn En an diesem k => Emax
                    self.plotlinks[i, j] = None # Punkt nicht zeichnen

    def _mausklick(self, event):
        """Linker Mausklick in linke Achse waehlt kontinuierlich ein
        k = `event.xdata`. Berechnet und Zeichnet fuer dieses `k`
        die Eigenfunktionen und deren Eigenenergie in rechte Achse.
        Aktualisiert den Titel des rechten Bereichs.
        Die Klickpos. in E(k) wird durch eine vertikale Linie gekennzeichnet.
        """
        mode = plt.get_current_fig_manager().toolbar.mode
        # Test ob Klick mit linker Maustaste und im Koordinatensystem
        # erfolgt ist, sowie ob Funktionen des Plotfensters deaktiviert sind:
        if not (event.button==1 and event.inaxes==self.axlinks and mode==''):
            return
        # Uebernehmen der Mausposition
        kklick = event.xdata
        kklick = max(min(kklick, np.pi), -np.pi)    # Rand 1. Brillouinzone
        self.kwert = kklick                 # Festlegen k fuer rechten Plot.

        # Aktualisieren: Eintrag unten links und oben rechts - Randbedingungen.
        self.h[-1, 0] = -self.z * np.exp(1j * self.kwert)
        self.h[0, -1] = -self.z * np.exp(-1j * self.kwert)
        self.axrechts.clear()                    # Zuruecksetzen Zeichnung
        self.axrechts.plot(self.plot_x, self.plot_V, color='k')
        for i in range(self.n_per + 1):
            self.axrechts.axvline(self.xr_s + i, c='k', ls=":")

        ew, ef = eigh(self.h)                       # Diagonalisierung: ew, ef
        cut = np.where(ew < self.emax)[0]          # Weiter genutzte Indizes
        for i, val in enumerate(ew[cut]): # Auswahl der genutzten Funktionen
            ef_darstellung = np.array(abs(ef[:,i])**2)      # EF ||^2
            ef_darstellung /= self.delta_x              # Integral Normierung
            ef_darstellung = ef_darstellung*self.ef_skal + val   # Skalierung
            plotef = np.tile(ef_darstellung, self.n_per)
            # Darstellung Eigenfunktion
            self.axrechts.plot(self.plot_x, plotef,
                               color=self.colors[i % len(self.colors)])
            # Darstellung Energieerwartung als horizontale Linie.
            self.axrechts.axhline(val, ls="--",
                                  color=self.colors[i % len(self.colors)])
        self.axrechts.set_title("k " + r"$\approx\ {{{:+.3f}}}\ \pi$"
                                "".format(self.kwert/np.pi))
        if self.k_line is None:         # Vertikale Linie Klickposition in E(k).
            self.k_line = self.axlinks.axvline(self.kwert, color="black")
        else:                           # Veraendere vertikale k_line.
            self.k_line.set_xdata(self.kwert)
        self.fig.canvas.draw()                      # Zeichne Figur.

    def show(self):
        """Zeige alle erstellten Figuren. Warte auf Benutzerinteraktion."""
        print("Warte auf Maus-Interaktion.")
        plt.show()

def main():
    """Mainfunktion Quantenmechanik III - Periodische Potentiale.
    Zur Eingabe der Parameter und Starten gewuenschter Realisierungen.
    """
    # Parameter
    heff = 0.2              # hquer effektiv fuer Teilchen mit Masse m
    nr = 160                # Matrixgroesse fuer Ortsdiskretisierung
    xr_s = 0.               # x-Bereich fuer Rechnung
    xr_e = 1.
    emax = 7.               # Obere Schranke E fuer gezeichnete Eigenenergien.
    krange = 100            # Schritte der Blochphase
    n_per = 4               # Wiederholung von Perioden in Zeichnung
    a = 1.00                # Parameter Potential
    potential = partial(per_pot, a=a)
    mpl.rcParams.update({'font.size': 14})
    # Anfangstext Konsole
    print(__doc__)
    # Instanziierung
    rea = PeriodischesPotential(potential, xr_s, xr_e, emax, nr,
                                krange, n_per, heff)
    rea.start()                                 # Starte Berechnung und Plot
    rea.show()                                  # Benutzerinteraktion

#-------------Main Programm---------------
if __name__ == "__main__":
    main()

"""Kommentar:
a) Das Spektrum einzelner Loesungen ist beobachtbar stetig.
    Fuer niedrige EF, E<max(V) ist im Ortsraum ein aehnliches Verhalten der
    Amplitude zu erkennen, wie bereits in 1D Potentialen allgemein (vorh. Ueb.)
    Linien mit E>V ruecken fuer k->0 bzw in andere Richtung fuer
    k->Rand mit einer Nachbarfunktion zusammen, und bilden dabei nahezu
    kontinuierliche Baender. Am Rand der B.Z. sind Energie-Luecken zu erkennen.
    Bei Variation von Parameter k veraendern sich die Eigenenergien der Loesun-
    gen, und naehern sich stets fuer E>>V der quadratischen Dispersion E~k^2.
    Bei k=0 ist der Phasenfaktor e^ik = exp(0) = 1,
    An |k|=Pi entsprechend +1 oder -1.
    An diesen drei Punkten ist das Energiespektrum fuer ein periodisches
    Potential erwartungsgemaess E'(k) = 0, und es bilden sich EF als
    stehende Wellen. Das Spektrum ist achsensymmetrisch zu k=n*Pi, n int.
    =>Periodizitaet der Brillouinzone zu k+2Pi/p
    Betragsquadrat der EF ist jeweils achsensymmetrisch zu Symmetrieachsen
    des Potentials; x=n/2, n int.

b) Bei einem Grenzfall des sehr schwachen Potentials und hohen Energien
    sollte die Eigenschaft des freien (ungebundenen) Teilchens zum Vorschein
    kommen. Elektronen mit hohen Energien werden monochrom und breiten sich
    nahezu uniform im gesamten Raum aus. Dann wird EF und Dispersion:
    Psi(x) = e^(ikx) sowie E(k) = hquer^2 *k^2 / (2m)
    Dies ist hier z.B. an einer Amplitude `a = 0.` nachzuvollziehen.
"""
