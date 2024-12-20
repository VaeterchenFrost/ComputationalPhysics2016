"""Computational Physics Aufgabe 4.1,  Autor: Martin Roebke 08.05.16
Dynamik eines Teilchens im angetriebenen Doppelmuldenpotential.
V(x) = x^4 - x^2 + x[A + B*sin(w*t)]

Integration der dimensionslosen Hamiltonfunktion mittels
*scipy.integrate.odeint*

Nutzung von: class DoppelmuldeDGL(object)
             Verwaltung der Plots und Nutzer-Interaktion.

Linker Plotbereich: Genaeherte Trajektorie in Phasenraum.
Rechter Plotbereich: zu t = 2pi/w*i stroboskopische Darstellung.

Linker Mausklick in einen der Plots legt Startwerte fuer Teilchen fest.
"""

import numpy as np  # Arrays, Mathe etc
import matplotlib.pyplot as plt  # Plotten
from matplotlib import rc as mplrc
from cycler import cycler  # Colorcycle
from scipy.integrate import odeint  # Integrationsroutine fuer DGL
from matplotlib.backend_bases import MouseEvent
from typing import Callable, List, Union


def abl_hamilton(x: np.ndarray, t: float, A: float, B: float, w: int) -> np.ndarray:
    """Rechte Seite der DGL - Hamilton-Funktion
    - dH(x, p, t)/dx
    Parameter t: Zeit, A: Absolut-Linearterm, B: Schwingung, w: Kreisfrequenz
    """
    return np.array([x[1], -4 * x[0] ** 3 + 2 * x[0] - (A + B * np.sin(w * t))])


class DoppelmuldeDGL(object):
    """Berechnet und zeichnet Phasenraumtrajektorien und stroboskopische
    Darstellung zu Teilchen in angetriebenem Doppelmuldenpotential.
    Loesung der Hamilton-DGL mittels *scipy.integrate.odeint*.
    Darstellung und Wahl des Startzustandes mittels Mausklick
    in einen der zwei Achsenbereiche.
    """

    def __init__(self, a: float, b: float, w: int, abl: Callable, perioden: int, num_t: int=500):
        """Initialisierung der Parameter.
        Pruefung auf Winkelgeschwindigkeit rund Null.
        Konsolenausgabe der Parameter nach Initialisierung.
        self.traj_t:
            Erstellung der gewuenschten Rueckgabe-Zeiten der DGL-Loesung.
        """
        self.a = a
        self.b = b
        if w < 10**-15:
            raise ValueError("Kreisfrequenz w=" + str(w) + " zu klein!")
        self.w = w
        self.abl = abl
        self.perioden = perioden
        self.num_t = num_t
        print(
            "DoppelmuldeDGL mit A={}, B={}, w={} fuer {} Zeiten in {} "
            "Perioden gestartet.".format(a, b, w, num_t, perioden)
        )
        self.axw = []
        self.traj_t = np.linspace(
            0.0,  # Startzeit
            perioden * 2.0 * np.pi / self.w,  # Letzte Zeit
            perioden * num_t + 1,  # Punkte + Start
        )

    def cr_figure(self, achsenweite: List[Union[float, int]]):
        """Initialisieren und Beschriften der Plot-Figur mit zwei Achsen.
        Verbindet Mausklick mit *self.klick*.
        achsenweite : [xmin, xmax, ymin, ymax]
            Setze die Begrenzungen der zu erstellenden Plotbereiche.
        """
        self.axw = achsenweite
        self.fig = plt.figure(figsize=(14, 10))
        self.axes1 = self.fig.add_subplot(121)
        self.axes1.axis(achsenweite)
        self.axes1.set_title("Doppelmuldenpotential Trajektorie", fontsize="x-large")
        self.axes1.set_xlabel("$x$")
        self.axes1.set_ylabel("$p$")

        self.axes2 = self.fig.add_subplot(122)
        self.axes2.axis(achsenweite)
        self.axes2.set_title("Doppelmuldenpotential Stroboskopisch", fontsize="x-large")
        self.axes2.set_xlabel("$x$")
        self.axes2.set_ylabel("$p$")
        # Bei Mausklick aufrufen:
        self.fig.canvas.mpl_connect("button_press_event", self._klick)

    def _kontur(self, levels: None=None, delta: float=0.025):
        """Helferfunktion. Berechnen von Konturlinien der Werte in *levels* mit
        Energie H = 0.5*pl*pl + xl**4 - xl**2 + self.a*xl
        innerhalb von Breich *self.axw* mit einer Disketisierung *delta*.
        Einzeichnen in Schwarz in *self.axes1* und *self.axes2*.
        levels : Sequenz (muss in aufsteigender Reihenfolge sein)
            Bestimmt Hoehen der Konturlinien.
            Default ist np.linspace(0, 3, 9)**2.5
        delta : reelle Zahl, Feinheit zwischen Stuetzstellen.
        """
        if levels is None:
            levels = np.linspace(0, 3, 9) ** 2.5
        x = np.arange(self.axw[0], self.axw[1], delta)
        y = np.arange(self.axw[2], self.axw[3], delta)
        xl, pl = np.meshgrid(x, y)
        H = 0.5 * pl * pl + xl**4 - xl**2 + self.a * xl
        self.axes1.contour(xl, pl, H, levels, colors="black")
        self.axes2.contour(xl, pl, H, levels, colors="black")
        # ~ QCS = self.axes1.contour(xl, pl, H, levels, colors='black')
        # ~ for seg in QCS.allsegs: # Instabil Aber Schneller :
        # ~ self.axes2.plot(seg[0][:,0], seg[0][:,1], c='k')

    def _klick(self, event: MouseEvent):
        """Berechnet und Plottet neue Trajektorien.
        Der Anfangspunkt wird durch Mausklick festgelegt.
        Konsolenausgabe Startwert.
        """
        # Test, ob Klick mit linker Maustaste und im Koordinatensystem
        # erfolgt sowie ob Zoomfunktion des Plotfensters deaktiviert ist:
        mode = plt.get_current_fig_manager().toolbar.mode
        if event.button == 1 and event.inaxes and mode == "":
            startwert = np.array([event.xdata, event.ydata])
            self.plot(startwert)
            print("Start [x0, p0] :", startwert)

    def plot(self, startwert: np.ndarray):
        """Berechnet Trajektorie zu Startwert (x0, p0) und fuegt diese in beiden
        Plots jeweils hinzu.
        """
        x_t = odeint(
            self.abl, startwert, self.traj_t, args=(self.a, self.b, self.w)
        )  # Berechnen Trajektorie
        strobo = x_t[:: self.num_t]  # Kopieren der stroboskopischen Punkte
        self.axes1.plot(x_t[:, 0], x_t[:, 1], ".", ls="-")
        self.axes2.plot(strobo[:, 0], strobo[:, 1], "o", ms=3.0, mew=0)
        self.fig.canvas.draw()

    def show(self):
        """Zeige alle erstellten Figuren. Warte auf Benutzerinteraktion."""
        plt.show()


def main():
    """Mainfunktion Doppelmuldenpotential.
    Zur Eingabe der Parameter und Starten gewuenschter Realisierungen.
    """
    # Parameter
    a = 0.1  # Linearterm in Potential
    b = 0.1  # Sinusbewegung Potential
    w = 1  # Omega in sin(w*t)
    abl = abl_hamilton  # Rechte Seite der DGL
    perioden = 200  # Laenge der Berechnung
    t_pro_periode = 400  # Anz. Zwischenzeiten
    achsenweite = [-2.3, 2.3, -5, 5]
    mplrc("axes", prop_cycle=(cycler("color", ["r", "c", "m", "y", "b", "g"])))

    # Anfangstext Konsole
    print(__doc__, "\n")

    # Realisierung
    rea = DoppelmuldeDGL(a, b, w, abl, perioden, t_pro_periode)
    rea.cr_figure(achsenweite)  # Figure erstellen
    rea._kontur()  # Konturlinien zeichnen
    rea.show()  # Darstellung + Interaktion


# -------------Main Programm----------------
if __name__ == "__main__":
    main()

"""Kommentare:
a) Geschlossene Phasenraum-Trajektorien stellen periodische Bewegungen dar.
Da es zwei Mulden gibt, kann eine Trajektorie in einer der beiden liegen,
zwischen beiden hin und her wechseln, oder beide umschliessen.
Phasenraum-Trajektorien koennen sich nicht schneiden.
b) Das Teilchen kann mit geringer Energie in einer Mulde schwingen oder
Buckel ueberwinden oder von Aussenrand her mit grosser Energie ueber beide
Mulden rollen.

c) B=0.1:
Prinzipiell aehnlich (kleine Stoerung des Potentials).
Es sind Bahnen in flachem Potential besonders
mit Abweichungen von der Bahn im stationaeren Potential zu beobachten,
da sich die Sinus-Aenderung des Potentiales auf die
Stabilitaet der Bahnen auswirkt.
Weiter aussen ist die Sinusschwingung im Vergleich zu der ohnehin vorhandenen
Energie des Teilchens nicht so bedeutend.
Es sind auch selbstaehnliche, stark-periodische, Bereiche zu beobachten.

d) Ermittelte Startkoordinaten periodischer Trajektorie auf
Start [x,p] : [0.03091677, -0.56679542]
mit t_p = 1 * 2*Pi/w = 2*Pi.
"""
