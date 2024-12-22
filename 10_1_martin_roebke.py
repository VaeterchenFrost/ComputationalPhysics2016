"""Computational Physics Aufgabe 10.1,  Autor: Martin Roebke 03.07.16
Visualisiert das 2D-Ising-Modell, wobei in einem linken Plot der Spinzustand
eines 50x50 Gitters mit periodischen Randbedingungen dargestellt wird.

In rechtem Plot-Bereich wird die mittlere Magnetisierung `m`
ueber der dimensionslosen Temperatur tau=k_b*T/J gezeichnet.
"""

from warnings import filterwarnings

import matplotlib.pyplot as plt  # Plotten
import numpy as np  # Arrays, Mathe etc


def isIncreasing(l):
    """Testet array_like `l` auf ansteigend-geordnete Werte."""
    for i, elem in enumerate(l[1:]):
        if elem <= l[i]:
            print("Kein Anstieg bei: {}->{}".format(l[i], elem))
            return False
    return True


class SpinKonfig(object):
    """Spinkonfiguration auf 2D-Gitter bei einheitenloser 'Temperatur'
    tau=kb*T/J.
    Bietet:
        __init__(self, n): Initialisiere Gitter n*n dtype=np.int8 und Attribute.
        neue_konfig(self, tau, m0): Konfiguration auf `tau` und `m`.
        erstelle_spins(self, m0): Helper, erstelle Array self.n*self.n .
        setze_m(self): Setze self.m, Mittlere Magnetisierung auf Gitter.
        setze_arwk(self, gefroren=0.05): Wahrsch. fuer Uebergang setzten.
        flip(self, x, y): Drehe Spin an Stelle (x, y) um.
        wk_spinflip(self, x, y): Wahrscheinlichkeit fuer Spinflip an (x, y).
    """

    def __init__(self, n, tau=0.0, m0=1.0):
        """Initialisiere Parameter.
        tau : Nichtnegative Zahl, Dimensionslose Temperatur.
        m0 : Angestrebte mittlere Magnetisierung [-1, 1].
        Setze:
         self.n : Gitterlaenge int(n) groesser Null.
         self.dH : Array der Moeglichkeiten einer "Energie"-Aenderung bei Flip.
         self.s_arr : None.
            Genutzt fuer Array der Spins, gesetzt mit self.erstelle_spins.
         self.arwk : None.
            Genutzt fuer Array der Uebergangswahrsch. entsprechend self.dH
         self.m : None.
            Genutzt fuer aktuelle Mittlere Magnetisierung der Spins.
        """
        self.dH = 2 * np.linspace(-4, 4, 5)  # [-8., -4., 0., 4., 8.]
        self.n = int(n)  # Gitterlaenge; N=n*n
        assert self.n > 0
        self.tau = float(tau)
        assert self.tau >= 0  # Konsistenz tau >= 0
        m0 = float(m0)
        assert abs(m0) <= 1  # Konsistenz m0 [-1, 1]
        self.s_arr = self.erstelle_spins(m0)
        self.setze_m()  # Aktualisiere `self.m`
        self.setze_arwk()  # Aktualisiere `self.arwk`
        print(
            "SpinKonfig initialisiert mit: tau = {:.3f}, m = {:.3f}"
            "".format(self.tau, self.m)
        )

    def neue_konfig(self, tau, m0):
        """Konfiguriere Spingitter mit diml. Temperatur tau und 'm' moeglichst
        nahe an m0. Aktualisiere self.m, self.arwk.
        """
        m0 = float(m0)
        assert not abs(m0) > 1  # Konsistenz m0 [-1, 1]
        self.tau = float(tau)
        assert not self.tau < 0  # Konsistenz tau >= 0
        self.s_arr = self.erstelle_spins(m0)
        self.setze_m()  # Aktualisiere `self.m`
        print("Aktualisiere auf tau = {:.3f}, m = {:.3f}".format(self.tau, self.m))
        self.setze_arwk()  # Aktualisiere `self.arwk`

    def erstelle_spins(self, m0):
        """Erstelle 2D Array self.n*self.n mit moeglichst guter Darstellung der
        mittleren Magnetisierung `m0`.
        """
        anz_neg = int(round(self.n * self.n * (1 - m0) / 2))
        arhilf = np.ones(self.n * self.n, dtype=np.int8)
        arhilf[range(anz_neg)] = -1  # Negative Spins
        if anz_neg > 0 and anz_neg < self.n * self.n:
            # Wenn verschiedene Spins in System:
            np.random.shuffle(arhilf)  # in-place schuetteln
        return np.reshape(arhilf, (self.n, self.n))  # Rueckgabe 2D Array.

    def setze_m(self):
        """Mittlere Magnetisierung auf Gitter = Summe Spins / Anzahl Spins."""
        self.m = np.sum(self.s_arr) / self.n / self.n

    def setze_arwk(self, gefroren=0.05):
        """Nach Neusetzen von `self.tau` die Flip-Wahrscheinlichkeiten
        passend zu self.dH aktualisieren.
        gefroren : reelle Zahl, unter der Uebergangswk. maximal gesetzt werden.
        Fuer tau~0.05: arwk = [1, 1, 1, 3.79523030e-35, 1.44037730e-69]
        Dies ist numerisch hier gleich [1,1,1,0,0] zu setzen.
        Sonst Gefahr bei sehr kleinen tau: overflows/unnoetige Numerik.
        """
        if self.tau <= gefroren:  # Eingefroren
            self.arwk = np.zeros_like(self.dH)  # Zahlt Energie -> 0.
            self.arwk[self.dH <= 0] = 1.0  # Verringert Energie -> 1.
        else:
            self.arwk = np.exp(-self.dH / self.tau)
            self.arwk[self.arwk > 1] = 1.0

    def flip(self, x, y):
        """Drehe Spin an Stelle (x, y) um."""
        self.s_arr[x, y] *= -1

    def wk_spinflip(self, x, y):
        """Berechnung Wk fuer potentiellen Spinflip an (i, j) fuer
        Periodische Randbedingung und naechste Nachbarn Wechselwirkung.
        """
        r = self.s_arr[x, (y + 1) % self.n]
        l = self.s_arr[x, (y - 1) % self.n]
        o = self.s_arr[(x + 1) % self.n, y]
        u = self.s_arr[(x - 1) % self.n, y]
        dH = 2 * self.s_arr[x, y] * (r + l + o + u)
        # Auf gleiche Indizes gelegt.
        return self.arwk[self.dH == dH]


class IsingModell(object):
    """Verwaltet Darstellung und Interaktion eines 2D Ising-Modell.
    Bietet:
        __init__(self, axspin, axpha, n, mc_steps=10, setup=True)
        __call__(self, tau_s, m_s): Startet Berechnung und Darstellung.
        plotumgebung(self): Richte Zeichenbereiche ein.
        theorie_m(self, taux): theoretische mittlere Magnetisierungen.
        mc_schritt(self): Ein Schritt nach Metropolis-Algorithmus auf Spins.
        show(self): Zeige erstellten Figuren. Warte auf Benutzerinteraktion.
    Privat genutzte Methoden:
        _klick(self): Verarbeitet Mausklick in self.axes.
    """

    def __init__(self, axspin, axpha, n, mc_steps=10, setup=True):
        """Initialisierung.
        Parameter:
            axspin : Axes-Bereich fuer Bild der Spinkonfiguration.
            axpha : Axes-Bereich fuer m-von-tau.
            n : int(n) Gitterlaenge des 2D Spingitters.
            mc_steps : int(mc_steps) Monte-Carlo Schritte nach einem Klick.
            setup : boolean, True : Richte Plotbereiche fuer Nutzung ein.
        Setzt `self.axspin` und `self.axpha` zurueck.
        Initialisiert SpinKonfig(self.n) in `self.Spins`.
        Setzt `self.maxtau` als groesstes zu zeichnendes tau.
        Initialisiert Schalter `self.plot_aktiv`, True: Rechnung laeuft.
        """
        self.axspin = axspin  # Plotbereich Spins
        self.axpha = axpha  # Plotbereich m von tau
        self.n = int(n)
        assert self.n > 0
        self.mc_steps = int(mc_steps)
        assert self.mc_steps > 0
        self.maxtau = 5
        self.plot_aktiv = False
        # Aktiv
        self.axspin.cla()
        self.axpha.cla()
        self.Spins = SpinKonfig(self.n)
        if setup:
            self.plotumgebung()
        print(
            "\nSchwarze Bereiche : Spin '+1' / Weisse Bereiche : Spin '-1'."
            "\nRote Linien : Mittlere Magnetisierung im thermod. Limes."
            "\nSchwarzer Punkt rechts: Zeigt aktuelle m[tau] "
            "der Spinkonfiguration.\n::Bei Linksklick auf:"
            "\n:: Spins : Starte '{}' Monte-Carlo-Zeitschritte."
            "\n:: m[tau]-Plot : Initialisiere Spins mit 'm' und 'tau'."
            "\n".format(self.mc_steps)
        )

    def __call__(self, tau_s, m_s):
        """Starte Darstellung; Erster Startpunkt auf m ~ m_s, tau ~ tau_s."""
        self.Spins.neue_konfig(tau_s, m_s)  # Aufruf mit Startwert
        self.imh = self.axspin.imshow(
            self.Spins.s_arr, interpolation="none", cmap=plt.get_cmap("Greys")
        )
        self.pointer.set_data([self.Spins.tau], [self.Spins.m])

    def plotumgebung(self):
        """Einrichten `self.axspin` und `self.axpha`.
        Verbindet 'button_press_event' mit self._klick.
        """
        self.axspin.set_title("$Spinkonfiguration$", y=1.02, fontsize=20)
        self.axpha.set_title(r"$Mittlere\ Magnetisierung$", y=1.02, fontsize=20)
        self.axspin.set_xticks([])
        self.axspin.set_yticks([])
        self.axpha.set_xlabel(r"$\tau$")
        self.axpha.set_ylabel("m", rotation="horizontal", y=0.85)
        self.axpha.axis([0, self.maxtau, -1.1, 1.1])
        (self.pointer,) = self.axpha.plot([], [], "ko")
        color_mag = "red"  # Einheitliche th. Zeichenfarbe
        # Stuetzen theoretische Kurven; kritische Temperatur ca 2.269
        th_kurven, tau = self.theorie_m(0, self.maxtau)
        for tk in th_kurven:
            self.axpha.plot(tau, tk, color=color_mag)
        self.axpha.figure.canvas.mpl_connect("button_press_event", self._klick)

    def theorie_m(self, t_min, t_max, dtau=100):
        """Formel m : +,- (1 - 1/np.sinh(2/tau)**4)**0.125"""
        # TEXT
        if t_max < 0:
            return [], []  # Negative Temperatur
        assert t_min < t_max  # Ordnung korrekt
        grenztau = 2.0 / np.arcsinh(1)  # Grenztemperatur
        t_min = max(0.0, t_min)  # t_min>=0
        taux_ar = []  # Sammelt Stuetzpunkte
        # x-Bereich zusammensetzen
        if t_min < 2.2:  # flacher Bereich
            taux_ar += [np.linspace(t_min, min(t_max, 2.2), dtau)]
        if t_min < grenztau and t_max > 2.2:  # steiler Bereich
            taux_ar += [
                np.linspace(max(2.2, t_min), min(t_max, grenztau), dtau, endpoint=False)
            ]
        if t_max >= grenztau:  # Waagerecht
            taux_ar += [grenztau, t_max]
        tau = np.hstack(taux_ar)
        mo = np.ones_like(tau, dtype=float)
        # mk: tau nach differenzierbarer Form
        tau_slice = (tau > 0) * (tau < grenztau)  # Bool*Bool -> Logisch und
        mo[tau_slice] = (1 - 1 / np.sinh(2 / tau[tau_slice]) ** 4) ** 0.125
        mo[tau >= grenztau] = 0.0
        mu = -mo  # Erstellen unterer Linie
        return (mo, mu), tau

    def _klick(self, event):
        """Verwaltet Mausklick.
        Bei Linksklick in einen der Achsenbereiche:
        """
        # TEXT
        # Test ob Funktionen des Plotfensters deaktiviert sind:
        mode = plt.get_current_fig_manager().toolbar.mode
        if not (mode == "" and event.button == 1):
            return
        if event.inaxes == self.axspin:  # Iterationen
            self.plot_aktiv = True
            for i in range(self.mc_steps):
                self.mc_schritt()
                plt.setp(self.imh, data=self.Spins.s_arr)
                self.pointer.set_ydata([self.Spins.m])
                plt.pause(0.10)
            self.plot_aktiv = False
        elif event.inaxes == self.axpha:
            if self.plot_aktiv:
                return  # Noch in Zeichnung&Rechnung
            # Neue Startposition
            t, m = event.xdata, event.ydata
            tau_s = max(t, 0.0)  # tau Nicht negativ
            m_s = min(1.0, max(m, -1.0))  # m in [-1, +1]
            self.Spins.neue_konfig(tau_s, m_s)  # Aufruf mit Startwert
            plt.setp(self.imh, data=self.Spins.s_arr)
            self.pointer.set_data([self.Spins.tau], [self.Spins.m])
            self.axpha.figure.canvas.draw()

    def mc_schritt(self):
        """Spins von n*n Gitterpunkten (zufaellig gewaehlt) nach
        Metropolis-Algorithmus umklappen,
        wenn Wahrscheinlichkeit von `self.Spins.wk_spinflip()` erfuellt.
        """
        for k in range(self.n * self.n):
            # Zwei Zufallszahlen Pos.
            x, y = np.random.randint(self.n, size=2)
            wk = self.Spins.wk_spinflip(x, y)  # Wahrscheinlichkeit flip
            if wk == 1:
                self.Spins.flip(x, y)  # 1: Direkt Spin drehen.
            elif np.random.rand() <= wk:
                self.Spins.flip(x, y)
        # Neue Spin-Konfiguration erstellt.
        self.Spins.setze_m()

    def show(self):
        """Zeigt alle erstellten Figuren. Wartet auf Benutzerinteraktion."""
        plt.show()


def main():
    """Mainfunktion fuer Ising-Modell.
    Eingabe der Parameter und Starten gewuenschter Realisierungen.
    """
    # Anfangstext Konsole
    print(__doc__)

    # Numerik
    n = 50
    m0 = 0.6
    t0 = 1.0
    mc_steps = 1

    # Darstellung
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.subplots_adjust(left=0.06, bottom=0.08, top=0.85, wspace=0.21)
    fig.suptitle("Ising-Modell", fontsize=28)

    Modell = IsingModell(axes[0], axes[1], n, mc_steps)
    Modell(t0, m0)
    Modell.show()


# -------------Main Programm----------------
if __name__ == "__main__":
    main()  # Rufe Mainroutine


"""Kommentar:
Teilweise Ausbildung fester Gebiete fuer t<tc bei Start bei m~0.
tau=1.0 : Gute Konvergenz gegen m_unendlich.
 m=-1: Teilweises verteiltes Umklappen von Spins, welche wieder zurueckklappen.
 m=-0.5: Bilden und Aufloesen von Domaenen bei Konvergenz gegen m~-0.999.
 m=-0.3: Konvergenz nach erstem Klick auf m~-0.8, bei Bildung von Domaenen.
         Es bilden sich Domaenen von ca 20-Breite, welche danach verschwinden.
         Es dauert ca 4 Klicks bis Konvergenz zu Theorie.
 m=0: Es bilden sich sehr grosse Domaenen auf ca 1/2 der Flaeche.
      Diese verschwinden nach ~10 Klicks nicht, sondern veraendern nur ihre
      Kantenverlaeufe. Diese verbinden sich weiterhin, bis nur zwei Flaechen
      gleicher Spins uebrug bleiben. Diese 'bergradigen' sich zum Grossteil.
tau=3.0 : Staerkere Schwankungen um m_unendlich.
 m=-1: Konvergenz nach erstem Klick auf m~-0.3, bei Domaenengroessen von ca 6.
       Danach zerstreute Gebiete mit ca halbe-halbe Belegung.
 m=-0.5: Konvergenz nach erstem Klick auf m~-0.2, sehr zerstreute Gebiete.
 m=-0.3: Aehnlich zu -0.5. Schwankung ca m+-0.1 um Null.
 m=0: Ausbildung von zerstreuten Domaenen, Schwankung ca m+-0.1 um Null.
"""
