"""Computational Physics Aufgabe 1.1,  Autor: Martin Roebke 29.07.16

Berechnet und zeichnet Standardabbildung des 'Gekickten Rotor'
fuer Kickstaerke `K` auf linken Mausklick in
Phasenraum mit periodischen Randbedingungen.
Zusaetzlich wird ein Slider fuer `K` und ein Button fuer 'Zuruecksetzen'
bereitgestellt,
sowie konzentrische Klicks auf rechten Mausklick zusammengefasst.

class StdAbb(object)
    Notwendige Parameter und Iteration zur Berechnung der Standardabbildung.
class StdAbbPlot(object)
    Zeichnung Phasenraum und Verwaltung der Nutzer-Interaktion.
"""

import matplotlib.pyplot as plt  # Plotten
import numpy as np  # Arrays, Mathe etc
from matplotlib.widgets import Slider, Button  # Plot-Widgets


def sli_freeze(sli):
    """Slider `sli` reagiert danach nicht mehr auf einkommende Events."""
    sli.active = False


def sli_release(sli, mouse_release=True):
    """Slider `sli` reagiert wieder auf einkommende Events.
    mouse_release : bool, True
        Gibt Mauszeiger von Slider frei.
    """
    if mouse_release:  # Selbst freigeben nach inaktiv
        sli.canvas.release_mouse(sli.ax)
    sli.active = True  # Slider reagiert auf Events


class StdAbb(object):
    """Berechnet Standardabbildung nach Parametern Kickstaerke und
    maximaler Iterationstiefe.

    Getrennte Arrays fuer Winkel und Impuls erstellen.
    self.standard_abbildung(self, t, p, redraw=False): Arrays Fuellen.
    """

    def __init__(self, K, max_iterationen):
        """Initialisierung der Parameter.
        self.kick = K : Aktueller Parameter Kickstaerke
        self.max_iterationen = max_iterationen :
            Iterations-Punkte zugehoerig zu einem Startwert.
        """
        self.kick = K  # Jeweils aktueller Kick
        self.max_iterationen = max_iterationen  # Maximale Punkte Plot
        self.tupstart = ()  # Speichert Startwerte
        # Theta- und Impuls-Array erzeugen
        self.t_ar = np.zeros(self.max_iterationen + 1, dtype=float)
        self.p_ar = np.zeros(self.max_iterationen + 1, dtype=float)

    def kickneu(self, val):
        """Ersetzt self.kick durch neuen Wert `val`.
        val : reelle Zahl.
        """
        self.kick = val

    def standard_abbildung(self, t, p, redraw=False):
        """Berechnet Standardabbildung nach Parametern
        Getrennte Arrays fuer theta und Impuls erstellen und berechnen.

        Argumente: t, p, redraw
        Rueckgabe: tlist, plist
        """
        self.t_ar[0] = t  # Startwert Theta
        self.p_ar[0] = p  # Startwert Impuls
        # Durchauf von i+1=1 bis maxiter, jeweils iterative Berechnung
        for i in range(self.max_iterationen):
            self.t_ar[i + 1] = self.t_ar[i] + self.p_ar[i]
            self.p_ar[i + 1] = self.p_ar[i] + self.kick * np.sin(self.t_ar[i + 1])

        self.t_ar = self.t_ar % (2.0 * np.pi)  # Modulo-Operationen
        self.p_ar = (self.p_ar + np.pi) % (2.0 * np.pi) - np.pi
        if not redraw:  # Append Startposition
            self.tupstart += ([t, p],)

        return self.t_ar[:], self.p_ar[:]  # Return beide Listen


class StdAbbPlot(object):
    """Zeichnen einer Abbildung in Phasenraum nach einer
    Berechnungsvorschrift eines Berechnungsobjektes.
    Slider fuer einen Parameter `kick` der Abbildung.
    Button fuer Reset des Plots.
    Festlegen von Startpunkten mittels Mausklick.
    Maus : Links - Ausführen von Startposition.
           Rechts- Mehrere Startpositionen um Klick konzentriert.
    """

    def __init__(self, stdabb, ueb=False, uebx=10, ueby=4):
        """
        Plot initialisieren.
        Parameter:
            stdabb (object): Berechnungsobjekt der Plotlinien.
            ueb (boolean): True: Uebersicht der Linien zeichnen.
            uebx (int): Wenn, dann Anzahl der vertikalen Stuetzstellen.
            ueby (int): Wenn, dann Anzahl der horizont. Stuetzstellen.
        Erstellt Figur `self.fig`. Setzt `self._interface` = False.
        """
        self.stdabb = stdabb  # Berechnungsobjekt
        self.kick = stdabb.kick  # Startwert K
        self.axcolor = "lightgoldenrodyellow"
        self.fig = plt.figure(figsize=(12, 10))  # Plot-Initialisierung
        self.ueb = ueb
        self.uebx = uebx
        self.ueby = ueby
        self._interface = False

    def interface(self, kmin=0.0, kmax=8.0, drag=True):
        """Erstellt Hauptbereich, Reset Button, Slider K.
        Zeichnet Uebersicht wenn `self.ueb`.
        Verbindet 'button_press_event' mit self.mausklick.
        """
        if self._interface:
            return
        self._interface = True

        # Verknuepfung des button_press_event mit Funktion
        self.fig.canvas.mpl_connect("button_press_event", self.mausklick)
        # Rufe Erstellen `hauptbereich`
        self.draw_hauptbereich()
        # Slider-K
        self.axK = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=self.axcolor)
        self.slider_K = Slider(
            self.axK, r"$Kickst\"{a}rke$", kmin, kmax, valinit=self.kick, dragging=drag
        )
        self.slid_cid = self.slider_K.on_changed(self.kupdate)
        # Einstellung der Achsenmarkierung
        self.axK.set_xticks(np.linspace(0, 8.0, 9))

        # Reset Button
        self.resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.button_r = Button(
            self.resetax, "Reset", color=self.axcolor, hovercolor="0.975"
        )
        self.button_r.on_clicked(self.reset_plot)
        # Uebersicht zeichnen
        if self.ueb:
            self.uebersicht_std()
        print("Kickstaerke = {0:.5f}".format(self.kick))

    def draw_hauptbereich(self):
        """Komplettiert Achse `self.hauptbereich` fuer Darstellung der
        Standardabbildung. Beschriftung und Achsenmarkierung.
        """
        if not self._interface:
            return
        # Subplot mit Bedingungen
        self.hauptbereich = plt.subplot(111)
        self.fig.subplots_adjust(bottom=0.20)

        self.hauptbereich.axis([0, 2 * np.pi, -np.pi, np.pi])  # Plotbereich
        self.hauptbereich.set_title(
            r"$Standardabbildung\ zu\ variabler\ Kickst\"{a}rke$",
            x=0.49,
            y=1.05,
            fontsize=24,
        )
        self.hauptbereich.set_xlabel(r"$Winkel\ \Theta$", fontsize=17)
        self.hauptbereich.set_ylabel(r"$Impuls\ p$", fontsize=17)

        # Einstellung der Achsenmarkierungen
        self.hauptbereich.set_xticks(np.linspace(0, 2 * np.pi, 6))
        self.hauptbereich.set_yticks(np.linspace(-np.pi, np.pi, 6))

    def reset_hauptbereich(self):
        """Loeschen der gezeichneten `lines`."""
        if not self._interface:
            return
        # Reset Hauptbereich
        [line.remove() for line in self.hauptbereich.axes.lines]

    def mausklick(self, event):
        """Pruefung auf Mausklick in gueltigen Bereich. Dann:
        x: Start-Winkel ; y: Start-Impuls.
        Klick: (links: self.plotabb), (rechts: self.multiklick)
        Berechnen und Zeichnen der zugehoerigen Standardabbildung.
        """
        if not self._interface:
            return
        mode = plt.get_current_fig_manager().toolbar.mode
        # Test ob Klick mit linker oder rechter Maustaste und im Koordsys.
        # erfolgt sowie ob Funktionen des Plotfensters deaktiviert sind:
        if event.button in (1, 3) and event.inaxes == self.hauptbereich and mode == "":
            # Uebernehmen der Mausposition
            x = event.xdata
            y = event.ydata
            if not 0 <= x <= 2.0 * np.pi or not -np.pi <= y <= np.pi:
                # Klick ausserhalb Standardabbildung -> return
                print("Ungueltiger Bereich")
                return
            if event.button == 1:  # Linker Mausklick
                print("Start:", (x, y))  # Ausgabe der Startwerte
                self.plotabb(x, y)
            if event.button == 3:  # Rechter Mausklick
                self.multiklick(x, y)

    def plotabb(self, startx, starty):
        """Zeichnen von Startposition und Berechnen und Zeichnen Iterationen."""
        if not self._interface:
            return
        # Berechnen der zwei Listen
        t_array, p_array = self.stdabb.standard_abbildung(startx, starty)
        # Zeichnen in Grafik, jeweils neue Farbe
        self.hauptbereich.plot(t_array, p_array, "o", markersize=2.0)
        del t_array, p_array  # Freigeben
        self.fig.canvas.draw()

    def multiklick(self, x, y, maxr=0.08, numr=4, numw=3):
        """Ein Klick ruft `self.plotabb(x, y)` von mehreren Startpunkten auf.
        Zusaetzliche `numw` aequidistante Punkte auf Umkreisen um Klickpos.
        """
        if not self._interface:
            return
        print("Berechne Iterationen...")
        # Mittelpunkt
        self.plotabb(x, y)
        # numr Kreise mit numw Punkten
        for r in np.arange(numr) + 1:  # von 1 bis numr
            radius = maxr / numr * r
            for w in range(numw):  # von Null bis Winkel-1
                xp = x + radius * np.cos(2.0 * np.pi * w / numw)
                yp = y + radius * np.sin(2.0 * np.pi * w / numw)
                self.plotabb(xp, yp)
        # Ausgabe der Startwerte
        print("{} Multiklicks um".format(numr * numw), (x, y))

    def uebersicht_std(self):
        """Fuellt Phasenraum durch Berechnung von standard_abbildung in Richtung
        x in self.uebx, und in y in self.ueby Schritten.
        Diese werden in Zeichenbereich hinzugefuegt - jeweils in neuer Farbe.
        """
        if not self._interface:
            return
        for x in np.linspace(0.0, 2.0 * np.pi, num=self.uebx, endpoint=False):
            for y in np.linspace(-np.pi, np.pi, num=self.ueby, endpoint=False):
                # Berechnen der zwei Listen
                tlist, plist = self.standard_abbildung(x, y)
                # Zeichnen in Grafik, jeweils neue Farbe
                self.hauptbereich.plot(tlist, plist, "o", markersize=1.8)
        self.fig.canvas.draw()  # Uebersicht-Draw

    def kupdate(self, val):
        """Rechnen und Zeichnen nach Slider Update auf neuen Wert 'val'.
        Resettet mit `self.reset_hauptbereich` und laesst alle Startpositionen
        mittels `self.stdabb.standard_abbildung` neu iterieren.
        """
        if not self._interface:
            return
        sli_freeze(self.slider_K)  # Slider inaktiv
        print("Aktualisiere Iterationen...")
        # Uebernehmen neuer Wert
        self.stdabb.kickneu(val)
        self.reset_hauptbereich()  # Reset Hauptbereich

        for i in range(len(self.stdabb.tupstart)):  # Neu Berechnen
            # Berechnen der zwei Listen
            tlist, plist = self.stdabb.standard_abbildung(
                self.stdabb.tupstart[i][0], self.stdabb.tupstart[i][1], redraw=True
            )
            # Zeichnen in Grafik, jeweils neue Farbe
            self.hauptbereich.plot(tlist, plist, "o", markersize=2.0)

        # Kein Einreihen von neuen Klicks - Frontend aktiv
        plt.pause(0.01)  # Plot Draw+Pause
        print("Neue Kickstaerke = {0:.5f}".format(val))
        sli_release(self.slider_K)  # Slider wieder aktiv

    def reset_plot(self, event):
        """Bereinige vorherige Nutzerinteraktionen."""
        if not self._interface:
            return
        # Reset Hauptbereich
        self.reset_hauptbereich()
        # Reset gespeicherte Startpositionen
        self.stdabb.tupstart = ()
        # Reset Slider
        self.slider_K.reset()
        self.stdabb.kickneu(self.kick)
        print("Reset Plot")
        if self.ueb:  # Uebersicht Neu
            self.uebersicht_std()
        # Zeichnen
        self.fig.canvas.draw()

    def show(self):
        """Zeige alle erstellten Figuren. Warte auf Benutzerinteraktion."""
        plt.show()


def main():
    """Mainfunktion Doppelmuldenpotential.
    Zur Eingabe der Parameter und Starten gewuenschter Realisierungen.
    """
    # Parameter
    K = 4.4  # Kickstaerke der Abbildung
    max_iterationen = 1000  # Iterationen

    # Anfangstext Konsole
    print(__doc__)  # Print Docstring

    realisierung = StdAbb(1.6, max_iterationen)
    plot1 = StdAbbPlot(realisierung)

    plot1.interface()
    plot1.show()


# -------------Main Programm-------------------
if __name__ == "__main__":
    main()

"""Kommentar
Diskretisierungsfehler: Unbeduetend bei Theorie der diskreten Kicks.
Rundungsfehler: Auf sehr kleinen Skalen kann die Abbildung unscharf werden.
Fuer die betrachteten Phaenomene sind 64bit-Gleitkommazahlen ausreichend.

Mit 1000 Iterationen ergeben sich bereits gute Eindruecke
der typischen Koexistenzen regulaerer und chaotischer
Phasenraumbereiche fuer die verschiedenen Kickstaerken.
Selbstaehnliche Strukturen sind besonders um K ca 3-5 gut durch Zoom zu sehen.

Phasenraumbahnen 'springen' oefters zwischen den stabilen Inseln.

Vorwiegend Regulaer:
K=0     :Waagerechte Bahnen - keine Impulsaenderung und ungestoert.
K=0.1   :In Mitte ovale geschlossene Bahnen, nach oben und unten
        ein Uebergang zu flachen Linien.
K=1     :Koexistenz vieler Bewegungsablaeufe, gekippte
        geschlossene Kurven um Zentrum.
Chaotisch/Koexistenzen:
K=2.6   :Grosse Bereiche chaotischer Dynamik/See, 4 erkennbare
        Stabilitaetsinseln ausserhalb des Zentrums.
K=6     :Nahezu vollstaendig chaotisch, vier kleine Inseln
        geschlosener Bahnen sind gut zu erkennen.
K=6.5   :Lediglich zwei Inseln auf p0~0 sind gut erkennbar.
"""
