"""Computational Physics Aufgabe 9.1,  Autor: Martin Roebke 26.06.16
Numerische Simulation von Teilchenorten bei Diffusion + Drift und Absorption.
Darstellung der Ergebnisse fuer R Realisierungen ueber t=1 bis tmax in norm.
Histogramm und graphischer Auswertung mittels theoretischer Funktionen und
statistischer Parameter Norm, Erwartungswert und Varianz.

Die Langevin-Gleichung : x(t+dt)= x(t) + v*dt + wurzel(2*D*dt)*Gauss_normiert

::Linker Mausklick in einen der vier Plotbereiche startet Zeititeration::
"""

from __future__ import division, print_function     # problemlose Division
import numpy as np                                  # Arrays, Mathe etc
import matplotlib.pyplot as plt                     # Plotten
import matplotlib as mpl                            # MPL
import warnings                                     # Warnungs-Handler


def gausskurve_v(x_punkte, ew, vari):
        """Gibt die analytische Normalverteilung an den Punkten *x_punkte*;
        mittels Erwartungswert *ew* und Varianz *vari*. Normiert auf 1.
        """
        # Normalisierungsfaktor auf Integral=Eins:
        norm = np.sqrt(0.5 / np.pi / vari)
        return np.exp(-0.5 * (x_punkte - ew)**2 / vari) * norm


class Aufgabe_Neun(object):
    """Berechnung und Darstellung eines Systems von punktfoermigen Teilchen
    mit gleichfoermigem Drift und Diffusion;
    'Rechts' am System werden Teilchen mit `x>=xabs` absorbiert.
    Ein Ensemble enthaelt mehrere unabhaengige Teilchen.
    Die Position der Teilchen zu ganzzahligen Zeiten wird, nach Iteration der
    Langevin-Gleichung in Zeitschritten `dt`, als Histogramm zusammengefasst,
    und in weiteren Plots statistisch ausgewertet.

    Der Zeitverlauf wird mittels linker Maustaste gestartet,
    und kann mittels rechtem Mausklick pausiert und fortgefuehrt werden.

    Bietet:
        __init__(self, x0, vdrift, diffusion, pos_xabs, iteilchen, tmax,
                 dt=.01, binbreite=0.5, xlim=[-25, 20], seed=None)
        __call__(self): Startet Berechnung und Darstellung.
        set_axes(self, axes): Setze self.axes
        set_dt(self, dt): Setze self.rechensteps und self.dt
        ort_iteration(self, xt): Return Langevin-Gleichung-Schritt.
        show(self): Zeige erstellten Figuren. Warte auf Benutzerinteraktion.
    Privat genutzte Methoden:
        _plotumgebung(self): Richte Zeichenbereiche ein.
        _klick(self): Verarbeitet Mausklick in self.axes.
        _plot_handler(self): Handler fuer Darstellung, self._zeitentwicklung().
        _zeitentwicklung(self): Starte ein neues Ensemble an self.x0.
        _darstellung(self, zeit): Wertet self.zustaende zu aktueller Zeit aus.
    """
    def __init__(self, x0, vdrift, diffusion, pos_xabs, iteilchen, tmax, axes,
                 dt=.01, binbreite=0.5, xlim=[-25, 20], seed=None, numz=250):
        """Initialisiert Parameter. Konsolenausgabe der Variablenwerte.
        x0 : reelle Zahl, x(t=0) der Teilchen.
        vdrift: reelle Zahl, Driftgeschwindigkeit.
        diffusion: reelle Zahl, Diffusionskonstante (D).
        pos_xabs: reelle Zahl, Position des absorbierenden Randes.
        iteilchen: reelle Zahl, Anzahl an Realisierungen in einem Ensemble.
        tmax: reelle Zahl, Maximale Zeit der Berechnung - mindestens 1.
        axes: array-like :class:`~matplotlib.axis.Axis, mindestens 4 Eintraege.
        dt=.01: positive Zahl; berechnet Zahl Rechenschritte: int(np.ceil(1/dt))
        binbreite=0.5: positive Zahl, regelt angestrebte Binbreite von plt.hist.
        xlim=[-25, 20]: x-Achsenbereich der Histogramm-Plot Achse.
        seed=None: Startwert fuer `RandomState`. Wenn None: nicht gesetzt.
        numz = 250: Stuetzpunkte der theoretischen Wahrscheinlichkeitsdichten.
        Bemerkungen:
         Es muessen mehrere Realisierungen entstehen fuer eine
         sinnvolle Statistik. Der absorbierende Rand muss 'rechts'
         von der Startposition der Teilchen liegen.
        """
        self.x0 = float(x0)
        self.xabs = float(pos_xabs)
        if self.x0 > self.xabs:
            err_text = ("OBdA soll der absorbierende Rand 'rechts' von "
                        "der Startposition x0 liegen!")
            raise ValueError(err_text)
        self.vdrift = float(vdrift)
        self.diffusion = float(diffusion)
        self.nt0 = int(iteilchen)
        if self.nt0 <= 1:
            err_text = ("Anzahl iteilchen = {} muss zur Auswertung "
                        ">=1 sein!").format(self.nt0)
            raise ValueError(err_text)
        self.tmax = int(tmax)
        assert self.tmax > 0                 
        self.set_axes(axes)                     # Zeichenbereiche.
        self.set_dt(dt)                         # dt und Rechenschritte.
        self.x_min = min(xlim)
        self.x_max = max(xlim)
        assert self.x_min < self.x_max          # Zeichenbereich groesser 0.
        self.binb = binbreite
        assert self.binb > 0                    # Binbreite groesser 0.
        self.seed = seed                        # np.random seed

        print("Initialisierung mit Parametern:")
        print("x0= {}, xa= {}, vdrift= {}, diffusion= {},tmax= {}, dt= 1/{}"
            "\n{} Realisierungen.".format(self.x0, self.xabs, self.vdrift,
            self.diffusion, self.tmax, self.rechensteps, self.nt0))

        self.startnum = 0                       # Gestartete Ensembles
        self.ew_min = -1                        # Vorbelegung Zeichenparameter
        self.ew_max = 1
        self.var_max = 10
        self.cabs = 'red'                       # Vorbereitete Farbe fuer Absor.
        self.ls_t = 'b-'                        # Vorb. Linestyle der th. Kurven
        print("\nRote Werte: Messungen an Ensemble mit Absorption.")
        print("Blaue Werte: Theoretische Kurven von Ensemble ohne Absorption.")
        self.pause = False
        self.plot_aktiv = False
        self.x_gd = np.linspace(self.x_min, self.x_max, numz)
        self.x_gda = np.linspace(self.x_min, self.xabs, numz)

    def __call__(self):
        """Vorgefertigte Interaktion mit vorbereiteten Parametern."""
        if self.seed is not None:       # random.seed fuer Reproduzierbarkeit.
            np.random.seed(self.seed)
            print("Pseudo-Zufallszahlen mit seed={}.".format(self.seed))
        self._plotumgebung()
        figc = self.ax_hist.get_figure()
        figc.canvas.mpl_connect('button_press_event', self._klick)

    def set_axes(self, axes):
        """Setze self.axes. Kontrolliert auf korrekte Anzahl.
        Setzt benoetigte Achsenbereiche zurueck auf Default-Einstellungen.
        """
        self.axes = axes.flatten()[:4]      # Nutze axes[0...3]
        if len(self.axes) < 4:
            raise ValueError("`self.axes` braucht 4 Bereiche.")
        [a.cla() for a in self.axes]        # Bereiche aufraeumen.
        self.ax_hist = self.axes[0]         # Benennung.
        self.ax_norm = self.axes[1]
        self.ax_ew = self.axes[2]
        self.ax_var = self.axes[3]

    def set_dt(self, dt):
        """Setze self.rechensteps und self.dt=1/self.rechensteps. """
        self.rechensteps = int(np.ceil(1 / dt)) # Aufrunden um 0 zu vermeiden.
        self.dt = 1 / self.rechensteps
        
    def ort_iteration(self, xt):
        """Berechnung von Teilchenorten mittels Langevin-Gleichung fuer einen
        Zeitschritt von t zu t + `self.dt`.
        xt : array-like, Startposition.
        Nutzt : self.vdrift, self.diffusion, self.dt
        Rueckgabe : einmal-iterierte Position aller Teilchen aus `xt`.
        """
        di = (2.*self.diffusion*self.dt)**0.5 * np.random.normal(size=len(xt))
        xtplus = xt + self.vdrift*self.dt + di
        return xtplus

    def show(self):
        """Zeigt alle erstellten Figuren. Wartet auf Benutzerinteraktion."""
        plt.show()

    def _plotumgebung(self):
        """Richte benoetigte Achsen fuer eine Darstellung ein.
        Behandelte Achsen: self.ax_hist, self.ax_norm, self.ax_ew, self.ax_var.
        """
        # Achsenbereiche
        self.ax_hist.axis([self.x_min, self.x_max, 0., 0.25])
        self.ax_norm.axis([0., self.tmax, 0., 1.1])
        self.ax_ew.axis([0., self.tmax, self.ew_min, self.ew_max])
        self.ax_var.axis([0., self.tmax, 0, self.var_max])

        # Linie Gauss mit Drift und Diffusion
        self.line_gd, = self.ax_hist.plot([], [], self.ls_t, alpha=0.7)
        self.line_gd.set_data(self.x_gd, np.nan*np.ones_like(self.x_gd))
        # Linie Gauss + Absorption
        self.line_gda, = self.ax_hist.plot([], [], c='g', lw=2)
        self.line_gda.set_data(self.x_gda, np.nan*np.ones_like(self.x_gda))
        # Label
        self.ax_hist.set_xlabel("$Position\ im\ Raum$")
        self.ax_hist.set_ylabel("$Normierte\ Dichte\ der\ Teilchen$")
        self.ax_norm.set_xlabel("$Zeit$")
        self.ax_norm.set_ylabel("$Norm$")
        self.ax_ew.set_xlabel("$Zeit$")
        self.ax_ew.set_ylabel("$Erwartungswert$")
        self.ax_var.set_xlabel("$Zeit$")
        self.ax_var.set_ylabel("$Varianz$")
        # Markierungen
        self.ax_hist.axvline(self.x0, linewidth=0.6, c='black', ls="--")
        self.ax_hist.axvline(self.xabs, linewidth=2, c=self.cabs, ls=":")
        # Text
        title = r"$Darstellung\ \ Ortsverteilung$"
        self.ax_hist.set_title(title, fontsize=18, y=1.03)
        self.ax_norm.set_title("-> Mausklick startet Ensemble an x0 <-", y=1.03)

    def _klick(self, event):
        """Klick-Handler fuer Interaktion mit `self.axes`.
        event : 'button_press_event'
        Linksklick und nicht `self.pause`: ruft self._plot_handler() .
        Rechtsklick und aktiver Plot: toggelt `self.pause` und pausiert bei
                                      aktivierter `self.pause`.
        """
        # Test ob Funktionen des Plotfensters deaktiviert sind:
        mode = plt.get_current_fig_manager().toolbar.mode
        if not (mode=='' and event.inaxes in self.axes):
            return
        if event.button==3 and self.plot_aktiv: # Pause
            self.pause = not self.pause             # Aendere Pausenstatus.
            if not self.pause: print("Berechung fortgesetzt."); return
            print("Berechnung durch Benutzer pausiert "
                  "- Fortsetzen durch Rechtsklick.")
            while self.pause:
                plt.pause(0.1)
            return
        if event.button==1 and not self.pause:       # Berechnung Starten.
            self._plot_handler()

    def _plot_handler(self):
        """Handler fuer Darstellung mittels `self._zeitentwicklung()`."""
        self.plot_aktiv = True
        ret = self._zeitentwicklung()
        self.plot_aktiv = False
        if ret[0] == 0:      # Normaler Ablauf
            print("[{}]: Berechnung ist beendet.".format(ret[1]),
                  "Teilchen zu Tmax:", len(self.zustaende))
        elif ret[0] == 1:    # Abgebrochen vor tmax
            print("[{}]: Berechnung unterbrochen.".format(ret[1]))
        elif ret[0] == 2:    # Ein frueher gestartetes Ensemble unterdrueckt."
            print("[{}]: Alte Berechnung unterdrueckt.".format(ret[1]))

    def _zeitentwicklung(self):
        """Starte ein neues Ensemble an `self.x0` von `self.nt0` Teilchen mit
        Startnummer: self.startnum + 1.
        Konstruiere Zeitschritte von t=0 bis `self.tmax`.
        Abbruch wenn 0 oder 1 Teilchen uebrig.
        Berechnung von `self.rechensteps` Zwischenschritten der Bewegung fuer
        graphische Darstellung an t = 1, 2, 3...

        Return : error[0: ok;1: zu wenig Teilchen;2: alter Start], Startnummer
        """
        self.startnum += 1
        startnum = self.startnum
        print("[{}]: Starte Ensemble".format(startnum))
        # Bereinige Statistik-Parameter vorheriger Darstellung
        del self.ax_norm.lines[:]
        del self.ax_ew.lines[:]
        del self.ax_var.lines[:]
        self.line_normt, = self.ax_norm.plot([], [], self.ls_t)
        self.line_ewt, = self.ax_ew.plot([], [], self.ls_t)
        self.line_vart, = self.ax_var.plot([], [], self.ls_t)

        self.zustaende = self.x0*np.ones(self.nt0)  # Startpositionen
        for i in range(self.tmax):                  # Zeitschleife
            # Neue Orte
            for j in range(self.rechensteps):       # Zeitschritte umsetzen
                # Nicht-Nachholen bei Unterbrechung.
                if startnum < self.startnum: return (2, startnum)
                zustaende = self.ort_iteration(self.zustaende)
                self.zustaende = zustaende[zustaende<self.xabs]
            # Ausnahme
            if len(self.zustaende) <= 1:              # Keine Statistik moeglich
                print("Warnung: Nur {} Teilchen zu t={} uebrig!"
                      "".format(len(self.zustaende), i+1)
                     )
                self.ax_hist.patches = []           # Histogramm leeren.
                self.line_gd.set_ydata(0)
                self.line_gda.set_ydata(0)
                plt.pause(0.01)
                return (1, startnum)
            # Zeichnen
            self._darstellung(zeit=i+1)             # Graphische Darstellungen.
            plt.pause(0.01)
        return (0, startnum)                        # Zeitdurchlauf beendet.

    def _darstellung(self, zeit):
        """Wertet `self.zustaende` zu aktueller Zeit aus.
        Darstellung auf
         self.ax_norm: Norm des Ensembles, ohne Absorption: 1.
         self.ax_ew: Mittelwert, ohne Absorption: self.x0 + self.vdrift*zeit.
         self.ax_var: Varianz, ohne Absorption: 2. * self.diffusion*zeit.
        Parameter:
            zeit : reelle Zahl groesser Null.
        Erweitert Plotbereiche von `self.ax_var` und `self.ax_ew`.
        Erstelle y-Werte der Theorie-Kurven (self.line_gd, self.line_gda).
        Anpassung Bin-Parameter `self.akt_bins`.
        Plotte damit Histogramm auf *self.ax_hist*.
        """
        laenge = len(self.zustaende)
        # Betrachtete Groessen und theoretischer Wert an t=`zeit`.
        norm, norm_t = (laenge / self.nt0, 1.)
        erw = np.mean(self.zustaende)
        erw_t = self.x0 + self.vdrift*zeit
        vari = np.var(self.zustaende, ddof=1)       # N-1, da EW experimentell.
        vari_t = 2. * self.diffusion*zeit
        # Plotbereich aktualisieren
        if self.var_max < max(vari, vari_t): self.var_max = max(vari, vari_t)
        if self.ew_max < max(erw, erw_t): self.ew_max = max(erw, erw_t)
        if self.ew_min > min(erw, erw_t): self.ew_min = min(erw, erw_t)
        self.ax_var.set_ybound(upper=self.var_max+1)
        self.ax_ew.set_ybound(lower=self.ew_min-1, upper=self.ew_max+1)
        # Verteilungsparameter und erwarteten Grenzwert zeichnen.
        self.ax_norm.plot(zeit, norm, ".", c=self.cabs)
        self.line_normt.set_data([0, zeit], [1, 1])
        self.ax_ew.plot(zeit, erw, ".", c=self.cabs)
        self.line_ewt.set_data([0, zeit], [self.x0, erw_t])
        self.ax_var.plot(zeit, vari, ".", c=self.cabs)
        self.line_vart.set_data([0, zeit], [0, vari_t])
        # Erstelle y-Werte der Theorie-Kurven P(x).
        kurve_D = gausskurve_v(self.x_gd, erw_t, vari_t)
        ew_spiegel = 2*self.xabs - self.x0 + self.vdrift*zeit
        kurve_DA = (gausskurve_v(self.x_gda, erw_t, vari_t)
                    - gausskurve_v(self.x_gda, ew_spiegel, vari_t)
                    * gausskurve_v(self.xabs, erw_t, vari_t)
                    / gausskurve_v(self.xabs, ew_spiegel, vari_t))

        self.line_gd.set_ydata(kurve_D)             # Setze neue Theorie-Vert.
        self.line_gda.set_ydata(kurve_DA)

        # Bestimmung Bin-Parameter.
        span = max(self.zustaende) - min(self.zustaende)
        self.akt_bins = 1 + round(span / self.binb)
        dbin = span / self.akt_bins
        weights = np.ones(laenge) / self.nt0 / dbin
        # Das Histogramm der Orte.
        self.ax_hist.patches = []                   # Histogramm zuruecksetzen
        self.ax_hist.hist(self.zustaende, int(self.akt_bins), facecolor = 'green',
                          alpha = 0.5, weights=weights)

def main():
    """Mainfunktion fuer Numerische Diffusion mit Drift und Absorption.
    Festlegen der Parameter und Starten gewuenschter Realisierungen.
    Unterdruecke `mplDeprecation` waehrend Durchlauf.
    Dies hat hier nur Einfluss auf die Konsolenausgabe des Programmes.
    """
    warnings.filterwarnings("ignore", category=mpl.cbook.mplDeprecation)
    mpl.rcParams.update({'axes.labelsize': 14, "axes.labelpad": 3.0})
    print(__doc__)                              # Anfangstext Konsole

    # Bewegung
    x0 = -5                                     # Startposition der Teilchen
    vdrift = 0.2                                # Driftgeschwindigkeit
    diffusion = 1.5                             # Diffusionskonstante
    pos_xabs = 15                               # Position absorbierender Rand
    # Numerik : Anzahl Realisierungen, maximale Zeit und Zeit fuer Rechenschritt
    R = 10**4
    tmax = 80
    dt = 0.01
    # Zeichenbereich
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.93,
                        wspace=0.16, hspace=0.23)
    # Instanziierung von `Aufgabe_Neun` mit obigen Parametern.
    Rea = Aufgabe_Neun(x0, vdrift, diffusion, pos_xabs, R, tmax, axes, dt)
    Rea()                                       # Starten der Darstellung
    Rea.show()                                  # Anzeigen der Plots

#-------------Main Programm----------------
if __name__ == "__main__":
    main()                                      # Rufe Mainroutine

"""Kommentar:
a) Absorbierender Rand
 Sichtbar nach ca t=10, wenn die ersten Teilchen durch Drift und Diffusion
 an die Kante bewegt wurden.
 Offensichtlich besteht bei x>=xabs keine Wahrscheinlichkeitsdichte, und
 Norm faellt ab; Erwartungswert erreicht xabs nicht sondern bleibt geringer;
 Varianz wird kleiner, da sich das Ensemble nicht ueber xabs hinaus bewegen
 kann, und somit eingeschraenkter ist. Im Verlauf des Histogrammes fuer
 x<=x0 ist hier kaum Veraenderung durch Absorption zu beobachten.
b) dt=1
 Es steigt in der Langevin-Gleichung der Einfluss des Drifts~dt gegenueber der
 Diffusion~dt^0.5 im Vergleich zu dt=0.01.
 Sichtbar wird dies bei entsprechenden Zeiten an:
  Schwach abfallende Norm / Weniger aborbierten Teilchen.
  Ebenso hoehere Histogramm-Werte bei x~xabs.
  Dadurch Groesserer Erwartungswert und Varianz.
 Anschaulich: Weniger Rechenschritte fuehren zu 'seltenerer Absorption'.
c) vdrift=0.5
 Schnellere Bewegung nach 'rechts', Abweichung von blauen Linien ab t>6.
 Deutlich staerkere Abnahme der Norm (um ca 75%).
 Erwartungswert und Varianz zeigen fuer t->40 nahezu waagerechten Verlauf;
 Der Erwartungswert erreicht dann groessere Werte ~ +6 (-0.6),
 die Varianz bleibt bei deutlich geringeren Werten um 35 (69).
"""
