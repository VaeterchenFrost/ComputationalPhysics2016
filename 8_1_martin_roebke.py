"""Computational Physics Aufgabe 8.1,  Autor: Martin Roebke 19.06.16
Bestimmung mittlerer Druck in Zeitintervall dt, der auf Seitenflaeche A eines
Quaders von N Teilchen innerhalb ausgeuebt wird.

Darstellung von Ensemble mit R Realisierungen als normiertes Histogramm.

Erwartungswert und Standardabweichung werden abgeschaetzt,
in Konsole ausgegeben. In das Histogramm wird die korrespondierende
Gausskurve gezeichnet.

class IdealesGasEnsemble(object)
    Drucksimulation im idealen Gas.
    Speichert Teilchenzal N, Anzahl Realisierungen R, Zeit dt.
    Laesst weitere statistische Parameter in Realisierungen erstellen.
class Realisierung(object)
    Berechnet aus Teilchenzahl und Zeitintervall dt
    Stoesse mit rechter Wand 'n' und dazugehoerigen Druck.
class Histogramm(object)
    Erstellt und verwaltet ein plt Histogramm mit uebergebenen Daten
    Daten, Anzahl Balken und figsize wird im Konstruktor bereitgestellt.
    Stellt Methoden zeichnen, zeichnen_gauss.
"""

from __future__ import division, print_function  # problemlose Division
import numpy as np                              # Arrays, Mathe etc
import matplotlib.pyplot as plt                 # Plotten
from functools import partial                   # Vorbelegung args.


def gausskurve(x, ew, stdabw):
    """Gibt die analytische Normalverteilung an den Punkten *x*;
    mittels `ew`, `stdabw` zurueck.
    Parameter:
        x: array-like reelle Zahlen, Stuetzstellen.
        ew : reelle Zahl, Erwartungswert der Verteilung.
        stdabw : reelle Zahl, Standardabweichung der Verteilung.
    Return: array-like np.exp(arg) / norm.
    """
    # Normalisierungsfaktor auf Eins:
    norm = stdabw * np.sqrt(2 * np.pi)
    arg = -0.5 * ((x - ew) / stdabw)**2
    return np.exp(arg) / norm


class IdealesGasEnsemble(object):
    """Drucksimulation im idealen Gas.
    Speichert Teilchenzahl N, Anzahl Realisierungen R, Zeit dt.
    Laesst weitere statistische Parameter berechnen und das Ergebnis
    in Histogramm darstellen.
    """

    def __init__(self, N, R, zeitintervall, berechne_zustaende=False):
        """Speichert die Parameter Anzahl Teilchen int(N), Anzahl
        Realisierungen int(R) und betrachtetes Zeitintervall dt.
        Es muessen mehrere Realisierungen entstehen fuer eine sinnvolle
        Statistik.
        berechne_zustaende = True : Rufe Methode self.zustaende_erstellen.
        """
        self.N = int(N)
        self.R = int(R)
        self.dt = float(zeitintervall)

        if N <= 0:
            err_text = "Anzahl N = {} muss positiv sein!".format(self.N)
            raise ValueError(err_text)
        if R <= 1:
            err_text = ("Anzahl R = {} muss zur Auswertung groesser als 1 "
                        "sein!").format(self.R)
            raise ValueError(err_text)

        print("N = {}; R = {}; dt = {}".format(self.N, self.R, self.dt))
        # Alle Zustaende erstellen
        if berechne_zustaende:
            self.zustaende_erstellen()

    def zustaende_erstellen(self):
        """Speichert statistisch berechnete Druecke in Array 'druck' Laenge R.
        Instanziierung jeweils einer Realisierung fuer jeden Druck.
        """
        self.druck = np.zeros(self.R)
        rea = Realisierung(self.N, self.dt)         # Realisierung erstellen
        for i in range(self.R):
            if i:
                rea.set_start()                   # Neue Startwerte
            # Druck berechnen
            rea.berechne_druck()
            # Den Druck pA abspeichern
            self.druck[i] = rea.pA


class Realisierung(object):
    """Berechnet aus Teilchenzahl N und Zeitintervall dt
    Stoesse mit rechter Wand 'n' und den dazugehoerigen Druck.
    """

    def __init__(self, N, dt):
        """Parameter: Anzahl Teilchen N, Zeitintervall dt.
        Erstellt statistisch verteilte self.x0, self.v0.
        """
        self.N = N
        self.dt = dt
        self.set_start()

    def set_start(self):
        """Berechne `self.x0` und `self.v0` aus Zufallszahlen.
        self.x0 = np.random.ranf(size=self.N)
        self.v0 = np.random.normal(size=self.N)
        """
        # Return random floats in the half-open interval [0.0, 1.0).
        self.x0 = np.random.ranf(size=self.N)       # Startorte
        # Draw random samples from a normal (Gaussian) distribution.
        self.v0 = np.random.normal(size=self.N)     # Startgeschwindigkeiten

    def berechne_druck(self):
        """Druck gegen rechte Wand im einheitenlosen Quader nach analytischer
        Formel:
        pA = 2/N/dt * Sum_1^N (Abs(vi) * ni)
        """
        self.stoesse = self.reflexionen()
        self.pA = 2/self.N/self.dt * np.dot(abs(self.v0), self.stoesse)

    def reflexionen(self, ort=None):
        """Funktion die fuer alle N Teilchen die Zahl der Reflexionen an
        rechter Wand in Abhaengigkeit der Parameter zurueckgibt.

        Parameter:
            ort, array-like:
            Entfaltete Position der Teilchen zu gewisser Zeit.

        if ort is None: ort = self.x0 + self.dt*self.v0
        Mit
            self.x0 : array_like
                Startposition der einzelnen  Teilchen
            self.v0 : array_like
                Geschwindigkeit der einzelnen Teilchen.
            self.dt : reelle Zahl
                Zeitintervall, innerhalb dessen Stoesse gezaehlt werden.
        Bsp: ort = [-3.2, -2.1, -1, -0.9, -0.5, 0.2, 1.1, 1.5, 2.9, 5.3]
               n = [   2.    1.  1.    0.    0.   0.   1.   1.   1.  3.]
        """
        if ort is None:
            # Teilchen nach Zeit t fortbewegt mit jeweiliger Geschwindigkeit v.
            ort = self.x0 + self.dt*self.v0
        # Abbildung auf vollstaendige Bahnen und positive Werte der Reflektionen
        n = abs(np.trunc(ort))
        # Aufrunden der Healfte -> Nur Stoesse mit rechter Wand zaehlen.
        n = np.ceil(n/2)
        return n


class Histogramm(object):
    """Erstellt und verwaltet ein plt Histogramm mit uebergebenen Daten.
    Daten, Anzahl Balken und Plotbereich werden im Konstruktor bereitgestellt.
    Stellt Methoden: self.zeichnen, self.zeichnen_gauss.

    self.zeichnen muss VOR anderen Plots gerufen werden,
    da es mit self.bins die Punkte der Abszissenachse festlegt!
    num_bins: Wert fuer Gesamtzahl an Balken.
        Zu grosse Werte verlieren Uebersichtlichkeit,
        Zu kleine Werte fuehren ggf. zu grossem Informationsverlust.
        Default : 70
    """

    def __init__(self, axis, daten, num_bins=70):
        """Erstellt Figure mit self.figsize, und einem Subplot.
        daten : array_like, Darzustellende Daten.
        Das Histogramm wird ueber das flattened Array berechnet.
        num_bins : Natuerliche Zahl, Anzahl der Balken im Histogramm
        figsize : Tupel (Breite, Hoehe) der gewuenschten Fenstergroesse
        """
        self.ax = axis                              # Zeichenachse
        self.yarr = daten                           # Array der Daten
        self.num_bins = num_bins                    # Anzahl Bins

    def zeichnen(self):
        """Zeichnet normiertes Histogramm der Daten auf eigene Achse.
        Schreibt Titel und Achsenlabel.
        Erzeugt: self.bins, self.title
        """
        # Das Histogramm der Daten
        werte, self.bins = self.ax.hist(self.yarr, self.num_bins,
                                        density=True, facecolor='green', alpha=0.5)[:2]
        bin_breite = self.bins[1] - self.bins[0]
        # Titel
        self.title = "$Das\ Histogramm\ des\ Druckes\ p_A\
        \qquad Binbreite\ =\ {0:.5f}$".format(bin_breite)
        # Beschriftung
        self.ax.set_title(self.title, fontsize=24, y=1.02)
        self.ax.set_xlabel("$Druck$"" p", fontsize=22)
        self.ax.set_ylabel(r"$Normierte\ H\"{a}ufigkeitsdichte$", fontsize=22)

    def zeichnen_gauss(self, gauss):                # 'Gaussglocke'
        """Plottet mitgegebene Funktion `gauss` auf eigene Achse.
        Parameter:
            gauss: function, einzig abhaengig von erstem! Parameter.
        """
        try:
            self.ax.plot(self.bins, gauss(self.bins), 'm--', lw=1.5)
        except AttributeError:
            print("Zuvor wird Zeichenmethode self.zeichnen() benoetigt!")
            raise

    def show(self):
        """Zeige alle erstellten Figuren. Warte auf Benutzerinteraktion."""
        plt.show()


def main():
    """Mainfunktion Druckmessung Ideales Gas.
    Eingabe der Parameter und Starten gewuenschter Realisierungen.
    """
    benutzerfuehrung = ("Computational Physics Aufgabe 8.1,  "
                        "Autor: Martin Roebke 19.06.16\n"
                        "Bestimmung des mittleren Druckes in Zeitintervall dt, \n"
                        "der auf Seitenflaeche A eines Quaders von N Teilchen ausgeuebt wird.\n"
                        "Darstellung von Ensemble mit R Realisierungen als normiertes Histogramm.\n"
                        "Erwartungswert und Standardabweichung des Ensembles werden berechnet,\n"
                        "in Konsole ausgegeben. In das Histogramm wird die korrespondierende \n"
                        "Gausskurve gezeichnet." "\n")

    print(benutzerfuehrung)

    # Anzahl Teilchen
    N = 8
    # Anzahl der Realisierungen mit je N Teilchen
    R = 10**4
    # Zeitschritt
    dt = 4
    # Erstellt Figure mit self.figsize, und einem Subplot.
    fig, ax = plt.subplots(figsize=(14, 10))
    # Ensemble Instanziieren und Druck-Array berechnen.
    ensemble1 = IdealesGasEnsemble(N, R, dt, berechne_zustaende=True)
    p_arr = ensemble1.druck

    # Statistik und Histogramm-Plot erstellen.
    # Da abgeschaetzter Mittelwert: 1/(N-1) als Faktor in Standardabweichung
    # mittels ddof=1.
    erwartungswert = np.mean(p_arr)
    std_abweichung = np.std(p_arr, dtype=np.float64, ddof=1)
    # Gausskurve und Zeichnung des Histogramms erstellen.
    gauss = partial(gausskurve, ew=erwartungswert, stdabw=std_abweichung)
    # Erstellen und Modifizieren eines Histogramm
    histogramm = Histogramm(ax, p_arr)
    histogramm.zeichnen()
    histogramm.zeichnen_gauss(gauss)
    # Konsolenausgabe der verwendeten Plot-Parameter
    print("Erwartungswert = ", erwartungswert)
    print("Standardabweichung = ", std_abweichung)
    print("Anzahl Bins: ", histogramm.num_bins)

    histogramm.show()                               # Benutzerinteraktion


# -------------Main Programm----------------
if __name__ == "__main__":
    main()                                          # Rufe Mainroutine

"""Kommentar:
Einfluss delta t: Bei ganz kleinen Werten -> Alle Null Reflexionen.
(a) Welche Form, EW und STABW der Verteilung erhalten Sie fuer N=60?

N=80:
Das Maximum liegt bei etwas kleineren Werten als der Erwartungswert
der Druckverteilung. Steilerer Anstieg am linken Rand als die Gausskurve,
hingegen flacherer Abfall an rechtem Rand.
ErwW : weicht in vierter Kommastelle von 1.0 ab.
STABW: Rund 0.15 bis 0.17
      (bei N=6000: 0.018)

Vergleich: N=8
Das Maximum liegt stark zu kleineren Druck-Werten verschoben (bei 0.7-0.8).
Deutlich steiler Anstieg an rechter Flanke und langsamere Abnahme
bei grossen Druckwerten im Vergleich zu N=80 und
dem prinzipiellen Verlauf der Gauss-Kurve.
ErwW : weicht meist in dritter Kommastelle von 1.0 ab. - ungenauer
STABW: Rund 0.5  - deutliche Schwankungen.
-> geringere Schwankungen bei groesserer Teilchenzahl N erwartet.

(b) Welche Form, EW und STABW d.V. erwarten Sie fuer N=6*10**23?
Diese grosse Teilchenanzahl (des Mols) wuerde die Vorhersagen fuer
makroskopische Systeme sehr genau erfuellen, da jede Realisierung deutlich
aehnlichere Druckwerte im Vergleich aller Realisierungen erzeugen wird.
Statistische Abweichungen verringern sich hier mit Groesse der
'Stichprobe' Teilchenanzahl.
Die erwartete Form konvergiert gegen eine sehr schmale Gausskurve,
die aus Entfernung einem schmalen Balken um p=1 gleicht.
-> Nach Gesetz der grossen Zahlen gegen Gauss.
Es geht der Erwartungswert des einheitslosen Drucks
gegen Eins, die Standardabweichung gegen Null.
"""
