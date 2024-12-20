"""Computational Physics Aufgabe 6.1,  Autor: Martin Roebke 05.06.16
    Quantenmechanik von 1D-Potentialen II - Zeitentwicklung

Berechnet und zeichnet die Zeitentwicklung eines Gauss'schen Wellen-
paketes in einer Doppelmulde mit Hilfe des Moduls quantenmechanik.py .
Per Mausklick wird ein spezifiziertes Wellenpaket von der x-Position der Maus
in der Basis der Eigenfunktionen dargestellt, und fuer
diskrete Zeiten per Zeitoperator entwickelt dargestellt.

Energetisch niedrige EF sind numerisch gut (2. Ableitung klein).
 => Hoehere EF: Beitrag~Null erwuenscht fuer geringe Fehler in Entwicklung.
Bei neuen Starts waehrend der Zeitentwickelten Darstellung wird lediglich das
jeweils neueste Paket berechnet und verfolgt.
Alle vorherigen Wellenpakete werden verworfen.
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy

import quantenmechanik as qm  # Eigenwerte und -funktionen 1D


def doppelmulde(A: float = 0.05):
    x = sympy.Symbol("x")
    return x**4 - x * x + A * x


def gaussian_wave_paket(xa, sigma=1.0, x0=0.0, heff=1.0, p0=0.0):
    """
    Berechnung Gauss'sches Wellenpaket mit
    mittlerem Ort `x0`, Standardabweichung `sigma`,
    mittlerem Impuls `p0`, effektives hquer `heff` an den Orten `xa`.

    Rueckgabe:
        phi : array-like Wellenpaket.
    """
    # Umwandeln des Ausgangsarrays `x` zu komplex:
    x = np.array(xa, dtype=complex)
    fak = (2 * np.pi * sigma * sigma) ** -0.25  # Vorfaktor
    arg = -(((x - x0) / (2 * sigma)) ** 2)  # Erstes Argument von exp
    if p0 == 0.0:
        phi = fak * np.exp(arg)
    else:  # p0 != 0
        phi = fak * np.exp(arg) * np.exp(1j / heff * p0 * x)
    return phi


class QMZeitentwicklung(object):
    """Berechnung und Zeichnung der Zeitentwicklung eines Gauss'schen Wellen-
    paketes in einem 1D-Potential mit Hilfe des Modules `quantenmechanik`.
    Per Mausklick an Position x wird von dieser Position ein spezifiziertes
    Wellenpaket in der Basis der Eigenfunktionen dargestellt, und fuer
    gewaehlte Zeiten per Zeitoperator entwickelt.
    """

    def __init__(
        self,
        axis,
        potential,
        emax,
        p0,
        heff,
        sigma,
        xr_s,
        xr_e,
        nr,
        tmin=0,
        tmax=10,
        num_t=200,
        title=None,
        fak=0.01,
        phi_color="k",
        update_ef=True,
        wait_dt=0.01,
    ):
        """Initialisierung der Parameter.
        Pruefung auf `mindestens eine gegebene Zeit`, da Zeitentwicklung sonst
        hinfaellig.

        Parameter:
           axis:       AxesSubplot Zeichenbereich.
           potential:  Funktion des Potentials ueber Ort.
           emax:       reelle Zahl; maximale Energie des Plots.
           p0:         reelle Zahl; Startimpuls des Wellenpaketes.
           heff:       reelle Zahl; Zugehoeriges hquer effektiv.
           sigma:      reelle Zahl; Start-Breite Gauss-Wellenpaket.
           xr_s, xr_e: reelle Zahl; Start- und Endpunkt der Potentialauswertung.
           nr:         integer; Anzahl Stuetzstellen in x, Matrixgroesse N.
           tmin, tmax: reelle Zahl; Start- und Endzeit der Darstellung.
           num_t:      integer; Anzahl Zeitschritte der Darstellung.
           title:      string; Titel der Zeichnung.
                        `None` nutzt Default der qm.plot_eigenfunktionen.
           fak:        reelle Zahl; Faktor Eigenfunktions-Skalierung.
           phi_color:  Farbspezifikation fuer `plt.plot`. Phi-Farbe.
           update_ef:  boolean; Passe Erscheinung der Eigenfunktionen an c_n an.
           wait_dt:    Zahl >=0; Plot-Pause zwischen Zeitschritten.
        """
        print(
            "QMZeitentwicklung Initialisierung mit \np0 = {}, heff = {}, "
            "sigma = {}, N = {}.".format(p0, heff, sigma, nr)
        )
        self.axis = axis
        self.potential = sympy.lambdify(
            list(potential.free_symbols), potential, modules="numpy"
        )
        self.emax = emax
        self.p0 = p0
        self.heff = heff
        self.sigma = sigma
        self.xr_s = xr_s
        self.xr_e = xr_e
        self.nr = nr
        self.t = np.linspace(tmin, tmax, num_t)
        assert len(self.t) > 0
        self.title = title
        self.fak = fak
        self.phi_color = phi_color
        self.update_ef = update_ef
        self.wait_dt = wait_dt
        self.startnum = 0
        self.berechnet = False  # Schalter: Berechnung ok.

    def berechnung_esys(self):
        """Diagonalisierung der Hamilton-Matrix.
        Diskretisierung `self.x` des Ortes zu Matrixgroesse `self.nr`.
        Berechnung von `self.ew`, `self.ef` mittels `qm.diagonalisierung`.
        """
        # Berechnung
        self.x, self.dx = qm.diskretisierung(
            self.xr_s, self.xr_e, self.nr, retstep=True
        )
        self.pot_x = self.potential(self.x)
        self.ew, self.ef = qm.diagonalisierung(self.heff, self.x, self.pot_x)
        self.berechnet = True

    def plot(self):
        """Initialisieren und Beschriften des Plotbereiches entsprechend der
        Methode `qm.plot_eigenfunktionen` und der Betragsquadrate def EF.
        Dabei wird erstellt:
         `self.eflines` als Array der geordneten Eigenfunkions-Linien im Plot.
         `self.zeitentw`: Vorbereitung der Linie fuer die Wellenpaket-Darst.
        Verbindet 'button_press_event' mit `self.mausklick`.
        """
        if self.berechnet is False:
            self.berechnung_esys()
        # Plottet: Potential - Eigenwerte Basislinie, Eigenfunktionen
        qm.plot_eigenfunktionen(
            self.axis,
            self.ew,
            self.ef,
            self.x,
            self.pot_x,
            Emax=self.emax,
            fak=self.fak,
            betragsquadrat=True,
            title=self.title,
        )
        if self.update_ef:  # Abspeichern der EF-lines.
            self.num_ef = (len(self.axis.lines) - 1) / 2
            self.ef_iter = range(int(self.num_ef + 1), len(self.axis.lines))
            self.eflines = np.array(self.axis.lines)[self.ef_iter]
        plt.setp(self.axis.title, fontsize=20)  # Passe Titelgroesse an.

        # Bereitstellen der Plotlinie fuer `zeitentw`
        (self.zeitentw,) = self.axis.plot([], [], self.phi_color, linewidth=1.2)
        # Verknuepfung des button_press_event mit Funktion
        figc = self.axis.get_figure()
        figc.canvas.mpl_connect("button_press_event", self._mausklick)

    def _mausklick(self, event):
        """Bei Klick mit linker Maustaste in `self.axis`:
        Erstelle Wellenpaket und zeitentwicklung, und stelle diese dar.
        Fuehrt eine Fortschrittsbeschreibung auf der Konsole.
         Startpunkt des Wellenpaketes `self.phi`durch x-Koordinate
         des Klicks bestimmt.
         Fortlaufende Nummerierung der Wellenpakete mit
         `self.startnum`: Abbrechen aelterer Wellenpakete als das aktuelle.
         Berechne Entwicklungskoeffizienten `self.c`.
         Berechne rekonstruiertes Wellenaket `self.phi_rec`.
         Wenn `self.update_ef`: Visualisiere Beitrag der Eigenfunktionen
                                an `self.phi_rec` als linewidth.
        """
        mode = plt.get_current_fig_manager().toolbar.mode
        # Test ob Klick mit linker Maustaste und im Koordinatensystem
        # erfolgt ist, sowie ob Funktionen des Plotfensters deaktiviert sind:
        if not (event.button == 1 and event.inaxes == self.axis and mode == ""):
            return
        x0 = event.xdata
        self.phi = gaussian_wave_paket(self.x, self.sigma, x0, self.heff, self.p0)
        # Wellenpaket in Eigenfunktionen entwickeln
        self.c = self.dx * np.dot(self.ef.conj().T, self.phi)
        # Rekonstruktion aus Entwicklung nach EF
        self.phi_rec = np.dot(self.ef, self.c)
        normdiff = np.linalg.norm(self.phi - self.phi_rec) * np.sqrt(self.dx)

        # Energieerwartungswert unter Verwendung von c_n:
        self.phi_ew = np.dot(abs(self.c) ** 2, self.ew)

        if self.update_ef:
            # Visualisiere EF-Width
            for i, line in enumerate(self.eflines):
                line.set_linewidth(abs(self.c[i]) * 2.5)

        # Zeitentwicklung berechnen
        self.startnum += 1
        startnum = self.startnum
        print("Paket [{}]: Start an x0 = {:.3f}".format(startnum, x0))
        print("    Differenz in Rekonstruktion: {:.3e}".format(normdiff))
        print(
            "    Zeit-Darstellung von t={} bis {} in {} Schritten."
            "".format(self.t[0], self.t[-1], len(self.t))
        )

        self.zeitentw.set_xdata(self.x)
        for t in self.t:  # Zeitschritte darstellen.
            # Nicht-Nachholen bei Unterbrechung.
            if startnum < self.startnum:
                return
            # Zeitentwicklung auf Phi-Koeffizienten:
            self.phi_rec = np.dot(
                self.ef, self.c * np.exp(-1j * self.ew * t / self.heff)
            )
            self.plot_phi = self.fak * np.abs(self.phi_rec) ** 2 + self.phi_ew
            self.zeitentw.set_ydata(self.plot_phi)
            plt.pause(self.wait_dt)
        print("Paket [{}]: Zeit-Darstellung beendet.".format(startnum))

    def show(self):
        """Zeige alle erstellten Figuren. Warte auf Benutzerinteraktion."""
        plt.show()


def main():
    """Mainfunktion Quantenmechanik von 1D-Potentialen II - Zeitentwicklung.
    Eingabe der Parameter und Starten gewuenschter Realisierungen.
    """
    # Potential
    A = 0.00
    potential = doppelmulde(A)
    emax = 0.3
    p0 = 0.0  # Startimpuls
    # Zugehoeriges hquer effektiv ~ Masse.
    heff = 0.07
    # Start-Breite Gauss-Wellenpaket.
    sigma = 0.1
    # Start- Endbereich Rechnen, V(x<=xr_s)->unendlich, V(x>=xr_e)->unendlich.
    xr_s = -1.7
    xr_e = 1.7
    # Stuetzstellen, Matrixgroesse N
    N = 200
    # Zeitdarstellung von `tmin` bis `tmax` in `num_t` Schritten
    tmin = 0
    tmax = 100
    num_t = 1000
    # Allgemeiner Faktor EF-Skalierung.
    fak = 0.01
    title = "Potential:  ${}$".format(sympy.latex(potential))
    update_ef = True  # Anpassung EF an c_n
    # Plot Figure
    fig, axis = plt.subplots(figsize=(12, 14))

    # Anfangstext Konsole
    print(__doc__)

    # Instanziierung
    rea = QMZeitentwicklung(
        axis,
        potential,
        emax,
        p0,
        heff,
        sigma,
        xr_s,
        xr_e,
        N,
        tmin,
        tmax,
        num_t,
        title,
        fak=fak,
        update_ef=update_ef,
    )
    rea.plot()  # Starte Darstellung
    rea.show()  # Benutzerinteraktion


if __name__ == "__main__":
    main()

"""Kommentar:
Das qm. Tunneln ist kaum klassisch zu erklaeren - wohl aber durch den
Tunneleffekt eines quantenmechanischen Wellenpaketes und die Unschaerfe auf
vergleichsweise kleinen Skalen.

a) Das Paket zerfliesst ohne Startimpuls vom Maximum in die beiden Mulden.
    Es hat anschliessend wenig Aufenthaltswahrscheinlichkeit (AWK)
    an dem Maximum. Ein erwartetes Eindringen in klassisch verbotenen
    Bereich bei starker Abnahme der AWK, sowie Reflexion an
    den Potentialwaenden, sind zu beobachten.
   Start in Potential-Minimum: Die urspruengliche Form wird hauptsaechlich
    beibehalten, schwingt jedoch in ihrer Startmulde nach links und rechts.
    Wieder ist vorherige Beobachtung moeglich, zusaetzlich besteht die
    Moeglichkeit des Tunnelns durch den Potentialhuegel.
    Nach der etwa exp. Daempfung ist in der anderen Mulde fuer eine gewisse
    Zeit nur eine sehr geringe Aufenthaltswahrscheinlichkeit zu erwarten.

b) Das Paket bewegt sich bei p0 = 0.3 zu Beginn vorzugsweise nach "rechts".
    Nach den oben beschriebenen Reflexionen zeigt sich ueber laengeren Zeitraum
    eine aehnliche AWK.
   Insgesamt starten Wellenpakete mit einer groesseren Energieerwartung,
    da der kinetische Term E_kin>0 wird.
    Damit koennen Pakete bei gleicher Startposition staerker
    tunneln als ohne Startimpuls.

c) A=0, p0=0 fuer grosse Zeiten bei Start in einer Mulde:
    Hier ist ein periodisches Wechseln der AWK zwischen beiden Mulden
    zu beobachten. Da das Potential hier symmetrisch verlaeuft,
    bleibt die Gesamtenergie auch dabei erhalten.
    Das Wellenpaket 'tunnelt' periodisch hin und her und wechselt
    von hoher AWK zu niedrigerer AWK beider Mulden.
"""
