"""Computational Physics Aufgabe 5.1,  Autor: Martin Roebke 22.05.16
    Quantenmechanik von 1D-Potentialen I
    Berechnung und Zeichung der Loesungen fuer Eindimensionales Potential
    der einheitenlosen Schroedinger-Gleichung.

    Naeherung EF(x<=xmin)|EF(x>=xmax) = 0. Hoehere Energien sind breiter!

    Blaue-Strich-Punkt Kurve : Darstellung des genutzten Potentials.
    Graue Horizontale Strecke : Darstellung der ersten Eigenenergien.

    Einzeichnung der zugehoerigen skalierten Eigenfunktions-Amplituden
    um die jeweilige Eigenenergie.
    Die Loesungen der Eigenfunktionen koennten
    jeweils mit -1 multipliziert sein.
"""

from __future__ import division, print_function  # problemlose Ganzzahl-Division
import numpy as np                              # Arrays, Mathe etc
import matplotlib.pyplot as plt                 # Plotten
import matplotlib as mpl
from scipy.linalg import eigh                   # Geordnete Eigenwerte


def doppelmulde(x=None, A=0.15, string=False):
    """doppelmulde(x=None, A=0.05, string=False)
    Rueckgabe : x**4 - x*x - A*x
    Parameter:
    x : array_like
        Argument der Funktion.
    A : Zahl
        Reeller Parameter der Funktion.
    string : boolean, optional
        Wenn True wird eine Funktionsbeschreibung als String zurueckgegeben.
    """
    if string:
        return r"$x^4-x^2-{}*x$".format(A)
    return x**4-x*x-A*x


def doppelmulde_sym(x=None, string=False):
    """doppelmulde_sym(x=None, string=False)
    Rueckgabe : x**4-x*x
    Parameter:
    x : array_like
        Argument der Funktion.
    string : boolean, optional
        Wenn True wird eine Funktionsbeschreibung als String zurueckgegeben.
    """
    if string:
        return r"$x^4-x^2$"
    return x**4-x*x


def parabel(x=None, a=0.5, b=0, c=0, string=False):
    """parabel(x=None, a=0.5, b=0, c=0, string=False)
    Rueckgabe : a*x*x + b*x + c
    Parameter:
    x : array_like
        Argument der Funktion.
    a, b, c : Zahl
        Reelle Parameter der Funktion.
    string : boolean, optional
        Wenn True wird eine Funktionsbeschreibung als String zurueckgegeben.
    """
    if string:
        s = ""
        if a:
            s += "{}*x^2".format(a)
        if b:
            s += "{:+}*x".format(b)
        if c:
            s += "{:+}".format(c)
        if s == "":
            s = "0"
        return "$"+s+"$"
    return a*x*x + b*x + c


def schiefemulde(x=None, A=1.1, string=False):
    """schiefemulde(x=None, A=1.1, string=False)
    Rueckgabe : x**4-x*x+A*x
    Parameter:
    x : array_like
        Argument der Funktion.
    A : Zahl
        Reeller Parameter der Funktion.
    string : boolean, optional
        Wenn True wird eine Funktionsbeschreibung als String zurueckgegeben.
    """
    if string:
        return r"$x^4-x^2{:+}*x$".format(A)
    return x**4-x*x+A*x


def wellental(x=None, a=0.4, b=0.2, c=1.6, d=0.0, string=False):
    """wellental(x=None, a=0.4, b=0.2, c=1.6, d=0.0, string=False)
    Rueckgabe : a*np.sin(x**2) + b*abs(x)**c + d*x
    Parameter:
    x : array_like
        Argument der Funktion.
    a, b, c, d : Zahl
        Reelle Parameter der Funktion.
    string : boolean, optional
        Wenn True wird eine Funktionsbeschreibung als String zurueckgegeben.
    """
    if string:
        return r"${}\ sin\ x^{{2}} +\ {}|x|^{{{}}}+\ {}\ x$".format(a, b, c, d)

    return a*np.sin(x**2) + b*abs(x)**c + d*x


def matrix_rechnen(potential, stuetz, heff, overwrite_a=False):
    """matrix_rechnen(potential, stuetz, z, overwrite_a=False)
    Berechnet zugehoerige EV, EW einer diskretisierten SG.

    potential : function potential(array-like)
        Funktion des 1D-Potentials.
    stuetz : array-like
        Gleichmaessige Stuetzstellen der Potential-Auswertung.
        !Aufsteigend geordnetes! Array reeller Zahlen.
        Anzahl ergibt Matrixgroesse N.
    heff : reelle Zahl
        h_quer-effektiv des 'Teilchens'.
    overwrite_a : boolean
        Moeglich, die Matrix von der Diagonalisierungsroutine
        ueberschreiben zu lassen. Default:False.

    Normierung: Integral Betragsquadrat in betrachtetem Intervall ergibt 1.
    Rueckgabe: ev, ew
    """
    N = len(stuetz)                                 # Anzahl Stuetzstellen
    step = stuetz[1] - stuetz[0]                    # Schrittweite
    z = heff*heff / (2*step*step)
    # Drei Diagonalen:
    oben = np.diag(np.zeros(N-1) - z, 1)
    mitte = np.diag(potential(stuetz) + 2*z, 0)
    unten = np.diag(np.zeros(N-1) - z, -1)
    matrix_zus = oben + mitte + unten                   # Zusammengefuegt
    ew, ev = eigh(matrix_zus, overwrite_a=overwrite_a)  # Berechnung eigh
    # Vornormierung ist: np.sum(np.abs(ev[:, i])**2) == 1.
    # Als Integral normiert: Int(np.abs(ev[:, i])**2 * dx) == 1.
    return ev / np.sqrt(step), ew


def ef_zeichnen(ax, x_r, ev, ew, skal=1., auto_skal=True, ew_text=False,
                y_alt=False, x_txt=0.03, y_txt=0.007,
                leg=True, legend_loc='lower right', leg_max=10):
    """Routine fuer Zeichnen gewaehlter kleiner Eigenenergien
    und Eigenfunktionen in Achse 'ax'.
    Skalierung der Eigenfunktionen fuer qualitative Sichtbarkeit.
    Optionale Beschriftung der Eigenwerte, und Legende fuer Eigenfunktionen.
    Parameter:
    ax : Achsenobjekt
    x_r : Array Stuetzstellen
    ev : Numerische Eigenvektoren
    ew : Zugehoerige Eigenenergien
    skal : reelle Zahl, optional
        Skalierung der Eigenfunktionen.
        default=1.
    auto_skal : bool, defaul=True
        Falls True wird 'skal' bei mehreren EV auf
        die Haelfte des durchschnittlichen Abstandes der 'ew' gesetzt.
        Hier wird angenommen, dass eine Skalierung mittels der Amplitude der
        ersten Eigenfunktion ausreicht.
    ew_text : bool, defaul=False
        Fuegt EW-Zahl in Achse auf Hoehe der Eigenenergien hinzu.
    y_alt : bool, defaul=False
        Beschriftung der Eigenwerte abwechselnd unter und ueber den
        Eigenwerten positionieren.
    x_txt, y_txt : float, default: x_txt=0.03, y_txt=0.007
        Position der Eigenwert-Beschriftung in Einheiten der Hoehe und Breite
        von 'ax'. y_txt wird mit Hoehe des jeweiligen EW addiert.
    leg : bool, defaul=True
        Falls True fuegt Legende in Pos. 'legend_loc' der ersten 'leg_max'
        Eigenfunktionen ein, geordnet aufsteigend nach Eigenenergie.
    legend_loc : int or string or pair of floats, default='lower right'
        Platzierung 'loc' der Legende. Siehe `plt.legend` fuer Details.
    leg_max : integer,  default=10
        Maximale Anzahl an Eigenfunktionen in der Legende.
    """
    N = len(ew)                                     # N: Laenge gegebener ew.
    if N == 0:
        print("len(ew)==0. ew={}".format(ew))
        return

    if auto_skal == True:  # Skalierung Funktionswerte
        if N >= 2:   # Mittlerer Abstand der EW als Referenz
            h_max = (ew[-1] - ew[0]) / (2.*N)
            skal = h_max / max(abs(ev[:, 0]))
        print("Skalierung Eigenfunktionen in 'ef_zeichnen' :", skal)

    x_interval = [x_r[0], x_r[-1]]                  # Zeichenbereich Eigenwerte
    for i, E in enumerate(ew):                      # Zeichnung EF und EW
        poty = ev[:, i] * skal
        ax.plot(x_r, poty + E, lw=2, label="$\Psi_{{{}}}\ (x)$".format(i))
        # Energie als waagerechte Linie
        ax.hlines(E, x_r[0], x_r[-1], color='0.4', lw=1.5)
    if ew_text:                                     # ax.text der Eigenwerte.
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        text_x = xlim[0] + (xlim[1] - xlim[0]) * x_txt
        if y_alt:
            # Alternierende Textposition
            a = np.empty((N,), int)
            a[::2] = -4
            a[1::2] = 1
            y_txt = a*y_txt
        text_y = ew + (ylim[1] - ylim[0]) * y_txt
        for i in range(N):
            ax.text(text_x, text_y[i], "EW = " + str(round(ew[i], 5)))
    if leg:
        ha, leg = ax.get_legend_handles_labels()
        # Sortierung von unten nach oben. Einschraenkung auf leg_max.
        legend = ax.legend(ha[leg_max::-1], leg[leg_max::-1],
                           loc=legend_loc, fontsize=18)
    ax.set_xlabel("$x$", fontsize='large')
    ax.set_ylabel("$V(x),\ E_n\ [Energie]\qquad \Psi (x)\ (Amplitude)$",
                  fontsize='x-large')


def main(aufgabe='e'):
    """Mainfunktion Quantenmechanik von 1D Potentialen.
    Eingabe der Parameter fuer Berechnung und Zeichnung.
    Durchfuehrung und Benutzerfuehrung zu Aufgabenstellung 5.1.
    """
    # Parameter
    x_halb = None                                   # Halbes Rechenintervall
    y_alt = False                                   # Alternierender Text
    ew_text = True                                  # Eigenwerte beschriften
    leg = True                                      # Legende
    # Ausgabe erster n_ew_p Eigenwerte in Konsole. Setze 0 um zu ueberspringen.
    ew_print = 10
    mpl.rcParams.update({'font.size': 14})
    if aufgabe == 'a':
        potential = doppelmulde
        heff = 0.07
        e_max = 0.15
        x_halb = 2.0                      # Rechenintervall -x_halb bis +x_halb
        nr = 500                                    # Stuetzstellen Rechnen
        axw = [-1.9, 1.7, -0.32, 0.2]               # Zeichenbereich
    elif aufgabe == 'b':
        potential = doppelmulde_sym
        heff = 0.07
        e_max = 0.65
        x_halb = 1.8                     # Rechenintervall -x_halb bis +x_halb
        nr = 500                                    # Stuetzstellen Rechnen
        axw = [-2, 2, -0.3, 0.8]                    # Zeichenbereich
        y_alt = True
    elif aufgabe == 'c':
        potential = parabel
        heff = 1
        e_max = 10
        x_halb = 6                       # Rechenintervall -x_halb bis +x_halb
        nr = 800                                    # Stuetzstellen Rechnen
        axw = [-6, 6, 0, 11]                        # Zeichenbereich
    elif aufgabe == 'd':
        potential = schiefemulde
        heff = 0.07
        e_max = 0.6
        xr_s = -1.8
        xr_e = +1.1
        nr = 650                                    # Stuetzstellen Rechnen
        axw = [-2.1, 1.0, -1.22, .82]               # Zeichenbereich
    elif aufgabe == 'e':
        potential = wellental
        heff = 0.17
        e_max = 1.0
        x_halb = 3.8                      # Rechenintervall -x_halb bis +x_halb
        nr = 600                                    # Stuetzstellen Rechnen
        axw = [-4, 6, 0, 1.15]                      # Zeichenbereich
        ew_text = False                             # Eigenwerte beschriften
        leg = True
    else:
        raise ValueError("Unbekannte Aufgabe: " + aufgabe)

    nx = 300                                        # Potential-Plot
    if x_halb is not None:
        xr_s, xr_e = -x_halb, x_halb

    # Anfangstext Konsole
    print(__doc__)
    print("Berechnungsparameter:\n   h_eff={}, E<{},\n"
          "   Rechenintervall: [{}, {}]\n"
          "   mit N={} Stuetzstellen, dx={}."
          "".format(heff, e_max, xr_s, xr_e, nr, (xr_e-xr_s)/nr)
          )
    # Punkte fuer Zeichen
    drawx = np.linspace(axw[0], axw[1], nx)
    # Punkte fuer Rechnen
    step = (xr_e - xr_s) / (nr + 1)                     # Ortsgitterabstand
    x_r = np.linspace(xr_s + step, xr_e - step, nr)     # Ortsgitterpunkte
    # Erstellung fig
    fig, axes = plt.subplots(figsize=(14, 10))
    axes.set_title("$Potential:\quad$" + potential(string=True), fontsize=24)
    axes.axis(axw)
    # Berechnung Matrix
    ev, ew = matrix_rechnen(potential, x_r, heff, overwrite_a=True)
    ew_cut = ew[ew < e_max]                           # Selektion kleiner EW
    print("Anzahl genutzter Eigenenergien: {}".format(len(ew_cut)))
    # Darstellung der Ergebnisse auf 'axes'.
    ef_zeichnen(axes, x_r, ev, ew_cut, ew_text=ew_text, y_alt=y_alt, leg=leg)
    # Potential:
    axes.plot(drawx, potential(drawx), "b", ls="-.", lw=3)
    # Konsolenausgabe wenn n_ew_p > 0.
    if ew_print:
        print("\nErste {} Eigenenergien:".format(min(ew_print, len(ew))))
        for i, e in enumerate(ew[:ew_print]):
            print("   n={}".format(i), e)
    plt.show()                                      # Darstellung + Interaktion


# -------------Main Programm----------------
if __name__ == "__main__":
    main()                                          # Rufe Mainroutine

"""Kommentare:
a) Obige Wahl der numerischen Parameter:
   Betrachtetes Intervall: Abwaegung zwischen hoher Punktdichte und grossem
    Rechenintervall. Mit Annahme EF(x<xmin)|EF(x>xmax) = 0, bzw.
    n diesen Bereichen unendliches Potential Abschaetzung der Groessenordnung
    der EF bei Wahl eines groesseren betrachteten Intervalls.
    Hier Einschraenkung auf
    Rechenintervall: [-1.7, 1.7] mit N=600 Stuetzstellen, dx~0.0057.
    Relative Abweichung (deutlich) kleiner als 10^-3 zu
    Rechenintervall: [-2.2, 2.2] mit N=5000 Stuetzstellen, dx~0.00088.
   Matrixgroesse N=600:
    Relation von nutzbarer Rechengenauigkeit bei Wahl grosser Zahl zu
    steigendem Resourcenverbrauch.
    Gewaehlte Groessenordnung erreicht in beidem vertratbares Mass, und kann
    in beide Richtungen angepasst werden.

b) Eigenfunktionen:
    Struktur innerhalb des Potentiales wie gemaess Schroedinger-Gleichung
    in 1D Potentialen zu erwarten.
    Jede Eigenfunktion nimmt exponentiell ab, wenn Potential > Eigenenergie.
    So ist es moeglich eine Eigenfunktion mit Extrema in mehreren Potential-
    taelern zu beobachten.
    In 1D-Potentialen sind Eigenwerte nicht entartet, und die Knotenzahl
    (Nullstellen der Eigenfunktionen) steigt mit n (0, 1, 2, 3, ...).
   kleinere heff:
    Abstaende zwischen Eigenwerten und zu Potentialminimum werden kleiner.
   groessere heff:
    Abstaende zwischen Eigenwerten und zu Potentialminimum werden groesser.

c) Fall A=0, Symmetrisches Doppelmuldenpotential.
    Eigenwerte: Fuer hinreichend grosse heff numerisch sichtbar nicht-entartet.
     Auf Hoehen des Potentialhuegels jeweils zwei EW nahe beieinander,
     da lediglich durch Tunnelaufspaltung und Symmetrie getrennt.
     Ueber Potentialhuegel asymptotisch gleichmaessige Abstaende.
    Eigenfunktionen:
     Auf Hoehen des Potentialhuegels jeweils eine symmetrische und eine
     antisymmetrische EF nahe beieinander. Damit auch hier Kontensatz erfuellt.
     Ueber dem Potentialhuegel entsprechend qualitativ asymptotische
     Annaeherung an EF des harmonischen Oszillators.

Vorsicht: Diskretisierung und Naeherung der zweiten Ortsableitung ist gut
wenn EF glatt auf delta-x Skala. Hoehere Loesungen variieren staerker
-> numerische Ungenauigkeit wachst mit hoeheren Energien.
"""
