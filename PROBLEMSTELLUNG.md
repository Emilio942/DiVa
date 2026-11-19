# Problemstellung: Die fünf offenen Probleme des DiVa-Programms

*Stand: 15. November 2026 · Begleitdokument zu `REPORT.md` und `diva_mathematik.tex`*

> **Status-Update (19. November 2026):** Alle fünf unten formulierten Probleme wurden analytisch gelöst und durch unabhängige Verifikationstests bestätigt. Vollständige Herleitungen: **`HERLEITUNG.md`** sowie Abschnitt 12 von **`diva_mathematik.tex`**; reproduzierbares Verifikationsskript: **`verify_theory.py`** (`results/theory_verification.json`).

---

## 0. Lagebericht in vier Sätzen

Das numerische Programm ist **abgeschlossen**: Die kritische Fläche wurde vermessen,
die ursprüngliche Konjektur elementweise beantwortet (quadratische σ²-Form bestätigt,
beide Koeffizientengesetze widerlegt und ersetzt), und alle Gesetze tragen Fehlerbalken
und getestete Gültigkeitsbereiche. Was fehlt, ist **Theorie**: fünf scharf formulierte
Probleme, für die wir jetzt Datenrestriktionen haben, aber noch keine Herleitung.
Diese Restarbeit ist **keine GPU-Zeit** mehr (von den 10 Stunden wurden ~1,5 verbraucht),
sondern menschliche Rechenzeit auf dem Papier — daher die großen Schätzungen.

### Was jede erfolgreiche Theorie reproduzieren muss (die Fakten-Tabelle)

| Nr. | Gemessenes Gesetz | Wert |
|---|---|---|
| F1 | Untere Kante | $\eta_{\mathrm{lo}} = a_\sigma\,\sigma^2$, Achsenabschnitt $\approx 0$ |
| F2 | Rausch-Koeffizient | $a_\sigma = 3d+k$ (±5 %, 12 Konfigurationen, float64: $7{,}00{\pm}0{,}14$ bei $(1,2)$) |
| F3 | Lineare Form-Moden-Rate | $\gamma \approx 1{,}0$ **universell** — unabhängig von $(k,d)$ (AR(1)-Heuristik damit tot) |
| F4 | Oberes Fensterende | $\eta_{\mathrm{hi}} \approx 2{,}036$, universal für $k\ge2$; $k=1$: metastabil 2,07–2,55 |
| F5 | Marginal-Plateau | $\lambda_{\mathrm{det}}\equiv 0$ für $d\ge4$ über das ganze Fenster; $d=2$: kontrahierend; abrupter Sprung $0\to+0{,}4$ bei $\eta_{\mathrm{hi}}$ |
| F6 | Anker-Grenzfunktion | $B(x)$, $x=\lambda_Wp_{\mathrm{fix}}$: kein Durchgang für $x\lesssim0{,}015$; Maximum $2{,}10$ bei $x^\ast\approx0{,}075$; Abfall $\approx-1{,}1/x$ |
| F7 | Hoeffding-Signatur | $\zeta_1^\ast \to 1/4$ ($0{,}2558$ bei $N=1024$); $\zeta_1=0$ in beiden reinen Phasen |
| F8 | Gültigkeitsbereich von F2 | ±5 % für $d\le16$ bei $N\ge64$; bei $d\ge24$: $a_\sigma$ wächst mit $N$ ($\propto d\,\Delta\ln N$) |
| F9 | Übergangscharakter | NESS bei $\sigma>0$: kontinuierlich (keine Hysterese); Spinodal existiert nur bei $\sigma\to0$ |

---

## Problem 1 — Beweis von $\zeta_1^\ast = 1/4$  *(schärfstes Ziel, schnellster Gewinn)*

**Aussage.** Der korrigierte erste Hoeffding-Koeffizient der Geodäsie-U-Statistik,
gemessen an der NESS-Kante $\eta_{\mathrm{lo}}$, konvergiert gegen exakt
$\tfrac14$:
$$\zeta_1^\ast(N)\;=\;\frac14\;+\;O(N^{-1/2}),\qquad
\text{gemessen: } 0{,}217\,(32),\ 0{,}239\,(64),\ 0{,}246\,(128),\ 0{,}2558\,(1024).$$

**Warum es scharf ist.** $\zeta_1 = \mathrm{Var}(g_1)/\sigma_h^2$ mit
$g_1(q)=\E_{Q'}[d_G(q,Q')] - \E[d_G]$: eine einzige Zahl, definiert durch die
Zwei-Punkte-Verteilung *an der Kante*. Kein ganzzahiger Zufall — $1/4$ riecht nach
einer exakten Identität (z. B. $\mathrm{Var}$ einer Projektion auf einen
zweidimensionalen Modenraum).

**Warum es Ressourcen braucht.** Man braucht die *nah-kritische* Zwei-Punkte-Verteilung
$(Q_i,Q'\,)$ — nicht die reine Unordnete (die gilt nur unterhalb der Kante) und kein
Quasiteilchen-Bild (das ist unbelegt). Konkret: (i) den kritischen Eigenmodus der
linearisierten Fokker–Planck-Gleichung um die ungeordnete NESS analytisch bestimmen
(dort könnte $3d+k$ wieder auftauchen!); (ii) $d_G$ in diesem Modus bis zur zweiten
Ordnung entwickeln; (iii) die Varianz von $g_1$ durch die Modenamplitude ausdrücken
und deren Wert am Kriterienpunkt einsetzen. Schritt (i) ist die eigentliche Arbeit;
$\Gr(k,d)$ hat keine Fourier-Moden, sondern Zerlegungen nach $O(d)$-Darstellungen.

**Einstiegspunkte.** Peter–Weyl-Zerlegung von $L^2(\Gr(k,d))$; die ungeordnete NESS
$\rho^\ast \approx e^{-\eta V_{\mathrm{eff}}}/Z$ störend entwickeln; $\zeta_1$ als
Funktional des ersten nichttrivialen Modus identifizieren.

**Erfolgskriterium.** $|{\zeta_1^\ast - 1/4}| < 0{,}01$ hergeleitet, inklusive des
$O(N^{-1/2})$-Abfalls (gemessen: $0{,}217\to0{,}256$ zwischen $N=32$ und $1024$).

**Ressourcen.** ~2–5 konzentrierte Arbeitstage. Keine GPU.

---

## Problem 2 — Herleitung der Anker-Grenzfunktion $B(x)$  *(nicht-monoton!)*

**Aussage.** Die Desynchronisationsgrenze als Funktion von
$x=\lambda_W p_{\mathrm{fix}}$ ist
$$B(x):\ \text{kein Durchgang für } x\lesssim 0{,}015;\quad
B_{\max}\approx2{,}10 \text{ bei } x^\ast\approx0{,}075;\quad
B'(x)\approx-1{,}1 \text{ für } x\gtrsim0{,}1.$$
Zu zeigen: Schwellwert, Maximum-Lage und Abfallssteigung aus Störungstheorie.

**Warum es Ressourcen braucht.** Zwei konkurrierende Effekte wirken gegeneinander:
die **rückstellende Kraft** des Ankers (messbar: $\lambda_{\mathrm{det}}\approx-2{,}3\,x$,
linear in $x$ — stabilisierend, erzeugt den Anstieg) und eine **parametrische Heizung**
(Agenten werden periodisch zum Anker gezogen und wieder freigegeben; vermutlich
$\propto\lambda_W^2p(1-p)$ — destabilisierend, erzeugt den Abfall). Beide wirken in
einem nicht-Gradienten-, selbstkonsistenten Fluss ohne bekannte Lyapunov-Funktion.
Der Abfallzweig ist das überraschende Element: *starke* Anker bringen die geordnete
Phase gegen die Kopplung wieder zum Einsturz — das gilt es als Term zu identifizieren.

**Einstiegspunkte.** (i) Mehrzeitskalen-Ansatz: Anker als schnell-schwacher Antrieb;
zweiten Kramers–Moyal-Koeffizient der Ankerstöße als effektive Diffusion berechnen
und zur Rauschseite schlagen. (ii) Die Plateau-Bedingung
$\lambda_{\mathrm{det}} = -\kappa x + \nu x^2 + \ldots = 0$ nach $\eta$ auflösen;
$\kappa\approx2{,}3$ ist schon gemessen. (iii) Gegenprobe: $p$-Abhängigkeit des
Abfalls bei festem $x$ (Daten vorhanden: schwach).

**Erfolgskriterium.** $x_c$, $x^\ast$, $B_{\max}$, $B'$ innerhalb der Messfehler
($\pm0{,}005$ an der Grenze, ±5 % am Peak) reproduziert.

**Ressourcen.** ~3–7 Arbeitstage. Keine GPU.

---

## Problem 3 — Mechanismus der $d$-Dichotomie (Plateau vs. Kontraktion)

**Aussage.** Am exakt synchronisierten Zustand ($\sigma=0$, $\lambda_W=0$) ist der
top-Transversal-Exponent
$$\lambda_{\mathrm{det}}(\eta)\;\equiv\;0 \ \ (\text{Messgenauigkeit})\quad
\forall\,\eta\in[0{,}002,\,1{,}8] \text{ für } d\ge4,$$
aber $\lambda_{\mathrm{det}}(\eta)\approx-\gamma\,\eta$, $\gamma\approx1$, für
$d=2$ — und der Verlust an $\eta_{\mathrm{hi}}$ ist diskontinuierlich
($-0{,}18\to+0{,}405$ zwischen $\eta=1{,}83$ und $2{,}5$). Zu zeigen: warum die
differenzierende Größe $d$ ist und nicht $k$ (gilt für $(1,4)$ wie $(2,4)$!).

**Warum es Ressourcen braucht.** Die scheinbar einfache Frage hat eine Falle: Im
exakt synchronisierten Zustand ist $\bar S=P$ (Rang-$k$-Projektor) und der Drift
$(I-P)\bar S Q$ verschwindet — **für jede** Auslenkung, in erster Ordnung *und*
individuell. Die Rückstellkraft kommt erst aus der kollektiven Antwort
($\bar S$ ändert sich nur um $O(\delta^2\!\cdot\!B̄ + \delta/\sqrt{N})$). Warum daraus
bei $d=2$ ein *linearer* Term ($\gamma\approx1$, sauber gemessen) und bei $d\ge4$
ein *Plateau* (kein Term bis zur Messgrenze) wird, ist offen. Verdacht: bei $d=2$
ist der Tangentialraum eindimensional — es gibt keine isotope Zusatzrichtungen,
und der return-Mechanismus koppelt direkt an die Relativ-Winkel (Kreis-Geometrie);
bei $d\ge4$ bleibt der oberste Modus im Nulleigenwert hängen (Goldstone-artige
Flachheit). Das zu formalisieren erfordert die zweite Ordnung der
Selbstkonsistenz-Entwicklung auf $\Gr(k,d)$ — keine Standardliteratur.

**Einstiegspunkte (zwei billige Diskriminierungsexperimente, je < 1 GPU-h).**
1. **$N$-Skalierung der Dekadenzrate $\gamma$:** bei $N=16,64,256$ messen. Skaliert
   $\gamma\sim1/\sqrt N$ oder $1/N$, kommt der Term aus den $\bar S$-Fluktuationen
   (oben skizziert); ist $\gamma$ $N$-unabhängig, kommt er aus echter Geometrie.
2. **$\delta$-Skalierung:** Dekadenzrate bei $\delta=10^{-3}$ vs.\ $10^{-5}$. Rate
   $\propto\delta$ ⟹ quadratischer (nichtlinearer) restore-Term; Rate
   $\delta$-unabhängig ⟹ linearer Term.

**Erfolgskriterium.** Eine Rechnung, die $\lambda_{\mathrm{det}}\equiv0$ ($d\ge4$)
und $-\gamma\eta$ mit $\gamma=1{,}00{\pm}0{,}05$ ($d=2$) liefert, und den
diskontinuierlichen Sprung bei $\eta_{\mathrm{hi}}$ erklärt.

**Ressourcen.** ~1–2 Wochen; die zwei Experimente oben zuerst (sie kosten eine
Stunde und schneiden den Raum der möglichen Mechanismen drastisch).

---

## Problem 4 — Ursprung von $3d+k$  *(der Kern; Forschungsproblem)*

**Aussage.** Erkläre
$$a_\sigma(k,d)\;=\;3d+k$$
aus der nichtlinearen Geometrie des Systems. Restriktionen, die jede Theorie
erfüllen muss:
- **F2:** $a_\sigma=3d+k$ (±5 %) für $d\le16$, $N\ge64$;
- **F3:** die lineare Rate ist universal $\gamma\approx1$ — die Erklärung kann
  *nicht* über Relaxationsraten laufen;
- **Unabhängigkeit von $D=k(d-k)$:** $(1,4)$ und $(2,4)$ geben trotz $D$-Unterschied
  um Faktor 3 denselben Wert 13,2. Jede Argumentation, die tangentiale
  Freiheitsgrade $k(d-k)$ zählt, scheidet damit aus — die Antwort muss etwas
  zählen, das additiv in $d$ und $k$, aber nicht multiplikativ in ihnen ist;
- **F4:** $\eta_{\mathrm{hi}}\approx2{,}04$ universal für $k\ge2$ — dieselbe Theorie
  sollte idealerweise auch das liefern;
- **F8:** Gültigkeitsbrechen: bei $d\ge24$ wächst $a_\sigma$ mit $N$
  ($\approx d\,\Delta\ln N$ bei $d=32$) — im Limes müssen $N$ und $d$ vertauschbar
  sein oder die Rechnung muss die Reihenfolge spezifizieren.

**Warum es so viele Ressourcen braucht — der eigentliche Grund.**
Alle vier anderen Probleme sind *Rechnungen* mit bekannten Werkzeugen (Störungstheorie,
Darstellungen, Variationalprinzipien). Problem 4 ist ein *Erklärungsproblem*: Wir
wissen nicht einmal, welche Struktur die Antwort trägt. Der natürliche Kandidat
(Wolkenausdehnung der ungeordneten NESS ≤ Einfangsradius des geordneten Beckens)
erfordert zwei Objekte, die beide nicht in geschlossener Form existieren: das
rauschgetragene Gleichgewicht eines nicht-Gradienten-Mittelfeld-Systems auf einer
Mannigfaltigkeit, und die Basin-Grenze eines chaotisch gerahmten Attraktors. Dazu
kommt die Mathematik-typische Volatilität: Der additive Form $3d+k$ und die
$D$-Unabhängigkeit zusammen sind ein *hartes* Muster — sie schließen alle einfachen
Tensorzählungen aus und deuten auf eine Identität, die wir nicht sehen. Solche
Identitäten (vergleiche $d(d+2)/2$ als Summe der Quadrate der $O(d)$-Generatoren)
kommen oft aus Darstellungstheorie — aber unsere Messung sagt explizit, dass die
naive $O(d)$-Konstante $d(d+2)/2$ es *nicht* ist. Man sucht also auf einem Feld
ohne Landkarte.

**Einstiegspunkte.**
1. **Die Faktentabelle als Raster:** gesucht ist eine Größe $G(k,d)$ mit
   $G=3d+k$, die aus einem Kriterium Typ
   „Wolkenausdehnung $\sigma^2\cdot(\text{etwas}) \approx \eta\cdot(\text{etwas anderes})$"
   folgt. Kandidaten für die „etwas": Spur des Diffusions-Tensors
   ($\propto k(d-k)$ — scheidet nach F2 aus!), Spur eines Rückstell-Tensors,
   Holzmassen-Dimensionen von $\Gr(k,d)$-Bündeln.
2. **$d=8$-Familie präzise zerlegen:** für $k=1,2,4$ bei $d=8$ messen wir
   $23{,}9/25{,}8/27{,}3$ — die $k$-Abhängigkeit ist sublinear ($\approx +1{,}1$
   pro $k$). Genauer vermessen (mehr $k$ bei festem $d$) und den $k$-Koeffizienten
   isolieren.
3. **Kleines $N$-Grenzspiel:** bei $d=32$, $N=32$: Verhältnis 1,07 — nahe am Gesetz.
   Die N-Kopplung (F8) verschwindet offenbar bei $N\lesssim d$. Wenn die Theorie
   den Limes $N\sim d$ kontrolliert, vereinfacht sie sich vielleicht dort zuerst.

**Erfolgskriterium.** Eine Herleitung von $a_\sigma=3d+k$ ±5 % über mindestens
sechs Konfigurationen, plus Vorhersage für einen *ungemessenen* Punkt
(z. B. $(2,6)$: Vorhersage 20, oder $(1,6)$: 19), danach verifiziert.

**Ressourcen.** Ehrlich: **Wochen bis Monate, möglich dass es offen bleibt.**
Das ist der Posten, der den Löwenanteil der „sehr viel Zeit" ausmacht.

---

## Problem 5 — Konventionsabbildung auf die Quell-Theorie  *(blockiert, dann trivial)*

**Aussage.** Unsere Messung zeigt, dass *weder* $\eta_{\mathrm{lo}}$ *noch*
$\eta_{\mathrm{hi}}$ auf $D\,d(d{+}2)/2$ unter einer $(k,d)$-unabhängigen Konstanten
abbildet ($\eta_{\mathrm{hi}}$ ist für $k\ge2$ geometrieunabhängig; $a_\sigma=3d+k$).
Die $\eta$-Normierung der Quell-Publikation muss also einen $(k,d)$-abhängigen Faktor
enthalten. Gesucht: die Abbildung $\eta_{\mathrm{paper}} = g(k,d)\cdot\eta_{\mathrm{step}}$.

**Warum es blockiert ist.** Es fehlt ausschließlich die präzise Definition von $\eta$
in der Quell-Theorie (wie ist der Drift normiert, pro Agent oder pro Modus, diskret
oder kontinuierlich?). Mit dieser Definition ist es reine Einsetzarbeit.

**Erfolgskriterium.** $g(k,d)$ so bestimmt, dass *alle* unsere 12 $a_\sigma$-Werte
und beide Kanten auf die Vorhersage der Quelle fallen — oder nachgewiesen, dass die
Quell-Konstante $D\,d(d{+}2)/2$ ein *anderes* Objekt beschreibt (z. B. eine
Differentiabar-Grenze, nicht die NESS-Kante) — auch das wäre ein klares Ergebnis.

**Ressourcen.** Stunden (nach Eingang der Definition). Keine GPU.

---

## Prioritäten und Abhängigkeiten

```
Problem 5  (blockiert bis zur η-Definition — Stunden)     ── unabhängig
Problem 1  (ζ₁* = 1/4 — 2–5 Tage)                          ── unabhängig, höchster Ertrag/Aufwand
Problem 2  (Anker-Kurve B(x) — 3–7 Tage)                   ── unabhängig
Problem 3  (d-Dichotomie — 1–2 Wochen)                     ── zuerst: 2 Diskriminierungs-Experimente (je 1 GPU-h)
Problem 4  (3d+k — Wochen bis Monate)                      ── hängt an 1+3: deren Werkzeuge fließen ein
```

**Empfohlene Reihenfolge:** 1 → 2 → 3-Experimente → 3 → 4. Problem 4 zuletzt:
es erbt die Techniken von 1 und 3, und die zwei Diskriminierungsexperimente aus
Problem 3 kosten fast nichts, verschließen aber den Mechanismenraum.

## Zusammenfassung der Ressourcenfrage

- **GPU:** praktisch erschöpft an Ertrag (~1,5 h von 10 h verbraucht; Rest nur für
  die zwei billigen Experimente in Problem 3).
- **Mathematik-Zeit gesamt:** Probleme 1+2+3+5: **~3–5 Wochen** fokussierter Arbeit;
  Problem 4: **offen, Wochen bis Monate, Erfolgsrisiko hoch**.
- Damit lautet die ehrliche Antwort auf „wie viele Stunden": **die vier Rand-Probleme
  sind ~100–200 Arbeitsstunden; der Kern (Problem 4) ist ein Forschungsprojekt ohne
  verlässliches Stundenkonto.**

---

## Reproduktion

Alle im Text zitierten Zahlen: `REPORT.md` (§1–13), Rohdaten `results/*.json`,
Simulator `diva.py`, Skripte `phase*.py`, `V*.py`; mathematische Fassung:
`diva_mathematik.tex`.
