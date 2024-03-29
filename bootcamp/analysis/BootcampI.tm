<TeXmacs|1.99.2>

<style|<tuple|generic|american|german>>

<\body>
  <doc-data|<doc-title|Mathematik-Bootcamp>|<\doc-subtitle>
    Teil 1: Differentialrechnung
  </doc-subtitle>||||<doc-author|<\author-data|<author-name|Philipp D�ren &
  Ingo Blechschmidt>>
    \;
  <|author-data|<\author-email>
    philipp.dueren@gmail.com
  </author-email>>
    \;

    \;

    \;
  </author-data>>>

  <abstract-data|<\abstract>
    In diesem Skript wirst du (fast) alles lernen, was du f�r unseren Kurs an
    Wissen �ber Differentialrechnung brauchst. Die Differentialrechnung ist
    ein gro�artiges Gebiet der Mathematik, das sich mit der Summe von
    unendlich vielen unendlich kleinen Zahlen besch�ftigt. Klingt etwas
    esoterisch, ist aber unglaublich n�tzlich!

    Klassische Fragestellungen sind:

    <\itemize>
      <item>Was ist \RGeschwindigkeit``? Was ist \RBeschleunigung``? Wie
      h�ngen die beiden zusammen?\ 

      <item>Wie finde ich maximale und minimale Werte von Funktionen?

      <item>Wie finde ich Nullstellen von Funktionen?
    </itemize>
  </abstract>>

  <section|Grundlagen>

  <subsection|Funktionen und Visualisierungen>

  In der Differentialrechnung befassen wir uns mit Funktionen, die auf einer
  Teilmenge der reellen Zahlen definiert sind und Werte in den reellen Zahlen
  haben, z.B.

  <\eqnarray*>
    <tformat|<table|<row|<cell|f:\<bbb-R\><rsub|+>\<supset\>M>|<cell|\<rightarrow\>>|<cell|\<bbb-R\>>>|<row|<cell|x>|<cell|\<mapsto\>>|<cell|<sqrt|x>>>>>
  </eqnarray*>

  Diese Funktion ist auf einer Teilmenge <math|M> der positiven reellen
  Zahlen <math|\<bbb-R\><rsub|+>> definiert (das ist wiederum eine Teilmenge
  der reellen Zahlen) und bildet jeden Punkt aus dieser Definitionsmenge auf
  seine Quadratwurzel ab.

  In der angewandten Mathematik erweist es sich oft n�tzlich, Funktionen
  zuzulassen, die Definitions- und Wertemengen haben, die eine h�here
  Dimension haben als 1. Zum Beispiel kann man eine Karte eines Berges so
  beschreiben, dass man jedem Punkt (der durch geographische Breiten- und
  L�ngengrade beschrieben wird, also zwei Koordinatenkomponenten hat) die
  H�he zuordnet, die der Berg an diesem Punkt hat.

  <big-figure|<image|mnt.png|638px|476px||>|Der Graph einer Funktion
  <math|f:\<bbb-R\><rsup|2>\<supset\>M\<rightarrow\>\<bbb-R\>> (also von zwei
  Dimensionen in eine Dimension). Die beiden \REingabekoordinaten`` sind auf
  der waagerechten Ebene eingetragen. Der Wert der Funktion in diesem Punkt
  entspricht der H�he der Oberfl�che senkrecht �ber diesem Punkt.>

  Ein weiteres Beispiel ist die Geschwindigkeit von Wasserwellen auf einer
  Wasseroberfl�che: Wir k�nnen in jedem Punkt auf der Wasseroberfl�che einen
  Pfeil einzeichnen, der in die Richtung zeigt, in den sich ein Molek�l
  Wasser in der n�chsten Sekunde bewegen wird.

  <big-figure|<image|fluid.png|799px|599px||>|Eine Funktion
  <math|f:\<bbb-R\><rsup|2>\<supset\>M\<rightarrow\>\<bbb-R\><rsup|2>> (also
  von zwei Dimensionen in zwei Dimensionen). In jedem Punkt
  <math|w\<in\>M\<subset\>\<bbb-R\><rsup|2>> erhalten wir einen
  zweikomponentigen Funktionswert <math|f<around*|(|w|)>\<in\>\<bbb-R\><rsup|2>>,
  den wir graphisch an <math|w> als Vektor anh�ngen. Dies machen wir
  nat�rlich nicht in jedem Punkt (sonst w�re alles voller Linien). Die L�nge
  des Vektors entspricht der St�rke der Str�mung in diesem Punkt.>

  Ein letztes Beispiel: Wenn wir ein Pendel ansto�en und filmen, erhalten wir
  zu jedem Zeitpunkt <math|t\<in\><around*|[|t<rsub|1>,t<rsub|2>|]>> (das
  Zeitintervall, in dem gefilmt wurde) eine Position des Pendelgewichts in
  der Schwingungsebene (das ist eine Teilmenge des <math|\<bbb-R\><rsup|2>>).
  Legen wir diese Einzelbilder wie Toastscheiben aneinander, erhalten wir
  eine Kurve durch den dreidimensionalen Raum.

  <big-figure|<image|pendulum.png|799px|599px||>|<label|im:pend>Die Bahnkurve
  eines Pendels. Zu jedem Zeitpunkt im Intervall
  <math|<around*|[|t<rsub|1>,t<rsub|2>|]>> (die rote Linie ist die Zeitachse)
  k�nnen wir eine dazu senkrecht stehende Schnittebene einsetzen. Auf dieser
  Ebene ist genau ein Punkt der blauen Kurve. Verschieben wir nun die
  Schnittebene entlang der Zeitachse, beschreibt der Punkt auf der
  Schnittebene, der auf der Kurve liegt, genau die Bahn eines Pendels. Dies
  ist ein Beispiel f�r eine Funktion <math|f:\<bbb-R\>\<supset\><around*|[|t<rsub|1>,t<rsub|2>|]>\<rightarrow\>\<bbb-R\><rsup|2>>,
  also von einer Dimension in zwei Dimensionen.>

  F�r Funktionen <math|f:\<bbb-R\>\<supset\>M\<rightarrow\>\<bbb-R\><rsup|n>>,
  also speziell Funktionen, die von einer Dimension in beliebige Dimensionen
  gehen, w�hlt man oft auch eine alternative Wahl der Visualisierung: Man
  interpretiert die Eingabekoordinate (also die Variablen in der
  Definitionsmenge) als Zeitkoordinate und tr�gt den \Raktuellen Punkt zur
  Zeit <math|t>`` in ein Koordinatensystem <math|\<bbb-R\><rsup|n>> ein: Das
  dritte Beispiel s�he dann so aus:

  <big-figure|<image|pend2d.png|799px|599px||>|Ein parametriserter Plot des
  Pendelbeispiels. Die Graphik entsteht, wenn man alle \RZeitscheiben``
  aufeinanderklept und die blaue Kurve in Abbildung <reference|im:pend>
  zusammenstaucht. Au�erdem sind die Funktionswerte f�r drei Zeitpunkte
  eingetragen: Den Punkten pink, rot und gr�n entsprechen die Zeitpunkte
  <math|1.9>, <math|2.4> und <math|2.9>. Wir sehen, dass die Punkte trotz
  gleicher Zeitdifferenzen unterschiedliche Entfernungen voneinander haben.
  Das liegt an der unterschiedlichen Geschwindigkeit der Bahnkurve (wir
  werden diese Idee noch genauer behandeln).>

  <\exercise>
    Welche Dimensionen hat eine Abbildung, die die Temperaturverteilung in
    einem Raum angibt, also den Wert der Temperatur in Grad Celsius in jedem
    Punkt im Zimmer? Mit anderen Worten: Bestimme <math|m> und <math|n> in
    <math|T:\<bbb-R\><rsup|m>\<supset\>M\<rightarrow\>\<bbb-R\><rsup|n>>.
  </exercise>

  <\exercise>
    Bestimme <math|m> und <math|n> in <math|v:\<bbb-R\><rsup|m>\<supset\>M\<rightarrow\>\<bbb-R\><rsup|n>>,
    wenn <math|v> den Geschwindigkeitsvektor der Luft in der Atmosph�re
    beschreibt? Nimm an, dass wir nur einen kleinen Abschnitt der Atmosph�re
    betrachten (also ein w�rfelf�rmiges \RSt�ck Luft``, das �ber der
    Oberfl�che schwebt). Versuche, dir eine Visualisierung dieser Funktion zu
    �berlegen (mittelschwierig), zu zeichnen (mittelschwierig) oder mit einem
    Computerprogramm zu zeichnen (schwierig)
  </exercise>

  <section|Position und Geschwindigkeit>

  \ Die Hauptfragestellung der Differentialrechnung ist die Folgende:\ 

  <\question*>
    Was ist die aktuelle �nderungsrate einer Funktion in einem Punkt?
  </question*>

  Zuerst m�ssen wir verstehen, was diese Frage �berhaupt bedeuten soll.
  Betrachten wir dazu die H�henfunktion einer Silvesterrakete nach dem
  Anz�nden der Z�ndschnur:

  <big-figure|<image|positionRocket.png|799px|599px||>|Die H�he einer (nicht
  explodierenden) Silvesterrakete nach dem Anz�nden als eine Funktion von der
  Zeit in Sekunden: <math|h:<around*|[|0,20|]>\<rightarrow\>\<bbb-R\>>. Als
  Punkt markiert ist der Zeitpunkt, an dem der Treibstoff der Rakete
  verbraucht ist.>

  <big-figure|<image|velocityRocket.png|799px|599px||>|Die vertikale
  Geschwindigkeit der Silvesterrakete. Die Geschwindigkeitszunahme
  (Beschleunigung) ist nicht konstant, da die Rakete mit dem Verbrauch des
  Treibstoffs leichter wird und bei gleicher Antriebskraft immer schneller
  beschleunigt wird (Newtons Gesetz <math|a=<frac|F|m>>). Markiert ist der
  Punkt, an dem der Treibstoff verbraucht ist. Hier nimmt die Geschwindigkeit
  aufgrund der Fallbeschleunigung wieder ab (bleibt aber weiterhin positiv,
  die Rakete steigt also immer noch). Zum Zeitpunkt, wo die rote Linie die
  <math|0>-Linie durchbricht (bei etwa 17s), ist die Geschwindigkeit der
  Rakete 0. In diesem Moment h�ngt die Rakete regungslos in der Luft, um
  gleich darauf Richtung Boden zu fallen (negative Geschwindigkeit).>

  Wir k�nnen folgende relevanten Zeitpunkte und Phasen ausmachen:

  <big-figure|<with|gr-mode|<tuple|edit|text-at>|gr-frame|<tuple|scale|1cm|<tuple|0.5gw|0.5gh>>|gr-geometry|<tuple|geometry|1par|3cm|center>|gr-line-width|5ln|gr-point-style|round|gr-grid|<tuple|cartesian|<point|-7|-1>|1>|gr-grid-old|<tuple|cartesian|<point|-7|-1>|1>|gr-edit-grid-aspect|<tuple|<tuple|axes|none>|<tuple|1|none>|<tuple|1|none>>|gr-edit-grid|<tuple|cartesian|<point|-7|-1>|1>|gr-edit-grid-old|<tuple|cartesian|<point|-7|-1>|1>|gr-grid-aspect-props|<tuple|<tuple|axes|#808080>|<tuple|1|#c0c0c0>|<tuple|10|#e0e0ff>>|gr-grid-aspect|<tuple|<tuple|axes|#808080>|<tuple|1|#c0c0c0>>|<graphics||<with|arrow-end|\<gtr\>|line-width|2ln|<line|<point|-7|-1>|<point|6.76680715439068|-1.0>>>|<text-at|Start
  (<math|0 s>)|<point|-7.33924791639106|-1.34268752480487>>|<text-at|Treibstoff
  ersch�pft (<math|\<approx\>8 s>)|<point|-2.0|-1.38163868019324>>|<with|line-width|5ln|<line|<point|-0.641933|-0.646167>|<point|-0.633372762879084|-1.0>>>|<with|line-width|5ln|<line|<point|4.28729|-0.610448>|<point|4.28917087802009|-1.0>>>|<text-at|Ende
  des Steigfluges (<math|\<approx\>17 s>)|<point|2.64421550469639|-1.37840653525599>>|<text-at|Aufw�rtsbewegung|<point|-6.0|0.216364854198757>>|<text-at|bei
  gleichzeitiger Beschleunigung|<point|-7.0|-0.217871581283796>>|<text-at|Aufw�rtsbewegung|<point|0.510128627817342|0.160851303082418>>|<text-at|bei
  gleichzeitiger Abbremsung|<point|-0.445478899325308|-0.235398200820214>>|<text-at|Abw�rtsbewegung|<point|3.41217422939542|1.21122172245006>>|<text-at|bei
  gleichzeitiger Abw�rtsbeschleunigung|<point|1.19759558142611|0.746874586585527>>|<text-at|(Erdanziehung)|<point|5|0.260589>>|<with|line-width|5ln|<line|<point|-0.463338|1.46125>|<point|-0.481685460907156|1.0>|<point|1.08002050535785|0.5607223177669>|<point|4.88244146051065|0.348756449265776>|<point|5.18813702703437|0.0>|<point|7.25896468275713|0.0>>>>>|Der
  Lebenslauf unserer Rakete (viel sp�ter, bei etwa <math|30 s>, wird die
  Rakete auf dem Boden aufprallen). Wir nennen die drei Zeitintervalle Phase
  1, 2 und 3. >

  Sehen wir uns die Graphen der Funktionen in diesen Phasen genauer an:

  <big-table|<tabular|<tformat|<cwith|1|-1|1|-1|cell-hyphen|n>|<table|<row|<cell|>|<cell|H�hendiagramm>|<cell|Geschwindigkeitsdiagramm>>|<row|<cell|Start>|<cell|<math|0>>|<cell|<math|0>>>|<row|<cell|Phase
  1>|<cell|steigend, linksgekr�mmt>|<cell|positiv,
  steigend>>|<row|<cell|Treibstoff verbraucht>|<cell|Kr�mmungswechsel>|<cell|Wechsel
  zwischen steigend und fallend>>|<row|<cell|Phase 2>|<cell|steigend,
  rechtsgekr�mmt>|<cell|positiv, fallend>>|<row|<cell|Ende des
  Steigfluges>|<cell|waagerechte Richtung>|<cell|<math|0>>>|<row|<cell|Phase
  3>|<cell|fallend, rechtsgekr�mmt>|<cell|negativ, fallend>>>>>|Eigenschaften
  der Graphen>

  Wir erkennen folgende Zusammenh�nge:

  <\note*>
    \;

    <\itemize>
      <item>Immer, wenn das H�hendiagramm ansteigt, ist das
      Geschwindigkeitsdiagramm positiv.

      <item>Immer, wenn das H�hendiagramm abf�llt, ist das
      Geschwindigkeitsdiagramm negativ.

      <item>Immer, wenn das H�hendiagramm linksgekr�mmt ist, ist das
      Geschwindigkeitsdiagramm steigend.

      <item>Immer, wenn das H�hendiagramm rechtsgekr�mmt ist, ist das
      Geschwindigkeitsdiagramm fallend.

      <item>Punkte, an denen das H�hendiagramm flach ist (<math|0 s> und
      <math|17 s>), sind Nullstellen des Geschwindigkeitsdiagramms.
    </itemize>
  </note*>

  Wir wollen nun verstehen, wie man diese Zusammenh�nge mathematisch fassen
  kann.

  <section|Die Ableitung einer Funktion>

  Die Ableitung einer Funktion <math|f:\<bbb-R\>\<supset\><around*|[|a,b|]>\<rightarrow\>\<bbb-R\>>
  ist so definiert, wie die Geschwindigkeit definiert ist. Man misst die
  (Durchschnitts-)Geschwindigkeit eines Objektes, indem man zwei Zeitpunkte
  <math|t<rsub|1>> und <math|t<rsub|2>> w�hlt und die Strecke <math|s> misst,
  die das Objekt in dieser Zeit <math|t=t<rsub|2>-t<rsub|1>> zur�cklegt. Die
  Durchschnittsgeschwindigkeit w�hrend dieser Bewegung (also im Zeitraum
  <math|<around*|[|t<rsub|1>,t<rsub|2>|]>>) ist dann der Quotient aus Distanz
  und Dauer, also

  <\equation*>
    v<rsub|t<rsub|1>\<rightarrow\>t<rsub|2>>=<frac|s|t<rsub|2>-t<rsub|1>>
  </equation*>

  <big-figure|<with|gr-mode|<tuple|edit|text-at>|gr-frame|<tuple|scale|1cm|<tuple|0.5gw|0.5gh>>|gr-geometry|<tuple|geometry|1par|5cm|center>|gr-line-width|2ln|gr-dash-style|11100|gr-arrow-end|\<gtr\>|<graphics||<line|<point|-6.17838|-1.92519>|<point|5.41243881465802|-1.88946950654849>>|<with|line-width|2ln|<carc|<point|-4.49959|-0.496428>|<point|-3.21370220928694|-0.317833046699299>|<point|-2.927950125678|-0.746461172112713>>>|<with|dash-style|11100|line-width|2ln|<carc|<point|0.66181|-0.389271>|<point|1.60836420161397|-0.228535520571504>|<point|2.12628985315518|-1.05007276094722>>>|<with|arrow-end|\<gtr\>|dash-style|11100|line-width|2ln|<line|<point|1.30475|-1.06793>|<point|-3.82092538695595|-1.0143537504961>>>|<math-at|t<rsub|1>=2s|<point|0.822546|0.575142>>|<math-at|t=4s|<point|-4.12454|0.682299>>|<text-at|Distanz<math|=3
  m>|<point|-1.92782|-0.532147>>>>|Durchschnittsgeschwindigkeitsmessung: Wir
  haben zwei Zeitpunkte <math|t<rsub|1>> und <math|t<rsub|2>> gew�hlt und
  eine Distanz gemessen. Damit ist die Durchschnittsgeschwindigkeit zwischen
  <math|t<rsub|1>> und <math|t<rsub|2>> gleich
  <math|v<rsub|t<rsub|1>\<rightarrow\>t<rsub|2>>=<frac|3 m|<around*|(|4 s-2
  s|)>>=1.5 <frac|m|s>>>

  \;
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-10|<tuple|7|?>>
    <associate|auto-11|<tuple|1|?>>
    <associate|auto-12|<tuple|3|?>>
    <associate|auto-13|<tuple|8|?>>
    <associate|auto-2|<tuple|1.1|?>>
    <associate|auto-3|<tuple|1|?>>
    <associate|auto-4|<tuple|2|?>>
    <associate|auto-5|<tuple|3|?>>
    <associate|auto-6|<tuple|4|?>>
    <associate|auto-7|<tuple|2|?>>
    <associate|auto-8|<tuple|5|?>>
    <associate|auto-9|<tuple|6|?>>
    <associate|im:pend|<tuple|3|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|figure>
      <tuple|normal|Der Graph einer Funktion
      <with|mode|<quote|math>|f:\<bbb-R\><rsup|2>\<supset\>M\<rightarrow\>\<bbb-R\>>
      (also von zwei Dimensionen in eine Dimension). Die beiden
      \REingabekoordinaten`` sind auf der waagerechten Ebene eingetragen. Der
      Wert der Funktion in diesem Punkt entspricht der H�he der Oberfl�che
      senkrecht �ber diesem Punkt.|<pageref|auto-3>>

      <tuple|normal|Eine Funktion <with|mode|<quote|math>|f:\<bbb-R\><rsup|2>\<supset\>M\<rightarrow\>\<bbb-R\><rsup|2>>
      (also von zwei Dimensionen in zwei Dimensionen). In jedem Punkt
      <with|mode|<quote|math>|w\<in\>M\<subset\>\<bbb-R\><rsup|2>> erhalten
      wir einen zweikomponentigen Funktionswert
      <with|mode|<quote|math>|f<around*|(|w|)>\<in\>\<bbb-R\><rsup|2>>, den
      wir graphisch an <with|mode|<quote|math>|w> als Vektor anh�ngen. Dies
      machen wir nat�rlich nicht in jedem Punkt (sonst w�re alles voller
      Linien). Die L�nge des Vektors entspricht der St�rke der Str�mung in
      diesem Punkt.|<pageref|auto-4>>

      <tuple|normal|Die Bahnkurve eines Pendels. Zu jedem Zeitpunkt im
      Intervall <with|mode|<quote|math>|<around*|[|t<rsub|1>,t<rsub|2>|]>>
      (die rote Linie ist die Zeitachse) k�nnen wir eine dazu senkrecht
      stehende Schnittebene einsetzen. Auf dieser Ebene ist genau ein Punkt
      der blauen Kurve. Verschieben wir nun die Schnittebene entlang der
      Zeitachse, beschreibt der Punkt auf der Schnittebene, der auf der Kurve
      liegt, genau die Bahn eines Pendels. Dies ist ein Beispiel f�r eine
      Funktion <with|mode|<quote|math>|f:\<bbb-R\>\<supset\><around*|[|t<rsub|1>,t<rsub|2>|]>\<rightarrow\>\<bbb-R\><rsup|2>>,
      also von einer Dimension in zwei Dimensionen.|<pageref|auto-5>>

      <tuple|normal|Ein parametriserter Plot des Pendelbeispiels. Die Graphik
      entsteht, wenn man alle \RZeitscheiben`` aufeinanderklept und die blaue
      Kurve in Abbildung <reference|im:pend> zusammenstaucht. Au�erdem sind
      die Funktionswerte f�r drei Zeitpunkte eingetragen: Den Punkten pink,
      rot und gr�n entsprechen die Zeitpunkte <with|mode|<quote|math>|1.9>,
      <with|mode|<quote|math>|2.4> und <with|mode|<quote|math>|2.9>. Wir
      sehen, dass die Punkte trotz gleicher Zeitdifferenzen unterschiedliche
      Entfernungen voneinander haben. Das liegt an der unterschiedlichen
      Geschwindigkeit der Bahnkurve (wir werden diese Idee noch genauer
      behandeln).|<pageref|auto-6>>

      <tuple|normal|Die H�he einer (nicht explodierenden) Silvesterrakete
      nach dem Anz�nden als eine Funktion von der Zeit in Sekunden:
      <with|mode|<quote|math>|h:<around*|[|0,20|]>\<rightarrow\>\<bbb-R\>>.
      Als Punkt markiert ist der Zeitpunkt, an dem der Treibstoff der Rakete
      verbraucht ist.|<pageref|auto-8>>

      <tuple|normal|Die vertikale Geschwindigkeit der Silvesterrakete. Die
      Geschwindigkeitszunahme (Beschleunigung) ist nicht konstant, da die
      Rakete mit dem Verbrauch des Treibstoffs leichter wird und bei gleicher
      Antriebskraft immer schneller beschleunigt wird (Newtons Gesetz
      <with|mode|<quote|math>|a=<frac|F|m>>). Markiert ist der Punkt, an dem
      der Treibstoff verbraucht ist. Hier nimmt die Geschwindigkeit aufgrund
      der Fallbeschleunigung wieder ab (bleibt aber weiterhin positiv, die
      Rakete steigt also immer noch). Zum Zeitpunkt, wo die rote Linie die
      <with|mode|<quote|math>|0>-Linie durchbricht (bei etwa 17s), ist die
      Geschwindigkeit der Rakete 0. In diesem Moment h�ngt die Rakete
      regungslos in der Luft, um gleich darauf Richtung Boden zu
      fallen.|<pageref|auto-9>>
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Grundlagen>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1<space|2spc>Funktionen und
      Visualisierungen <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Geschwindigkeiten,
      Differenzen und Zuw�chse> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>