import numpy as np
import random

# (kursnummer, zeitnummer)
# (kurs, praeszeit)
# fuer jeden Teilnehmenden vier Slots ziehen
# Bedingungen:
# 1. Kursnummer(Slot) != Kursnummer(Teilnehmenden)
# 2. Zeitnummer(Slot) != Praesentationszeit(Teilnehmenden)
# 3. Zeitnummern der vier Slots alle verschieden

slots = [(kurs, zeit) for kurs in [1,2,3,4] for zeit in [800,900,1000,1100,1200]]

def drawSlots(kurse, zeiten):
  if zeiten:
    for k in kurse:
      # (zeiten[0], k)
      for ss in drawSlots([k__ for k__ in kurse if k__ != k], zeiten[1:]):
	yield [(k,zeiten[0])] + ss
  else:
    yield []

def checkMinimalAttendance(kurse, zeiten, m, sss):
  for k in kurse:
    for z in zeiten:
      besucher = 0
      for ss in sss:
	if (k,z) in ss:
	  besucher += 1
      if besucher < m:
	return False
  return True

# K1: 16
# K2: 16
# K3: 16
# K4: 15
# K5: 16
# K6: 15

zeiten          = [800, 900, 1000, 1100, 1200]
kurse           = [1, 2, 3, 4, 5, 6]
anzahlen 	= [16, 16, 16, 15, 16, 15]
eigeneVortraege = []
antiWuensche    = []
for k, a in zip(kurse, anzahlen):
  moeglicheKurseZumBesuchen = [ k_ for k_ in kurse if k_ != k ]
  for z in zeiten:
    eigeneVortraege += [(k,z), (k,z), (k,z)]
    antiWuensche    += [random.choice(moeglicheKurseZumBesuchen), random.choice(moeglicheKurseZumBesuchen), random.choice(moeglicheKurseZumBesuchen)]
  if a % 3 != 0:
    eigeneVortraege += [(k, zeiten[-1])]
    antiWuensche    += [random.choice(moeglicheKurseZumBesuchen)]

def str2(i): return "%02d" % i

def createTimetable(zeiten, kurse, eigeneVortraege, antiWuensche, zuteilungsliste):
  numAtt = len(eigeneVortraege)
  timetable = [[' ' for z in zeiten] for k in kurse]
  indAtt = 0
  anzahlRespektierterAntiWuensche = len(eigeneVortraege)
  for eigenerVortrag, antiWunsch, hoereVortraege in zip(eigeneVortraege, antiWuensche, zuteilungsliste):
    kursEigenerVortrag = eigenerVortrag[0]
    zeitEigenerVortrag = eigenerVortrag[1]
    timetable[kurse.index(kursEigenerVortrag)][zeiten.index(zeitEigenerVortrag)] += str2(indAtt) + 'V, '
    for kursZuhoerer, zeitZuhoerer in hoereVortraege:
      timetable[kurse.index(kursZuhoerer)][zeiten.index(zeitZuhoerer)] += str2(indAtt) + ',  '
      if kursZuhoerer == antiWunsch:
        anzahlRespektierterAntiWuensche -= 1
    indAtt += 1
  personenCounts = [ zeit.count(",") for kurs in timetable for zeit in kurs ]
  return (anzahlRespektierterAntiWuensche, personenCounts, timetable)

def printTimetable(zeiten, kurse, (anzahlRespektierterAntiWuensche, personenCounts, timetable)):
  n = 0
  for kurs in timetable:
    print("Kurs {0}".format(kurse[n]))
    print("-----------------")
    m = 0
    for zeit in kurs:
      print(str(zeiten[m]) + "\t" + zeit)
      m += 1
    print("")
    n += 1

  print("Respektierte Antiwuensche: {0}".format(anzahlRespektierterAntiWuensche))
  print("Minimalzahl anwesender Personen in einem Slot: {0}".format(min(personenCounts)))
  print("Maximalzahl anwesender Personen in einem Slot: {0}".format(max(personenCounts)))
  print("Median      anwesender Personen in einem Slot: {0}".format(np.median(personenCounts)))
  print("Personenanzahlen: " + str(sorted(personenCounts)))

def drawZuteilungen(kurse, zeiten, eigeneVortraege, antiWuensche):
  kurse = list(kurse)
  if eigeneVortraege:
    random.shuffle(kurse)
    for ss in drawSlots([k for k in kurse if k != eigeneVortraege[0][0] and k != antiWuensche[0]], [z for z in zeiten if z != eigeneVortraege[0][1]]):
      random.shuffle(kurse)
      for sss in drawZuteilungen(kurse, zeiten, eigeneVortraege[1:], antiWuensche[1:]):
	yield [ss] + sss
  else:
    yield []
# [ [(2,900),(3,1000)], [(2,800), (3,1000)], [...] ]

def drawZuteilungenMitMinimalbedingung(kurse, zeiten, m, eigeneVortraege, antiWuensche, breakAfter=None):
  j = 0
  for sss in drawZuteilungen(kurse, zeiten, eigeneVortraege, antiWuensche):
    if checkMinimalAttendance(kurse, zeiten, m, sss):
      yield sss
    j = j + 1
    if breakAfter and j == breakAfter:
      break

def dump(xs):
  for x in xs:
    print(x)

# Wir bevorzugen solche Loesungen, bei denen ...
# ... die Minimalzahl Personen in irgendeinem Slot moeglichst gross,
# ... die Maximalzahl moeglichst klein,
# ... der Median moeglichst gross,
# ... die Anzahl der Slots mit Minimalbesetzung moeglichst klein,
# ... die Anzahl der Slots mit Maximalbesetzung moeglichst klein und
# ... die Anzahl der Slots mit insgesamt 15 Personen moeglichst gross
# ist. Dann sind die Kinder naemlich gleichmaessiger verteilt.
# Ja, der letzte Punkt ist etwas ad hoc.
def cost(cs):
  return [-min(cs), max(cs), -np.median(cs), len([1 for c in cs if c == min(cs)]), len([1 for c in cs if c == max(cs)]), -len([1 for c in cs if c == 15])]

l = None

# viele zufaellige Starts versuchen
for i in range(5000):
  for sss in drawZuteilungenMitMinimalbedingung(kurse, zeiten, 8, eigeneVortraege, antiWuensche, 150):
    (anzahlRespektierterAntiWuensche, personenCounts, tt) = createTimetable(zeiten, kurse, eigeneVortraege, antiWuensche, sss)
    if not l or cost(l[1]) > cost(personenCounts):
      l = (anzahlRespektierterAntiWuensche, personenCounts, tt)
      printTimetable(zeiten, kurse, l)

printTimetable(zeiten, kurse, l)

# Fixe Idee: Kann man ein Gradientenabstiegsverfahren verwenden? Zunaechst
# natuerlich nicht. Hier ist ja nichts kontinuierlich. Aber was, wenn man im
# Modell erlaubt, dass sich Personen kontinuierlich aufteilen koennen?
# Zum Beispiel: Zu 10% gehe ich zu Zeitpunkt 1 in Kurs 1, zu 70% in Kurs 2, zu
# 20% in Kurs 3.
