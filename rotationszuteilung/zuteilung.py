#def descent(listOldArrangements, remainingAttendants, 

import random

def bind(xs, f):
  for x in xs:
    for y in f(x):
      yield y
      

# (kursnummer, zeitnummer)
# (kurs, praeszeit)
# fuer jeden Teilnehmenden vier Slots ziehen
# Bedingungen:
# 1. Kursnummer(Slot) != Kursnummer(Teilnehmenden)
# 2. Zeitnummer(Slot) != Praesentationszeit(Teilnehmenden)
# 3. Zeitnummern der vier Slots alle verschieden

slots = [(kurs, zeit) for kurs in [1,2,3,4] for zeit in [800,900,1000,1100,1200]]

def drawSlots(slots, n):
  if n == 0:
    return [ [] ]
  else:
    return bind(slots, lambda (kurs,zeit): bind(drawSlots([(k,z) for (k,z) in slots if z != zeit], n-1), lambda ss: [[(kurs,zeit)] + ss]))

def drawSlots_(slots, n):
  if n == 0:
    yield []
  else:
    for (kurs,zeit) in slots:
      for ss in drawSlots_([(k,z) for (k,z) in slots if z != zeit], n-1):
	yield [(kurs,zeit)] + ss

def drawSlots__(kurse, zeiten):
  if zeiten:
    for k in kurse:
      # (zeiten[0], k)
      for ss in drawSlots__([k__ for k__ in kurse if k__ != k], zeiten[1:]):
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
for k, a in zip(kurse, anzahlen):
  for z in zeiten:
    eigeneVortraege += [(k,z), (k,z), (k,z)]
  if a % 3 != 0:
    eigeneVortraege += [(k, zeiten[-1])]
    
"""zeiten = [800, 900]
kurse = [1, 2]
eigeneVortraege = [(1, 800), (1, 800), (1, 900), (1, 900), (2, 800), (2, 800), (2, 900), (2, 900)]"""

def getIndFromTime(zeit, zeiten):
  for n in range(len(zeiten)):
    if zeiten[n] == zeit:
      return n

def getIndFromKurs(kurs, kurse):
  for n in range(len(kurse)):
    if kurse[n] == kurs:
      return n

def createTimetable(zeiten, kurse, eigeneVortraege, zuteilungsliste):
  numAtt = len(eigeneVortraege)
  timetable = [[' ' for z in zeiten] for k in kurse]
  indAtt = 0
  for eigenerVortrag, hoereVortraege in zip(eigeneVortraege, zuteilungsliste):
    kursEigenerVortrag = eigenerVortrag[0]
    zeitEigenerVortrag = eigenerVortrag[1]
    timetable[getIndFromKurs(kursEigenerVortrag, kurse)][getIndFromTime(zeitEigenerVortrag, zeiten)] += str(indAtt) + 'V, '
    for kursZuhoerer, zeitZuhoerer in hoereVortraege:
      timetable[getIndFromKurs(kursZuhoerer, kurse)][getIndFromTime(zeitZuhoerer, zeiten)] += str(indAtt) + ', '
    indAtt += 1
  return timetable

def printTimetable(zeiten, kurse, timetable):
  n = 0
  for kurs in timetable:
    print("Kurs {0}".format(kurse[n]))
    print("-----------------")
    m = 0
    for zeit in kurs:
      print(str(zeiten[m]) + "\t" + zeit)
      m += 1
    n += 1
      

#eigeneVortraege = [(k,z) for k in kurse for z in zeiten]
#eigeneVortraege = eigeneVortraege * 3 # + [(1, 800) , (2, 800), (3, 800), (1, 1000), (4,1200), (5, 1100), (6, 1200), (4, 1100), (5, 1000), (6,800)]

# 0(1, 800)3	2(2, 800)5	4(3, 800)1
# 1(1, 900)2	3(2, 900)4	5(3, 900)0

# 0,6,12(1, 800)11, 15, 17	2,8,14(2, 800)1, 5, 7		4,10,16(3, 800)3,9,13,
# 1,7,13(1, 900)10, 14, 16	3,9,15(2, 900)0, 4, 6		5,11,17(3, 900)2,8,12,

#print(checkMinimalAttendance(kurse, zeiten, 49, [[(3, 900)], [(3, 800)], [(1, 900)], [(1, 800)], [(2, 900)], [(2,800)]]))

def drawZuteilungen(kurse, zeiten, eigeneVortraege):
  if eigeneVortraege:
    random.shuffle(kurse)
    for ss in drawSlots__([k for k in kurse if k != eigeneVortraege[0][0]], [z for z in zeiten if z != eigeneVortraege[0][1]]):
      random.shuffle(kurse)
      for sss in drawZuteilungen(kurse, zeiten, eigeneVortraege[1:]):
	yield [ss] + sss
  else:
    yield []
# [ [(2,900),(3,1000)], [(2,800), (3,1000)], [...] ]

def drawZuteilungenMitMinimalbedingung(kurse, zeiten, m, eigeneVortraege):
  for sss in drawZuteilungen(kurse, zeiten, eigeneVortraege):
    if checkMinimalAttendance(kurse, zeiten, m, sss):
      yield sss

def dump(xs):
  for x in xs:
    print(x)

l = (drawZuteilungenMitMinimalbedingung(kurse, zeiten, 1, eigeneVortraege))
#dump(drawSlots__([800,900,1000,1200], [1,2,3,4,5]))

firstex = l.next()
tt = createTimetable(zeiten, kurse, eigeneVortraege, firstex)
printTimetable(zeiten, kurse, tt)


def ausgabe(zuteilung, kurse, zeiten):
 
 kurszeit = [[]for i in range(kurse)]

#for i in bind([

#for i in bind([1,2,3], lambda x: bind([4,5,6], lambda y: ([(x,y)] if (x + y) % 2 == 0 else []))):
#   print(i)