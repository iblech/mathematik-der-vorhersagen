#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Dieses Programm setzt den TF-IDF-Algorithmus um, mit dem Artikel nach
# thematischer Nähe gruppiert werden können. Es gibt zu jedem gespeicherten
# Artikel den Artikel aus, der am nähesten mit ihm verwandt ist, zum Beispiel:
#
#     ('article-Bruce Willis.txt', 'article-Alicia Silverstone.txt')
#     ('article-Alicia Silverstone.txt', 'article-Catherine Deneuve.txt')
#     ('article-Bosnische Sprache.txt', 'article-Glagolitische Schrift.txt')
#
# Wer sich dafür näher interessiert, wendet sich bitte an Philipp oder Ingo;
# der vorliegende Code ist in Eile entstanden und nicht kommentiert.
#
# Zum Ausführen muss man vorher das Archives articles.tar.bz2 entpacken.
# Es enthält etwa 1500 Artikel der deutschen Wikipedia.

import numpy as np
def clearText(s):
  s = s.lower()
  letters = "abcdefghijklmnopqrstuvwxyz "
  t = ""
  for l in s:
    if l in letters:
      t = t + l
  return t

def test():
  t = "Dies ist ein Text, der viele Kommata enthaelt. Das ist schoen"
  return (clearText(t))
    
def getBagsOfWords(listOfDocuments):  
  dictAllWords = {}
  listOfDicts = []
  for doc in listOfDocuments:
    listOfWords = doc.split(' ')
    listOfDicts.append({})
    for word in listOfWords:
      if word in listOfDicts[-1]:
	listOfDicts[-1][word] += 1
      else:
	listOfDicts[-1][word] = 1
      if word in dictAllWords:
	dictAllWords[word] += 1
      else:
	dictAllWords[word] = 1
  return dictAllWords, listOfDicts

def getTFIDF(listOfDocuments, numRem):
  dictAllWords, listOfDicts = getBagsOfWords(listOfDocuments)
  dictAllWords, listOfDicts = removeStopWords(dictAllWords, listOfDicts, numRem)
  list_tfidf_vecs = [{} for d in listOfDicts]
  ltv2 = [np.zeros(len(dictAllWords)) for doc in listOfDocuments]
  doc_lens = [sum(d.values()) for d in listOfDicts]
  n = 0
  for doc in listOfDocuments:
    #print(n)
    m = 0
    for word in dictAllWords.keys():
      if word in listOfDicts[n]:
	tf = (1.0*listOfDicts[n][word])/doc_lens[n]
	idf = np.log(len(listOfDicts)/(1.0*sum([word in d for d in listOfDicts])))
	list_tfidf_vecs[n][word] = tf*idf
	ltv2[n][m] = tf*idf
      else:
	list_tfidf_vecs[n][word] = 0
	ltv2[n][m] = 0
      m += 1
    n += 1 
      
  return ltv2, len(dictAllWords)
  
def dist(v1, v2):
   return np.arccos(np.dot(v1, v2)/(np.sqrt(np.dot(v1, v1)*np.dot(v2, v2))))
		  
def removeWord(dictAllWords, listOfDicts, word):
  if word in dictAllWords:
    dictAllWords.pop(word)
  for d in listOfDicts:
    if word in d:
      d.pop(word)
  return dictAllWords, listOfDicts

def findMaxKey(dic):
    maxval = 0
    maxkey = -1
    #print(dic.keys())
    for k in dic.keys():
      if k == " ":
	continue
      #print(k)
      #print(dic[k])
      if dic[k] > maxval and k != " ":
	#print(k + "{0} hat Haeufigkeit {1}".format(k, dic[k]))
	maxval = dic[k]
	#print("Neuer Spitzenreiter! maxval={0}".format(maxval))
	maxkey = k
    return maxkey
  
  
def removeStopWords(dictAllWords, listOfDicts, numStopWords):
    for n in range(numStopWords):
      k = findMaxKey(dictAllWords)
      print("removing {0}".format(k))
      dictAllWords, listOfDicts = removeWord(dictAllWords, listOfDicts, k)
    return dictAllWords, listOfDicts
  
t = []
name = []
import glob
listTexts = glob.glob('article-*.txt')
n = 0
for fname in listTexts:
  if (n > 200):
    break
  with open(fname, "r") as myfile:
    t.append(myfile.read().replace('\n', ''))
    t[-1] = clearText(t[-1])
    
  name.append(listTexts[n])
  n = n+1

numSW = 80
v, numWords = getTFIDF(t, numSW)
import numpy as np
dists = np.zeros((len(t), len(t)))
for k in range(len(t)):
  for l in range(len(t)):
    dists[k, l] = dist(v[k], v[l])

mindist = [0 for te in t]
for k in range(len(t)):
  minwert = 2
  lmin = 0
  for l in range(len(t)):
    if l == k:
      continue
    if dists[k,l] < minwert:
      lmin = l
      minwert = dists[k,l]
  mindist[k] = lmin

for k in range(len(t)):
  print(name[k], name[mindist[k]])

#print(dists)
#print(mindist)
#print(name)
"""
num_cluster = 5
cluster = []
for n in range(num_cluster):
  cluster.append(np.random.normal(0, 1, numWords))

#print("Clusterberechnung")
zuordnungen = [0 for vec in v]
for m in range(1000):
  for l in range(len(v)):
    vec = v[l]
    mindist = 10^5
    minc = None
    for n in range(len(cluster)):
      if dist(vec, cluster[n]) < mindist:
	minc = n
	mindist = dist(vec, cluster[n])
    zuordnungen[l] = minc
  avgs = [np.zeros(numWords) for c in cluster]
  sizecluster = [0 for c in cluster]
  for l in range(len(v)):
    vec = v[l]
    zo = zuordnungen[l]
    sizecluster[zo] += 1
    avgs[zo] += vec 
  avgs = [a/n for a, n in zip(avgs, sizecluster)]
  cluster = avgs

def getCluster(zuordnungen, num):
  l = []
  for n in range(len(zuordnungen)):
    if zuordnungen[n] == num:
      l.append(n)
  return l"""

#print(dist(v1, v2))
#print(dist(v1, v3))
#print(dist(v2, v3))
#dictAllWords, listOfDicts = getBagsOfWords([t1,t2])

"""
tBritney = clearText(tBritney)
tCristina = clearText(tCristina)
tBanana = clearText(tBanana)
tApfel = clearText(tApfel)
v1, v2, v3, v4 = getTFIDF([tBritney, tCristina, tBanana, tApfel], 40)
print(dist(v1, v1))
print(dist(v1, v2))
print(dist(v1, v3))
print(dist(v1, v4))
print(dist(v2, v3))
print(dist(v2, v4))
print(dist(v3, v4))"""
