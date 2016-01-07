{-
  Haskell-Programm zur experimentellen Untersuchung der Lösungsstrategie für
  das Zwergenrätsel von Gil Kalai (http://mathoverflow.net/a/30192/31233):

  100 Zwerge haben Hüte aufgesetzt bekommen, die zwar alle die gleiche Farbe
  haben, aber mit einer (ganzen) Zahl zwischen 0 und 99 (einschließlich)
  beschriftet sind. Die Zahlen können nach irgendeinem System geordnet sein
  oder auch völlig zufällig verteilt sein. Es kann auch sein, dass manche
  Zahlen mehrmals und andere gar nicht vorkommen. Jeder Zwerg sieht die Zahlen
  der anderen Zwerge, aber nicht seine eigene. Auf Zuruf müssen dann alle
  Zwerge gleichzeitig eine Zahl zwischen 0 und 99 nennen. Stimmt sie mit der
  eigenen Hutzahl überein, überlebt der betreffende Zwerg. Sonst nicht. Auf
  welche Strategie müssen sich die Zwerge im Vorhinein einigen, damit sicher
  (unabhängig vom Zufall) mindestens ein Zwerg überlebt?
-}
module Main where

import Control.Monad
import System.Random
import Control.Parallel.Strategies

-- Diese Funktion kodiert die Strategie, die jeder einzelne Zwerg fährt.
-- Übergeben wird ein Paar aus den sichtbaren Hutzahlen der anderen Zwerge
-- und der eigenen Indexnummer (nicht der eigenen Hutzahl, die ist ja
-- dem Zwerg unbekannt). Zurückgegeben wird die Zahl, die der Zwerg als
-- Vermutung für seine eigene Hutzahl ausspricht.
--
-- Die hier implementierte Strategie lautet: Addiere die sichtbaren Hutzahlen
-- all der anderen Zwerge. Ziehe von der eigenen Indexnummer diese Summe ab.
-- Verkünde das Ergebnis dieser Rechnung (modulo der Gesamtzahl Zwerge) als
-- Vermutung für die eigene Hutzahl.
strategy :: ([Int], Int) -> Int
strategy (xs, i) = (i - sum xs) `mod` (length xs + 1)

main :: IO ()
main = do
    -- Anzahl Zwerge festlegen
    n  <- prompt "Anzahl Zwerge:"

    -- Zufällige Hutzahlen zwischen 0 und n-1 (jeweils eingeschlossen) ziehen
    xs <- replicateM n $ randomRIO (0, n-1)

    -- Alle Zwerge ihre Vermutungen aussprechen lassen
    let xs' = pmap strategy $ zip (skips xs) [0..]

    -- Anzahl überlebender Zwerge bestimmen
        m   = length $ filter (uncurry (==)) $ zip xs xs'

    putStrLn $ "Mit der Strategie haben " ++ show m ++ " Zwerge überlebt."

-- Gegeben eine Liste, bestimmt alle Möglichkeiten, jeweils ein Element aus
-- der Liste zu entfernen. Zum Beispiel ist `skips "abcd"` gleich
-- `["bcd", "acd", "abd", "abc"]`.
skips :: [a] -> [[a]]
skips []     = []
skips (x:xs) = xs : map (x:) (skips xs)

-- Liest eine Eingabe von der Konsole ein.
prompt :: (Read a) => String -> IO a
prompt msg = putStrLn msg >> fmap read getLine

-- Wie `map`, nur auf mehrere Threads verteilt. Jeder Thread arbeitet
-- immer 20 Listenelemente am Stück ab.
pmap :: (NFData b) => (a -> b) -> [a] -> [b]
pmap f xs = map f xs `using` parListChunk 20 rdeepseq
