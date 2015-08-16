-- Dieses kurze Haskell-Programm klärt folgende Frage:
--
-- Wie muss man die Ziffern 1, 3, 4 und 6 mit Klammern und den
-- Operatoren +, *, -, / ergänzen, damit das Ergebnis 24 ist?
--
-- Um das Programm auszuführen, muss man die Haskell Platform installieren:
-- https://www.haskell.org/platform/
module Main where

import Data.List
import Control.Monad

-- Typ für Termbäume.
data Exp a
    = Lit  a
    | Sum  (Exp a) (Exp a)
    | Diff (Exp a) (Exp a)
    | Prod (Exp a) (Exp a)
    | Quot (Exp a) (Exp a)
    deriving (Eq)

-- Übersichtliche Darstellung von Termbäumen.
instance (Show a) => Show (Exp a) where
    show (Lit x)   = show x
    show (Sum  a b) = "(" ++ show a ++ ")+(" ++ show b ++ ")"
    show (Diff a b) = "(" ++ show a ++ ")-(" ++ show b ++ ")"
    show (Prod a b) = "(" ++ show a ++ ")*(" ++ show b ++ ")"
    show (Quot a b) = "(" ++ show a ++ ")/(" ++ show b ++ ")"

-- Wie man einen Termbaum zu einer Zahl auswertet.
eval :: (Fractional a) => Exp a -> a
eval (Lit x) = x
eval (Sum  a b) = eval a + eval b
eval (Diff a b) = eval a - eval b
eval (Prod a b) = eval a * eval b
eval (Quot a b) = eval a / eval b

-- Die Lösung des Rätsels.
main :: IO ()
main = mapM_ print $ filter ((== 24) . eval) $ concatMap arb $ permutations [1,3,4,6]

-- Gegeben eine Liste `xs` von (zum Beispiel) Zahlen, gibt eine Liste von
-- allen Termbäumen zurück, an deren Blättern (in genau der gegebenen
-- Reihenfolge) die Zahlen aus xs stehen. Alle Zahlen müssen verwendet werden,
-- und zwar jeweils genau einmal.
arb :: [a] -> [Exp a]
arb []  = []
arb [x] = [Lit x]
arb xs  = do
    (as,bs) <- groups xs
    guard $ not . null $ as
    guard $ not . null $ bs
    left    <- arb as
    right   <- arb bs
    op      <- [Sum, Diff, Prod, Quot]
    return $ op left right

-- Gegeben eine Liste, berechnet alle Möglichkeiten, diese Liste in zwei
-- Teile zu zerlegen: einen vorderen und einen hinteren.
groups :: [a] -> [([a],[a])]
groups []     = [([],[])]
groups (x:xs) = map (\(as,bs) -> (x:as,bs)) (groups xs) ++ [([], x:xs)]
