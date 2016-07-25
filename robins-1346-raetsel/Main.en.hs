module Main where

import Data.List
import Control.Monad

data Exp a
    = Lit  a
    | Sum  (Exp a) (Exp a)
    | Diff (Exp a) (Exp a)
    | Prod (Exp a) (Exp a)
    | Quot (Exp a) (Exp a)
    deriving (Eq)

instance (Show a) => Show (Exp a) where
    show (Lit x)   = show x
    show (Sum  a b) = "(" ++ show a ++ ")+(" ++ show b ++ ")"
    show (Diff a b) = "(" ++ show a ++ ")-(" ++ show b ++ ")"
    show (Prod a b) = "(" ++ show a ++ ")*(" ++ show b ++ ")"
    show (Quot a b) = "(" ++ show a ++ ")/(" ++ show b ++ ")"

eval :: (Fractional a) => Exp a -> a
eval (Lit x) = x
eval (Sum  a b) = eval a + eval b
eval (Diff a b) = eval a - eval b
eval (Prod a b) = eval a * eval b
eval (Quot a b) = eval a / eval b

main :: IO ()
main = mapM_ print $ filter ((== 17) . eval) $ concatMap arb $ permutations [6,6,5,2]

-- Given a list `xs`, `arb xs` is the list of those syntax trees which
-- have the elements of `xs` at their leaves (in the same order as in `xs`).
-- Every element of `xs` occurs exactly once in those trees.
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

-- Given a list `xs`, `groups xs` is the list of all possible ways
-- of splitting `xs` into two parts: a front part and a back part.
groups :: [a] -> [([a],[a])]
groups []     = [([],[])]
groups (x:xs) = map (\(as,bs) -> (x:as,bs)) (groups xs) ++ [([], x:xs)]
