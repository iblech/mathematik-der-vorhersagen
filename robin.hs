module Main where

import Data.List

chars  = "+*-/()"
digits = "1346"

xs us a b c d e = a ++ [us!!0] ++ b ++ [us!!1] ++ c ++ [us!!2] ++ d ++ [us!!3] ++ e

n = 3

possies = do
    ds <- permutations digits
    i <- [0..n]
    j <- [0..n]
    k <- [0..n]
    l <- [0..n]
    m <- [0..n]
    a <- go i chars
    b <- go j chars
    c <- go k chars
    d <- go l chars
    e <- go m chars
    let s = xs ds a b c d e
    return s

go :: Int -> [a] -> [[a]]
go 0 cs = []
go 1 cs = map (:[]) cs
go n cs = do
    c   <- cs
    cs' <- go (n-1) cs
    return (c:cs')
