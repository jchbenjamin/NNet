


module NNet where

type W = [[Double]]
data Layer = In X | Hid Nodes | Out Nodes

type X = [(Sample,D)]

type Sample = [Double]
type D = [Integer]

type Nodes = [Node]
data Node = Node {
--	i :: Int --index
	b :: Double,
	v :: Double,
	y :: Double,
	e :: Double,
	del :: Double
} deriving (Show)

phi :: Double -> Double
phi x = 1 / (1 + ( (exp 1) ** (-x)))

dphi :: Double -> Double
dphi x = (phi x) * (1 - (phi x))

error :: D -> Layer -> Layer
error _ (In _) = undefined
error _ (Hid _) = undefined
error a (Out xs) = undefined

calc_e :: Integer -> Double -> Double
calc_e x y = (fromIntegral x) - y

feedforward :: Int -> Int -> Int -> Node -> Layer -> W -> Double
feedforward s i j n (In x) w = (b n) + sum( zipWith (*) (w !! i) (fst (x!!s)) ) 
feedforward s i j n (Hid x) w = (b n) + sum( zipWith (*) (w !! i) (map (y) x) ) 

calc_del_end :: Node -> Double
calc_del_end n = (e n) * dphi(v n)

calc_del_hid :: Node -> Layer -> W -> Double
calc_del_hid n (Out x) = dphi(v n) * sum( zipWith (*) (w !! (i n)) (del x))

dw :: Double -> Int -> Int -> Node -> Layer -> Double
dw a s n (In x) = a * (del n) * (fst (x !! s))
dw a s n (Hid x) = a * (del n) * (

db :: Double -> Node -> Double
db a n = a * (del n)




{-
let w = [[1.0,2.0,3.0],[1.1,2.1,3.1]]

let x = [([1.5,2.5],[1]),([3.1,3.2],[0])]

let myX = In x

let myNode = Node 1.0 1.0 1.0 1.0 1.0

let myNodee = Node 2.0 2.0 2.0 2.0 2.0 

let myNodeee = Node 3.0 3.0 3.0 3.0 3.0

let myLastN = Node 0.0 0.0 0.0 0.0 0.0

let myLastLayer = [myLastN]

let myMidLayer = Hid [myNode,myNodee,myNodeee]

-}


