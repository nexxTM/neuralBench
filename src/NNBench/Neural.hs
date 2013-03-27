module NNBench.Neural
    ( neural
    ) where

import Neural.FeedForward

import System.Random
import Data.List
import Data.Graph.Inductive (Gr)

import NNBench.Common

type Net a = Gr a a

type Func a = ( Net a -> Net a -- ^ calculates the forward propagation. (The input is already applied.)
              , Net a -> [a] -- ^  calculates the difference between the output calculated by the network and the expected output (the error).
                             -- ^ See Neural.Neural.fErrors
              , a -> Net a -> Net a -- ^ learning function. Takes the learning rate and the net to train. See Neural.Neural.learn
              )

type NetHandle a = ( [a -> a] -- ^ activation function
                     -> [a]   -- ^ input
                     -> [a]   -- ^ targeted output
                     -> Func a, Net a)

initNet :: Int -> Int -> Int -> Int -> IO (NetHandle Double)
initNet nrIn nrUnit nrLayer nrOut = do
    gen <- getStdGen
    let rand = randomRs (-1.0, 1.0) gen :: [Double]
    return $ makeNet rand nrIn nrUnit nrLayer nrOut

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (-x))

sigmoid' :: Floating a => a -> a
sigmoid' x = case sigmoid x of
  y -> y * (1 - y)

sigmoidActivations :: Floating a => [a -> a]
sigmoidActivations = [sigmoid,sigmoid']

teach_h :: a -> [Func a] -> Net a -> Net a
teach_h rate = foldl1' (.) . map teach
            where
            teach (calc, _, learn') = learn' rate . calc

teach_f :: a -> Int -> [Func a] -> Net a -> [Net a]
teach_f rate reps funcs net = take reps $ iterate (teach_h rate funcs) net

err_h :: (Net Double -> [Double]) -> Net Double -> Double
err_h err = sum . map (^2) . err

err_f :: Net Double -> [Func Double] -> Double
err_f net funcs = sum $ map (\(_, err, _) -> err_h err net) funcs

feedData :: ([a] -> [a] -> Func a) -> [([a], [a])] -> [Func a]
feedData = map . uncurry

{-
instance NNet (NetHandle Double) (TrainingData Double) where
    -- | neural haskell is ristricted to networks with an equal number of hidden units per hidden layer.
    -- Therefor the number of hidden units of the first hidden layer is applied to all hidden layers,
    -- which means that (deviating) arguments for deeper hidden layers are ignored.
    withNet layers action = initNet (head layers) (head (tail layers)) (length layers - 2) (last layers) >>= action
    withData trnD action = action trnD
    trainNTimes = trainNTimesL
    calcError = calcErrorL
-}

neural :: NNet (NetHandle Double) (TrainingData Double)
neural = NNet
    {
      withNet = \layers action -> initNet (head layers) (head (tail layers)) (length layers - 2) (last layers) >>= action
      -- ^ neural haskell is ristricted to networks with an equal number of hidden units per hidden layer.
      -- ^ Therefor the number of hidden units of the first hidden layer is applied to all hidden layers,
      -- ^ which means that (deviating) arguments for deeper hidden layers are ignored.
    , withData = \trnD action -> action trnD
    , trainNTimes = trainNTimesL
    , calcError = calcErrorL
    }
    
trainNTimesL :: Int -> Double -> ActFunc -> NetHandle Double -> [([Double], [Double])] -> IO (NetHandle Double)
trainNTimesL times lRate act netH trnD = do
    let (makeCalcErrAndLearn, net) = netH
        mkcalcerrlearn = makeCalcErrAndLearn $ getActFunc act
        funcs = feedData mkcalcerrlearn trnD
        teached_net = last $ teach_f lRate times funcs net
    return (makeCalcErrAndLearn, teached_net) 

calcErrorL :: ActFunc -> NetHandle Double -> [([Double], [Double])] -> IO Double
calcErrorL act netH tstD = do
    let (makeCalcErrAndLearn, net) = netH
        mkcalcerrlearn = makeCalcErrAndLearn $ getActFunc act
        funcs = feedData mkcalcerrlearn tstD
    return $ err_f net funcs / fromIntegral (length tstD)

getActFunc :: ActFunc -> [Double -> Double]
getActFunc Sigmoid = sigmoidActivations
getActFunc Tanh = standartActivations