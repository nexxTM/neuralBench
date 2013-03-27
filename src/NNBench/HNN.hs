module NNBench.HNN
    ( hnn
    , hnnAD
    ) where

import Control.Arrow ((***))

import NNBench.Common

import AI.HNN.FF.Network as HNN
import Numeric.LinearAlgebra

import Numeric.AD

dropHeadTail :: [a] -> [a]
dropHeadTail xs
    | length xs < 3 = []
    | otherwise = take (length xs -2) $ tail xs

{-
instance NNet (HNN.Network Double) (HNN.Samples Double) where
    withNet layers action                = createNetwork (head layers) (dropHeadTail layers) (last layers) >>= action
    withData trnD action                 = action $ convert trnD
    trainNTimes times lRate act net trnD = return $ HNN.trainNTimes times lRate (getActFunc act) (getActFuncDeri act) net trnD
    calcError act net tstD               = return $ mse act net tstD
-}

hnn :: NNet (Network Double) (Samples Double)
hnn = NNet
    { withNet = \layers action -> createNetwork (head layers) (dropHeadTail layers) (last layers) >>= action
    , withData = \trnD action -> action $ convert trnD
    , NNBench.Common.trainNTimes = \times lRate act net samples ->
          return $ HNN.trainNTimes times lRate (getActFunc act) (getActFuncDeri act) net samples
    , calcError = \act net tstD -> return $ mse act net tstD
    }

-- | Test the influence of automatic differentiation on HNN
hnnAD :: NNet (Network Double) (Samples Double)
hnnAD = hnn
    { NNBench.Common.trainNTimes = \times lRate act net samples ->
          return $ HNN.trainNTimes times lRate (getActFunc act) (getActFuncDiff act) net samples
    }

convert :: TrainingData Double -> Samples Double
convert = map (fromList *** fromList)

mse :: ActFunc -> Network Double -> Samples Double -> Double
mse act net samples = quadError (getActFunc act) net samples / fromIntegral (length samples)

getActFunc :: ActFunc -> ActivationFunction Double
getActFunc Sigmoid = sigmoid
getActFunc Tanh = tanh

getActFuncDeri :: ActFunc -> ActivationFunctionDerivative Double
getActFuncDeri Sigmoid = sigmoid'
getActFuncDeri Tanh = tanh'

getActFuncDiff :: ActFunc -> ActivationFunctionDerivative Double
getActFuncDiff Sigmoid = diff sigmoid
getActFuncDiff Tanh = diff tanh