module NNBench.HFANN
    ( hfann
    ) where

import Control.Monad (liftM)
import Data.List (intercalate)
import System.IO.Temp (withSystemTempFile)
import System.IO (hPutStr,hClose)

import NNBench.Common

import HFANN

{-
instance NNet FannPtr TrainDataPtr where
    withNet = withStandardFann
    withData trnD action = withSystemTempFile "hfannTest" $ \fp h -> hPutStr h (genFannString trnD) >> hClose h >> withTrainData fp action
    trainNTimes = trainNTimesL
    calcError _ net tstD = liftM realToFrac $ testData net tstD -- ^ ignores the provided activation function

-}

hfann :: NNet FannPtr TrainDataPtr
hfann = NNet
    { withNet = withStandardFann
    , withData = \trnD action -> withSystemTempFile "hfannTest" $ \fp h -> hPutStr h (genFannString trnD) >> hClose h >> withTrainData fp action
    , trainNTimes = trainNTimesL
    , calcError = \_ net tstD -> liftM realToFrac $ testData net tstD -- ^ ignores the provided activation function
    }
             
genFannString :: TrainingData Double -> String
genFannString t = 
   --num_records num_input num_output
   let header = show (nrOfRows t) ++ " " ++ show (nrOfInputs t) ++ " " ++ show (nrOfOutputs t)
   in header ++ "\n" ++ intercalate "\n" [spaceSep (getInputs r) ++ "\n" ++ spaceSep (getOutputs r) | r <- t]
   where spaceSep xs = concat [show x ++ " " | x <- xs]
   
trainNTimesL :: Int -> Double -> ActFunc -> FannPtr -> TrainDataPtr -> IO FannPtr
trainNTimesL times lRate act net trnD = do
        setActivationFunctionHidden net $ getActFunc act
        setActivationFunctionOutput net $ getActFunc act
        setTrainingAlgorithm net trainIncremental
        setLearningRate net (fromRational $ toRational lRate)
        trainOnData net trnD times 0 0
        return net        

getActFunc :: ActFunc -> ActivationFunction
getActFunc Sigmoid = activationSigmoid
getActFunc Tanh = activationSigmoidSymmetric
-- Symmetric sigmoid activation function, aka. tanh. http://leenissen.dk/fann/html/files/fann_data-h.html#fann_activationfunc_enum