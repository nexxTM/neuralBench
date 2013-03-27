{-# LANGUAGE ExistentialQuantification #-}

module NNBench.Common where

type Inputs a = [a]
type Outputs a = [a]
type TrainingRow a = (Inputs a, Outputs a)
type TrainingData a = [TrainingRow a]

type BenchTarget = (GNNet, String)

data GNNet = forall a b. GNNet (NNet a b)

type NNetPara = ( [Int]   -- ^ The layer layout. 'head' is the number of inputs and 'last' the number of outputs. The rest are the hidden layers.
                , Double  -- ^ The learning rate
                , ActFunc -- ^ The activation function to use in the nn
                )

data ActFunc = Sigmoid | Tanh

{-
class NNetC net tdata | net -> tdata where
    withNetC     :: [Int] -- ^ Network architecture
                   -> (net -> IO Double) -- ^ A function which gets the created NN
                   -> IO Double
    withDataC    :: TrainingData Double -- ^ Data to train a NN with
                   -> (tdata -> IO Double)
                   -- ^ A function which gets the data in the correct format
                   -> IO Double
    trainNTimesC :: Int -- ^ The number of batch runs
                   -> Double -- ^ The learning rate
                   -> ActFunc -- ^ The activation function
                   -> net -- ^ The NN to train
                   -> tdata -- ^ The data to train the NN with
                   -> IO net
    calcErrorC   :: ActFunc -> net -> tdata -> IO Double
-}

data NNet net tdata = NNet
    { withNet :: [Int] -- ^ Network architecture
                  -> (net -> IO Double) -- ^ A function which gets the created NN
                  -> IO Double
    , withData :: TrainingData Double -- ^ Data to train a NN with
                  -> (tdata -> IO ())
                  -- ^ A function which gets the data in the correct format
                  -> IO ()
                  -- TODO adjust criterion to return the data produced by a tested
                  -- function and then change this function to return Double to
                  -- collect the errors from the benchmark
    , trainNTimes :: Int -- ^ The number of batch runs
                  -> Double -- ^ The learning rate
                  -> ActFunc -- ^ The activation function
                  -> net -- ^ The NN to train
                  -> tdata -- ^ The data to train the NN with
                  -> IO net
    , calcError :: ActFunc -> net -> tdata -> IO Double
    }

getInputs :: TrainingRow a -> Inputs a
getInputs = fst

getOutputs :: TrainingRow a -> Outputs a
getOutputs = snd

nrOfRows :: TrainingData a -> Int
nrOfRows = length

nrOfInputs :: TrainingData a -> Int
nrOfInputs = length . getInputs . head

nrOfOutputs :: TrainingData a -> Int
nrOfOutputs = length . getOutputs . head