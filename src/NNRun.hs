{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE BangPatterns #-}

module NNRun where

import Text.CSV
import Criterion.Main
import Criterion.Config

import NNBench.Common
import NNBench.HNN
import NNBench.HFANN
import NNBench.Neural

data BenchConf = BC
    { description' :: String  -- ^ Used in the criterion group name
    , netPara'     :: NNetPara -- ^ Parameters for the NNs
    , trainTimes'  :: [Int] -- ^ The number of batch runs for trainNTimes.
    -- ^ [4,9] means benchmark with train4Times and benchmark witch train9Times
    , trainD'      :: TrainingData Double -- ^ The data to train the neural network with
    , testD'       :: TrainingData Double -- ^ The data to test the neural network with
    }

benchTargets :: [BenchTarget]
benchTargets = [(GNNet hnn, "HNN"), (GNNet hfann, "HFANN")] --, (GNNet neural, "neural-haskell")]

benchConf :: String -> [Int] -> Double -> ActFunc -> [Int] -> TrainingData Double -> TrainingData Double -> BenchConf
benchConf description hiddenLayers lRate act trainTimes trainD testD =
    BC description (nrOfInputs trainD : hiddenLayers ++ [nrOfOutputs trainD], lRate, act) trainTimes trainD testD

benchConf' :: String -> [Int] -> Double -> ActFunc -> [Int] -> TrainingData Double -> BenchConf
benchConf' description hiddenLayers lRate act trainTimes tData =
    let divisor = floor $ fromIntegral (nrOfRows tData) * 0.8
        (trainD, testD) = splitAt divisor tData
    in benchConf description hiddenLayers lRate act trainTimes trainD testD

main :: IO ()
main = do
    spamTrnD <- loadData "data/spambase/spambase.data" 1
    spamSTrnD <- loadData "data/spambase/spambase-short.data" 1
    wineTrnD <- loadData "data/winequality/winequality-red-cor.csv" 10
    -- 10 because the rating is between 0 and 10, but sigmoid is between 0 and 1 
    wineSTrnD <- loadData "data/winequality/winequality-red-cor-short.csv" 10
    --print $ nrOfInputs spamTrnD -- 57
    --print $ nrOfInputs wineTrnD -- 11
    --print $ nrOfRows spamTrnD  -- 4601
    --print $ nrOfRows spamSTrnD -- 150
    --print $ nrOfRows wineTrnD  -- 1599
    --print $ nrOfRows wineSTrnD -- 150
    let xorConf   = benchConf  "XOR" [2] 0.7 Tanh [1,100,1000,10000,100000] xorTrnD xorTrnD
        spamConf  = benchConf' "Spam" [70] 0.5 Sigmoid [1,100,400] spamTrnD
        spamSConf = benchConf' "Spam_short" [70] 0.5 Sigmoid [1,100,1000] spamSTrnD
        wineConf  = benchConf' "Wine" [15,15] 0.4 Sigmoid [1,100,1000,3000] wineTrnD
        wineSConf = benchConf' "Wine_short" [15,15] 0.4 Sigmoid [1,100,1000,10000] wineSTrnD
        confs     = [xorConf, spamConf, spamSConf, wineConf, wineSConf]
        xorConfT   = benchConf  "XOR" [2] 0.7 Tanh [1] xorTrnD xorTrnD
        spamConfT  = benchConf' "Spam" [70] 0.5 Sigmoid [1] spamTrnD
        spamSConfT = benchConf' "Spam_short" [70] 0.5 Sigmoid [1] spamSTrnD
        wineConfT  = benchConf' "Wine" [15,15] 0.4 Sigmoid [1] wineTrnD
        wineSConfT = benchConf' "Wine_short" [15,15] 0.4 Sigmoid [1] wineSTrnD
        confsT     = [xorConfT, spamConfT, spamSConfT, wineConfT, wineSConfT]
    {- let xorConf   = benchConf  "XOR" [2] 0.7 Tanh [1,100,1000] xorTrnD xorTrnD
        spamConf  = benchConf' "Spam" [70] 0.5 Sigmoid [1] spamTrnD
        spamSConf = benchConf' "Spam_short" [70] 0.5 Sigmoid [1,5] spamSTrnD
        wineConf  = benchConf' "Wine" [15,15] 0.4 Sigmoid [1] wineTrnD
        wineSConf = benchConf' "Wine_short" [15,15] 0.4 Sigmoid [1,100] wineSTrnD
        confs     = [xorConf, spamConf, spamSConf, wineConf, wineSConf] -}
    mapM_ (myBench "NNReport" True benchTargets) confs
    mapM_ (myBench "FFOnlyNNReport" False benchTargets) confsT
    mapM_ (myBench "ADNNReport" True [(GNNet hnn, "HNN"), (GNNet hnnAD, "HNNAD")]) confs

myBench :: String -> Bool -> [BenchTarget] -> BenchConf -> IO ()
myBench prefix doTraining targets (BC description netPara trainTimes trainD testD) =
    mapM_ doBenchGroup trainTimes
    where
      doBenchGroup bRuns = do
          let myConfig = defaultConfig { cfgReport  = ljust $ "reports/" ++ prefix ++ description ++ show bRuns ++ ".html" }
          doBench bRuns targets (\bs -> defaultMainWith myConfig (return ()) [bgroup (description ++ "/" ++ show bRuns) bs])
      doBench :: Int -> [BenchTarget] -> ([Benchmark] -> IO ()) -> IO () 
      doBench _     []                      cont = cont []
      doBench bRuns ((GNNet netI, name):ts) cont = do
          let withDataIns = withData netI
          withDataIns trainD (\ !wTrnD ->  -- the withData function converts the data to the instance format and (depending on
            withDataIns testD (\ !wTstD -> -- the instance) cleans the converted data up. Therefore it is only save to access
                                           -- the data within the function. That is the reason for the continuation.
              doBench bRuns ts (\bs -> cont (bench name (nfIO (trainTest netI doTraining wTrnD wTstD netPara bRuns)) : bs))))
              
-- It is (or seems) impossible to mix GNNet and NNet in a type signatuer to provide functions like trainTest as a parameter to
-- myBench, which lead to the boolean paramater 

trainTest :: NNet net tdata -- ^ The neural network implementation to train with
      -> Bool        -- ^ Indicates whether the network should be trained
      -> tdata       -- ^ The data to train the neural network with
      -> tdata       -- ^ The data to test the neural network with
      -> NNetPara    -- ^ The parameters for the neural network
      -> Int         -- ^ The number of batch runs
      -> IO Double   -- ^ The square error on the test data
trainTest netI doTraining trnD tstD (layers, lRate, act) bRuns =
    withNet netI layers (\net ->
        if doTraining then do
            tnet <- trainNTimes netI bRuns lRate act net trnD
            calcError netI act tnet tstD
         else
            calcError netI act net tstD)

xorTrnD :: TrainingData Double
xorTrnD = [ ([0,0],[0]) -- 0 xor 0 = 0
          , ([0,1],[1]) -- 0 xor 1 = 1
          , ([1,0],[1])
          , ([1,1],[0])
          ]

loadData :: FilePath -> Double -> IO (TrainingData Double)
loadData path fac = do
    csvD <- parseCSVFromFile path
    case csvD of
         Left errmsg -> print errmsg >> undefined
         Right rs    -> return $ parseRecords rs fac

parseRecords :: [Record] -> Double -> TrainingData Double
parseRecords rs fac =
   let dss = [[read f :: Double | f <- r] | r <- rs] 
       iptL = length (head dss) - 1
       in [(take iptL ds, [last ds / fac]) | ds <- dss]