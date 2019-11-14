from backtester.trading_system import TradingSystem
from backtester.version import updateCheck
from backtester.features.feature import Feature
import numpy as np
import pandas as pd
from ukv1_toolbox.problem1.problem1_trading_params import MyTradingParams

from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns

##################################################################################
##################################################################################
## Template file for problem 1.                                                 ##
##################################################################################
## Make your changes to the functions below.
## SPECIFY features you want to use in getInstrumentFeatureConfigDicts()
## Fill predictions in getPrediction()
## The toolbox does the rest for you
## from downloading and loading data to running backtest
##################################################################################


class MyTradingFunctions():

    def __init__(self):  #Put any global variables here
        self.params = {}
        

	###########################################
	## ONLY FILL THE FUNCTION BELOW    ##
	###########################################

    def getInstrumentIds(self):
		# return ['ADBE', 'AMZN', 'ABT','BIIB',
		# 		'CL', 'DHR', 'GD', 'INTC', 'MCD']
        return []

    def getInstrumentFeatureConfigDicts(self):

		# ADD RELEVANT FEATURES HERE
        expma10dic = {'featureKey': 'expma10',
				 'featureId': 'exponential_moving_average',
				 'params': {'period': 4,
							  'featureName': 'Share Price'}}
        mom10dic = {'featureKey': 'mom10',
				 'featureId': 'difference',
				 'params': {'period': 4,
							  'featureName': 'Share Price'}}
        return [expma10dic,mom10dic]


    '''
	A function that returns your predicted value based on your heuristics.
	'''
    

    def getRevenuePrediction(self, time, updateNum, instrumentList, instrumentFeatures, MarketFeatures, instrumentFeatureList, targetVariable, predictions):

		# dataframe for a historical instrument feature (mom10 in this case). The index is the timestamps
		# of upto lookback data points. The columns of this dataframe are the stock symbols/instrumentIds.
		# Get the last row of the dataframe, the most recent datapoint
        mom10 = instrumentFeatures.getFeatureDf('mom10').iloc[-1]
		
        expma10 = instrumentFeatures.getFeatureDf('expma10').iloc[-1] 
        price = instrumentFeatures.getFeatureDf('Share Price').iloc[-1]
		# for f in instrumentFeatureList:
		#  	print(f)
		
		## Linear Regression Implementation

        coeff = [0.056, 0.053, -0.061, 
				0.057, 0.070, 0.099, 
				-0.065, -0.061, -0.061, 
				0.096, 0.105, -0.060, 
				-0.053, 0.074, 0.108, 
				0.108, -0.081, -0.080, 
				0.055, 0.058, -0.076, 
				0.071, 0.084, 0.134, 
				-0.123]
        features = ['Non-Cash Items', 'Other Non-Cash Adjustments', 
'(Increase) Decrease in Accounts Receivable', 'Increase (Decrease) in Accounts Payable', 
'Cash from Operating Activities', 'Decrease in Long Term Investment', 'Increase in Long Term Investment', 
'Net Cash From Acquisitions & Divestitures', 'Net Cash from Other Acquisitions', 
'Cost of revenue', 'Gross Profit', 
'Operating Expenses', 'Selling, General & Administrative', 
'Operating Income (Loss)', 'Pretax Income (Loss), Adjusted', 
'Pretax Income (Loss)', 'Income Tax (Expense) Benefit, net', 
'Inventories', 'Total Noncurrent Assets', 
'Other Payables & Accruals', 'Long Term Debt', 
'Market Cap', 'EBITDA', 
'P/S Ratio', 'EV/Sales' ]
        for s in instrumentList:
            for i in range(len(coeff)):
                predictions[s] += coeff[i] * (instrumentFeatures.getFeatureDf(features[i]).iloc[-1])[s]

		
        predictions.fillna(0,inplace=True)

        return predictions


    '''
    A function that returns your predicted value based on your heuristics.
    '''

    def getIncomePrediction(self, time, updateNum, instrumentList, instrumentFeatures, MarketFeatures, instrumentFeatureList, targetVariable, predictions):

		# dataframe for a historical instrument feature (mom10 in this case). The index is the timestamps
		# of upto lookback data points. The columns of this dataframe are the stock symbols/instrumentIds.
		# Get the last row of the dataframe, the most recent datapoint
		# import pdb; pdb.set_trace()
        mom10 = instrumentFeatures.getFeatureDf('mom10').iloc[-1]
		
        expma10 = instrumentFeatures.getFeatureDf('expma10').iloc[-1] 
        price = instrumentFeatures.getFeatureDf('Share Price').iloc[-1]
		# for f in instrumentFeatureList:
		# 	print(f)
		
		## Linear Regression Implementation

        coeff = [ 0.03249183, 0.49675487]
        for s in instrumentList:
            predictions[s] = coeff[0] * mom10[s] + coeff[1] * expma10[s]

		
        predictions.fillna(0,inplace=True)

        return predictions

	##############################################
	##  CHANGE ONLY IF YOU HAVE CUSTOM FEATURES  ##
	###############################################

    def getCustomFeatures(self):
        return {'my_custom_feature_identifier': MyCustomFeatureClassName}
    
    def varianceThreshold(self):
        """
        Returns columns indexes with variance above the threshold
        """

        with open('historicalData/qq16p1Data/stock_list.txt') as f:
            tickers = f.read().splitlines()
        
        df = pd.DataFrame()
        for ticker in tickers:
            df1 = pd.read_csv("historicalData/qq16p1Data/{}.csv".format(ticker))
            df1.set_index("time", inplace=True)
            df1 = df1.drop(["Revenue(Y)","Income(Y)"], axis=1)
            df1 = df1.replace([np.inf, -np.inf], 0)
            df = pd.concat([df, df1])
       
        print(df.shape)    
        selector = VarianceThreshold(2.5)
        selector.fit_transform(df)
      
        df = df[df.columns[selector.get_support(indices=True)]]
        print(df.shape)
        
        df_corr = df.corr()
        df_corr.to_csv("correlation_table.csv")
        top_corr_features = df_corr.index
        plt.figure(figsize=(50,50))
        #plot heat map
        g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
        
        return

####################################################
##   YOU CAN DEFINE ANY CUSTOM FEATURES HERE      ##
##  If YOU DO, MENTION THEM IN THE FUNCTION ABOVE ##
####################################################
class MyCustomFeatureClassName(Feature):
	''''
	Custom Feature to implement for instrument. This function would return the value of the feature you want to implement.
	1. create a new class MyCustomFeatureClassName for the feature and implement your logic in the function computeForInstrument() -
	2. modify function getCustomFeatures() to return a dictionary with Id for this class
		(follow formats like {'my_custom_feature_identifier': MyCustomFeatureClassName}.
		Make sure 'my_custom_feature_identifier' doesnt conflict with any of the pre defined feature Ids
		def getCustomFeatures(self):
			return {'my_custom_feature_identifier': MyCustomFeatureClassName}
	3. create a dict for this feature in getInstrumentFeatureConfigDicts() above. Dict format is:
			customFeatureDict = {'featureKey': 'my_custom_feature_key',
								'featureId': 'my_custom_feature_identifier',
								'params': {'param1': 'value1'}}
	You can now use this feature by calling it's featureKey, 'my_custom_feature_key' in getPrediction()
	'''
	@classmethod
	def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
		# Custom parameter which can be used as input to computation of this feature
		param1Value = featureParams['param1']

		# A holder for the all the instrument features
		lookbackInstrumentFeatures = instrumentManager.getLookbackInstrumentFeatures()

		# dataframe for a historical instrument feature (basis in this case). The index is the timestamps
		# atmost upto lookback data points. The columns of this dataframe are the symbols/instrumentIds.
		lookbackInstrumentValue = lookbackInstrumentFeatures.getFeatureDf('symbolVWAP')

		# The last row of the previous dataframe gives the last calculated value for that feature (basis in this case)
		# This returns a series with symbols/instrumentIds as the index.
		currentValue = lookbackInstrumentValue.iloc[-1]

		if param1Value == 'value1':
			return currentValue * 0.1
		else:
			return currentValue * 0.5


if __name__ == "__main__":
    if updateCheck():
        print('Your version of the auquan toolbox package is old. Please update by running the following command:')
        print('pip install -U auquan_toolbox')
    else:
        print('Loading your config dicts and prediction function')

        tf = MyTradingFunctions()
        tf.varianceThreshold()
        
# =============================================================================
# 		print('Loaded config dicts and prediction function, Loading Problem Params')
# 		tsParams1 = MyTradingParams(tf)
# 		tsParams1.setTargetVariableKey('Revenue(Y)')
# 		# tsParams1.setInstrumentsIds([])
# 		tradingSystem = TradingSystem(tsParams1)
# 	
# 		results1 = tradingSystem.startTrading(onlyAnalyze=False, shouldPlot=False, makeInstrumentCsvs=False)
# =============================================================================
        
        
	
# =============================================================================
# 		tsParams2 = MyTradingParams(tf)
# 		tsParams2.setTargetVariableKey('Income(Y)')
# 		# tsParams2.setInstrumentsIds([])
# 		tradingSystem = TradingSystem(tsParams2)
# 	
# 		results2 = tradingSystem.startTrading(onlyAnalyze=False, shouldPlot=False, makeInstrumentCsvs=False)
# =============================================================================
		#print('Score: %0.3f'%(results1['score']+results2['score']))
