import pandas as pd
import numpy as np

from sklearn.metrics import r2_score

class Utils:

	category_types = {'category'}
	
	@staticmethod
	def replace_numeric_missed_values(y):
		numeric_data = y._get_numeric_data()
		numeric_data.fillna(0,inplace=True)
		columns = numeric_data.columns.values.tolist()
		y[columns] = numeric_data
		
	@staticmethod
	def replace_categorical_missed_values(y,replace_val):
		categorical_data = y.select_dtypes(include=Utils.category_types)
		replacedMissed = categorical_data.apply(lambda x: x.str.strip()).replace(np.nan, replace_val)
		replacedMissed = replacedMissed.apply(lambda x: x.str.strip()).replace('', replace_val)		
		columns = replacedMissed.columns.values.tolist()
		y[columns] = replacedMissed

	@staticmethod
	def cal_r_squared(variable,target):
		return r2_score(target,variable)
		