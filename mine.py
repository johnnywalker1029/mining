import pandas as pd
import numpy as np
import sklearn.metrics import r2_source

from lib.util import Utils as utils

data_type = {
'ODATEDW':'category',
'TCODE':'category',
'STATE':'category',
'ZIP':'category',
'MAILCODE':'category',
'PVASTATE':'category',
'NOEXCH':'category',
'RECINHSE':'category',
'RECP3':'category',
'RECPGVG':'category',
'RECSWEEP':'category',
'MDMAUD':'category',
'DOMAIN':'category',
'CLUSTER2':'float'}

raw_data_path = 'C:/WorkDir/dataMining/raw_data/'
learn_data_path = raw_data_path + 'cup98LRN.txt'
validData = raw_data_path + 'cup98VAL.txt'

learn_data_df = pd.read_csv(learn_data_path,dtype=data_type)

#print(learn_data_df.head())
#result = learn_data_df.TARGET_B
utils.replace_numeric_missed_values(learn_data_df)
utils.replace_categorical_missed_values(learn_data_df,'Missed')

AGE_Rsquare = utils.cal_r_squared(learn_data_df.AGE,learn_data_df.TARGET_B)
print(AGE_Rsquare)
