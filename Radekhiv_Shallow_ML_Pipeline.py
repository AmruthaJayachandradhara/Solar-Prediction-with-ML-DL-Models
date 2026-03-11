#%%------------------------------------------------------Radekhiv shallow ML pipeline-------------------------------------------------------------------------##
import pandas as pd
import os

os.listdir()

df = pd.read_csv('Radekhiv_Cleaned_Without_Nighttime.csv')

#Refamiliarize with dataset
print(df.head())
print(df.describe())
print(df.info())

#An issue with datetime is here since it's a string and not a datetime object
df['Datetime'] = pd.to_datetime(df['Datetime'], format = '%Y-%m-%d %H:%M:%S')
#Check if it worked
print(df.info()) #Should return datetime
print(df.head()) #Should look okay when inspecting

#Check columns
print(df.columns)

#Check what index is
print(df.index) #We see it isn't datetime

#Make sure datetime is chronologic order
if df['Datetime'].is_monotonic_increasing:
    print('Datetime is in chronological order') #returns true

'''Recall that the dataset had a large gap near the end of the dataset from 11/22/2023(5pm)-01/30/2024(9:00am). This has to be addressed before splitting'''
#Make sure period of time in the dataset doesn't have massive gaps or holes
print(df['Datetime'].diff(periods=1).value_counts()) #Mostly 1 hour or 2 hour gaps, but there is one large 68 day gap
#The 68 day gap is from 11/2023 to 1/30/2024, one way around this is to just have the period of observation not include this gap

# %%----------------------------------------------------------------------Splitting the dataset into training and testing sets---------------------------------------------------------------------------------##

'''Thanks to the large gap in the dataset, the train test split has to avoid the gap. The gap isn't from nulls, but unobserved/unreported data'''

#Make sure split isn't random and for temporal data, but also avoids gap from after 5pm on 11/22/2023 to 9am on 01/30/2024.

'''Since we're primarily focused on using weather and environmental features to predict generation and analyzing how these features interact, 
We are able to do a random split without having to worry about the 68 day gap in the dataset'''

#We will split for train test and validation
from sklearn.model_selection import train_test_split

#Random split is okay
df_train, df_test = train_test_split(df, test_size=0.15, random_state = 42)
#Split again for validation set, split will be 70/15/15 for train/validation/test
df_train, df_val = train_test_split(df_train, test_size = 0.1765, random_state = 42) 

#Check the shape of the splits for df
print(df_train.shape)
print(df_val.shape)
print(df_test.shape)

#%%-------------------------------------------------------------------Handle uncommon features-------------------------------------------------------------------------------------------

'''No uncommmon features since the dataset was already cleaned and came in one set unlike kaggle datasets that come in one set training one set testing'''
'''The shape horizontally/feature wise is also the same among all so we know there are no uncommon features'''


#%%-------------------------------------------------------------------Handle identifiers---------------------------------------------------------------------------------------------




#%%-------------------------------------------------------------------Handle date time variables---------------------------------------------------------------------------------------------




#%%-------------------------------------------------------------------Handle missing data--------------------------------------------------------------------------------------------------------

'''Missing data was handled in the cleaning phase before EDA'''


#%%-------------------------------------------------------------------Encoding the data--------------------------------------------------------------------------------------------------------------


#%%--------------------------------------------------------------------Splitting feature and target--------------------------------------------------------------------------------------------



#%%-----------------------------------------------------------------------Scaling the data---------------------------------------------------------------------------------------



#%%------------------------------------------------------------------------Feature selection and engineering----------------------------------------------------------------