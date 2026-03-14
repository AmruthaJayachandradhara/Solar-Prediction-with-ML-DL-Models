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
print(df['Datetime'].diff(periods=1).value_counts()) #This code was run in the cleaning file, it looks different here since night time hours were largely removed, so gaps look more prevalent now
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
print(df_train.shape) #4880 x 24
print(df_val.shape) #1046 x 24
print(df_test.shape) #1046 x 24

#%%-------------------------------------------------------------------Handle uncommon features-------------------------------------------------------------------------------------------
'''No uncommmon features since the dataset was already cleaned and came in one set unlike kaggle datasets that come in one set training one set testing'''
'''The shape horizontally/feature wise is also the same among all so we know there are no uncommon features'''

#Double check for safety

columns_train = df_train.columns
columns_val = df_val.columns
columns_test = df_test.columns

uncommon_feats_train_val = []
for col in columns_train: 
    if col not in columns_val:
        uncommon_feats_train_val.append(col)
print(uncommon_feats_train_val) #Returned empty list, so no uncommon features between train and val

uncommon_feats_train_test = []
for col in columns_train:
    if col not in columns_test:
        uncommon_feats_train_test.append(col)
print(uncommon_feats_train_test) #Returned empty list again so good on this end

uncommon_feats_test_val = []
for col in columns_test:
    if col not in columns_val:
        uncommon_feats_test_val.append(col)
print(uncommon_feats_test_val) #Empty again



#%%-------------------------------------------------------------------Handle identifiers---------------------------------------------------------------------------------------------
#Check if there are identifiers in the dataset
print(df_train.columns)
print(df_val.columns)
print(df_test.columns)

'''Next two loops are ran on df intentionally since the splits are all the same shape horizontally and uncommon features were checked'''
#Check what the index is of the dataframe
print(df.index) #Returns rangeindex so the index isn't an identifier that's mixed with features

#Loop to check if any columns look like identifiers
df_columns = df.columns
for col in df_columns: 
    if df[col].nunique() == len(df): 
        print(df[col]) #It only shows sunheight and datetime to be fitting this critera, sunheight was an engineered feature using datetime and pvlib library in the cleaning and EDA code file

'''All good for next step'''
#%%-------------------------------------------------------------------Handle date time variables---------------------------------------------------------------------------------------------
#For ML models the components of datetime have to be extracted and used as individual features

import datetime as dt
import numpy as np

#Extract the components of datetime
''' 
This is how to extract components of datetime 
--------------------------------------------------

df_train['year'] = df_train['Datetime'].dt.year
df_train['month'] = df_train['Datetime'].dt.month
df_train['day'] = df_train['Datetime'].dt.day
df_train['hour'] = df_train['Datetime'].dt.hour
df_train['minute'] = df_train['Datetime'].dt.minute
df_train['second'] = df_train['Datetime'].dt.second 
'''
#Check if datetime is in right dtype
print(df_train['Datetime'].dtype) #Should return datetime64[ns]

#Now that we know how to extract the components of datetime, let's create a function to try and apply it and simplify the process
def extract_datetime(df): 
    df['year'] = df['Datetime'].dt.year
    df['month'] = df['Datetime'].dt.month
    df['day'] = df['Datetime'].dt.day
    df['hour'] = df['Datetime'].dt.hour
    df['minute'] = df['Datetime'].dt.minute
    df['second'] = df['Datetime'].dt.second
    df.drop(columns=['Datetime'], inplace = True)
    return df

#Apply to splits to check if it works
df_train = extract_datetime(df_train)
df_val = extract_datetime(df_val)
df_test = extract_datetime(df_test)

#Safety check to see if all splits are still good for shape
print(df_train.shape) #4880 x 29
print(df_val.shape) #1046 x 29
print(df_test.shape) #1046 x 29

'''It is worth noting that features like second and minute are not useful since our data is hourly
Features like day and year might also have little to no contribution in our modeling however,
I will handle these features in the feature selection and engineering stage for a clean and easy to follow pipeline'''




#%%-------------------------------------------------------------------Handle missing data--------------------------------------------------------------------------------------------------------

'''Missing data was handled in the cleaning phase before EDA'''

#Double check for safety 
print(df_train.isnull().sum())
print(df_val.isnull().sum())
print(df_test.isnull().sum()) #None missing for all 3 splits


#%%-------------------------------------------------------------------Encoding the data--------------------------------------------------------------------------------------------------------------

#There are some categorical features in the dataset, so these have to be encoded

def column_check(df):
    possible_categorical = []
    for col in df.columns:
        if df[col].dtype != 'float64' and df[col].dtype != 'int64' and df[col].dtype != 'int32' and df[col].dtype != 'float32':
            possible_categorical.append(col)
    return possible_categorical

column_check(df_train)

#Returns conditions, icon, season(our own engineered feature) but I will double check by checking all columns and metadata
print(df_train.dtypes)

'''According to the metadata, these are the only non-numeric features before one hot encoding let's take a look at the unique value count to see if it'll be a good idea to one hot encode'''

print(df_train['conditions'].nunique()) #10 unique values
print(df_train['icon'].nunique()) #8 unique values
print(df_train['season'].nunique()) #Season is four since I remember from engineering it

'''All three features don't have a ridiculous amount of unique values so one hot encoding isn't awful''' 

#Check values of icon and conditions to see if they're similar
print(df_train['conditions'].value_counts())
print(df_train['icon'].value_counts())

#both features are pretty similar in what they describe, so it's probably better to drop one before one hot encoding
'''Looking at the value counts for each, they don't map to each other perfectly 1to1, but they're similar enough. Conditions seems to be more descriptive so I will drop icon'''
df_train = df_train.drop(columns = ['icon'])
df_val = df_val.drop(columns = ['icon'])
df_test = df_test.drop(columns = ['icon'])

#Check if drop is successful
print(df_train.columns)
print(df_val.columns)
print(df_test.columns)

#Temporarily combine all splits to encode and then split again to avoid any issues with one hot encoding
len_train = len(df_train) #4880
len_val = len(df_val) #1046
len_test = len(df_test) #1046

df_combo = pd.concat([df_train, df_val, df_test], axis = 0, ignore_index = True)
print(df_combo.shape) #6972 x 28


#One hot encode season and conditions 
df_combo = pd.get_dummies(df_combo, columns = ['conditions', 'season'], drop_first = True)

#Check if it worked
print(df_combo.head())
print(df_combo.shape) #39 columns now

#Resplit the dataset back into train, val, and test
df_train = df_combo.iloc[:len_train, :]
df_val = df_combo.iloc[len_train:len_train+len_val, :]
df_test = df_combo.iloc[len_train+len_val:, :]

#Check if shape maintained
print(df_train.shape) #4880 x 39
print(df_val.shape) #1046 x 39
print(df_test.shape) #1046 x 39

#%%--------------------------------------------------------------------Splitting feature and target--------------------------------------------------------------------------------------------
#Here is where splitting feature and target will be handled
X_train = df_train.drop(columns = ['generation'])
X_val = df_val.drop(columns = ['generation'])
X_test = df_test.drop(columns = ['generation'])

#y splits
y_train = df_train['generation'].values
y_val = df_val['generation'].values
y_test = df_test['generation'].values

#Print shape of splits for safety
print(X_train.shape) #4880 x 38
print(X_val.shape) #1046 x 38
print(X_test.shape) #1046 x 38

print(y_train.shape) #4880
print(y_val.shape) #1046
print(y_test.shape) #1046

#Check y splits wwere also made into numpy
print(type(y_train)) #numpy.ndarray
print(type(y_val)) #numpy.ndarray   
print(type(y_test)) #numpy.ndarray

#%%-----------------------------------------------------------------------Scaling the data---------------------------------------------------------------------------------------
#The last step before feature selection and engineering is scaling the data. 

'''regression type models require scaling for both feature and target'''

from sklearn.preprocessing import StandardScaler #Given context of our data we're going to go with standard scaler over min max

scaler = StandardScaler()

#Scaling will be done on the continuous features only, so we reuse function from earlier to identify continuous features instead this time
'''We remember from that function that only conditions, season, icon were categorical'''



continuous_columns = [col for col in X_train.columns if not col.startswith('season_') and not col.startswith('conditions_')]


#Fit onto training
X_train_scaled = scaler.fit_transform(X_train[continuous_columns])

#Transform val and test
X_val_scaled = scaler.transform(X_val[continuous_columns])
X_test_scaled = scaler.transform(X_test[continuous_columns])


'''Correct this so that binary columns are added back during next work session. '''
# #Add back the categorical features to the scaled continuous features
# X_train_scaled = pd.DataFrame(X_train_scaled, columns = continuous_columns, index = X_train.index)
# X_val_scaled = pd.DataFrame(X_val_scaled, columns = continuous_columns, index = X_val.index)
# X_test_scaled = pd.DataFrame(X_test_scaled, columns = continuous_columns, index = X_test.index)


#%%------------------------------------------------------------------------Feature selection and engineering----------------------------------------------------------------

