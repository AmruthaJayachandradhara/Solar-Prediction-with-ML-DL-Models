#%%------------------------------------------------------Radekhiv shallow ML pipeline-------------------------------------------------------------------------##
'''This is the pipeline for the Radekhiv location using the csv with solarradiation < 10 filtered out and a random split since the focus is on how features interact with generation and not time, so temporal order isn't as important'''


import pandas as pd
import os

os.listdir()

df = pd.read_csv('Radekhiv_Cleaned_Without_Nighttime_2.csv')

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

df = df.drop(columns = ['index']) #Reset index to be a rangeindex since the index isn't datetime and we don't need it to be

print(df.shape)

#Make sure datetime is chronologic order
if df['Datetime'].is_monotonic_increasing:
    print('Datetime is in chronological order') #returns true

'''Recall that the dataset had a large gap near the end of the dataset from 11/22/2023(5pm)-01/30/2024(9:00am). For standard time series modeling this would have needed to be addressed
For the purposes of this pipeline with time being temporal and not the focus, I don't think it needs to be addressed.'''
#Make sure period of time in the dataset doesn't have massive gaps or holes
print(df['Datetime'].diff(periods=1).value_counts()) #This code was run in the cleaning file, it looks different here since night time hours were largely removed, so gaps look more prevalent now
#The 68 day gap is from 11/2023 to 1/30/2024, one way around this is to just have the period of observation not include this gap


#inspect the hours in the dataset
print(df['Datetime'].dt.hour.value_counts()) #Should be mostly daytime hours

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
print(df_train.shape) #4880 x 24 if filter was solarradiation = 0 were removed in cleaning
print(df_val.shape) #1046 x 24  if filter was solarradiation = 0 were removed in cleaning
print(df_test.shape) #1046 x 24 if filter was solarradiation = 0 were removed in cleaning

#New split with new filter from cleaning will be 
#4333 x 24
#929 x 24
#929 x 24


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

# #Shows index as an identifier
# df_train = df_train.drop(columns = ['index'])
# df_val = df_val.drop(columns = ['index'])
# df_test = df_test.drop(columns = ['index'])


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

#New shape following new of solarradiation > 10 filter is this 
# (4333, 29)
# (929, 29)
# (929, 29)


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

#New shape following new solarradiation filter is this 
# (4333, 38)
# (929, 38)
# (929, 38)
# (4333,)
# (929,)
# (929,)
#%%

import matplotlib.pyplot as plt
#Going to try and log transform everything but the target, date features, and encoded features to see if it improves model performance
#This is what was done in the Shakhovska paper

features_for_transformation = ['temp', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob', 'snow',
       'snowdepth', 'windgust', 'windspeed', 'winddir', 'sealevelpressure',
       'cloudcover', 'visibility', 'solarradiation', 'solarenergy', 'uvindex',
       'severerisk', 'sunheight']
print(len(features_for_transformation)) 

#Apply log transform to features features_for_transformation for each X split
# for col in features_for_transformation:
#     X_train[col] = np.log1p(X_train[col])
#     X_val[col] = np.log1p(X_val[col])
#     X_test[col] = np.log1p(X_test[col])
#%%-----------------------------------------------------------------------Scaling the data---------------------------------------------------------------------------------------
#The last step before feature selection and engineering is scaling the data. 

'''regression type models require scaling for continuous featurs'''

from sklearn.preprocessing import StandardScaler #Given context of our data we're going to go with standard scaler over min max

scaler = StandardScaler()

#Scaling will be done on the continuous features only, so we reuse function from earlier to identify continuous features instead this time
'''We remember from that function that only conditions, season, icon were categorical'''

continuous_columns = [col for col in X_train.columns if not col.startswith('season_') and not col.startswith('conditions_')]

#Other columns are binary thanks to one hot encoding done before
binary_columns = [col for col in X_train.columns if col not in continuous_columns]

#Fit onto training
X_train_scaled = scaler.fit_transform(X_train[continuous_columns])

#Transform val and test
X_val_scaled = scaler.transform(X_val[continuous_columns])
X_test_scaled = scaler.transform(X_test[continuous_columns])

#Make df again
X_train_scaled = pd.DataFrame(X_train_scaled, columns = continuous_columns, index = X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns = continuous_columns, index = X_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = continuous_columns, index = X_test.index)

#Add back the encoded features to get the complete dataframe again
X_train_scaled = pd.concat([X_train_scaled, X_train[binary_columns]], axis = 1)
X_val_scaled = pd.concat([X_val_scaled, X_val[binary_columns]], axis = 1)
X_test_scaled = pd.concat([X_test_scaled, X_test[binary_columns]], axis = 1)

#Check shapes to make sure everything is as expected
print(X_train_scaled.shape) #Should return 4880 x 38
print(X_val_scaled.shape) #Should return 1046 x 38
print(X_test_scaled.shape)#Should return 1046x38

#New shape following new solarradiation filter is this 
# (4333, 38)
# (929, 38)
# (929, 38)


#%%------------------------------------------------------------------------Feature selection and engineering----------------------------------------------------------------
from sklearn.feature_selection import VarianceThreshold

#As mentioned earlier, some features are obvously not useful like second and minute since the data is hourly, also year only has 3 unique
print(df_combo.nunique()) #We can see year only has 3 different values while features like second and minute only have one

#Drop the most obcious features from splits
X_train_scaled = X_train_scaled.drop(columns = ['minute', 'second', 'year', 'day']) # i understand that year has 3 observations, but year our data isn't uniform for every year, and we're focused on environmental features anyway
X_val_scaled = X_val_scaled.drop(columns = ['minute', 'second', 'year', 'day'])
X_test_scaled = X_test_scaled.drop(columns = ['minute', 'second', 'year', 'day'])

#Splits will have 34 features now!

'''
Explanation for these being dropped
-----------------------------------------------
Features like minute and second are dropped since the data is hourly, so they have no contribution to the model since they will always be zero.
Year is dropped since the data isn't uniform every year, for exampleour dataset starts on June 22 2022 and ends February 2024. The model might learn that some years produce less production which isn't the goal.
The goal is for how environmental features interact with generation. Day is dropped since again, when just looking at day, it doesn't provide context for generation.
Think of it like this, if we were to compare generation every day 1 of the dataset to every day 21, it wouldn't make sense. Day doesn't influence how generation fluctuates, time of day and month do because that tells us about
things like seasonality, sunheight, etc.
'''


'''
Note: These are a few different methods for feature selection
----------------------
Using SelectKBest is an option for filter methods
Wrappermethods likeRFE are an option
Embedded methods like LASSO

More technical drops or feature selection will be done from here onward

'''

#Check variances of all features
feature_variances = X_train_scaled.var()
print(feature_variances)

'''There are a few features that have very low variance, specifically parts of the conditions_ features. Features like 'conditions_Rain', 'conditions_Snow, Rain, Partially Cloudy'

'''

#Check variance and create threshold
selector = VarianceThreshold(threshold = 0.001) #Chose 0.001 to be really safe and avoid dropping possibly important features since we don't have a ton

#Fit on training data to check variance of features
selector.fit(X_train_scaled)

#Don't transform since it'll turn features into numpy arrays and I need df for multicolinearity checks, we can use get_support to see which features are kept and which are dropped
features_kept = X_train_scaled.columns[selector.get_support()]
print(features_kept)
print(len(features_kept)) #Returns 31 so 5 were dropped

features_dropped = [col for col in X_train_scaled.columns if col not in features_kept]
print(features_dropped)
print(len(features_dropped))

#Officially drop features from features_dropped in all splits
X_train_scaled = X_train_scaled.drop(columns = features_dropped)
X_val_scaled = X_val_scaled.drop(columns = features_dropped)
X_test_scaled = X_test_scaled.drop(columns = features_dropped)

#Check shape of splits
print(X_train_scaled.shape) #Should return 4880 x 31
print(X_val_scaled.shape) #Should return 1046 x 31  
print(X_test_scaled.shape)#They all return expected shape

#Safety check
print(X_train_scaled.columns)

#Multicollinearity check
'''For mutlicollinearity, we will check continuous features'''
#Create new list of binary columns since some were dropped
binary_columns_updated = [col for col in X_train_scaled.columns if col.startswith('season_') or col.startswith('conditions_')]
print(binary_columns_updated) 
print(len(binary_columns_updated)) #Now 10 features

#Make sure all continous ones are just the ones not in binary
Continuous_features = [col for col in X_train_scaled.columns if col not in binary_columns_updated] #Want to get all features besides binary features
print(Continuous_features)
print(len(Continuous_features)) #Now 21 features so totals line up perfect

#Get correlation values for continuous features
X_train_scaled_continuous = X_train_scaled[Continuous_features]
correlation_matrix = X_train_scaled_continuous.corr()
print(correlation_matrix)

#make plot 
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (20, 20))
sns.heatmap(correlation_matrix, annot = True, cmap = 'RdBu', square = True)
print(correlation_matrix) 

#Filter to see which features have high correlation

highly_correlated_features = []
for col in correlation_matrix.columns:
    for row in correlation_matrix.index:
        if abs(correlation_matrix.loc[row, col]) > 0.7:
            highly_correlated_features.append((row, col, correlation_matrix.loc[row, col]))

print(highly_correlated_features)

'''
The following unique feature pairs are highly correlated:
---------------------------------------------------------------
'feelslike', 'temp', np.float64(0.9943068212405849)
('dew', 'temp', np.float64(0.832256053844893))
('dew', 'feelslike', np.float64(0.8449100534661185))
('windspeed', 'windgust', np.float64(0.9143434255475628))
('solarenergy', 'solarradiation', np.float64(0.9995203966356911))
('uvindex', 'solarradiation', np.float64(0.9943059402222983))
('uvindex', 'solarenergy', np.float64(0.993859254434418))
('hour', 'sunheight', np.float64(-0.7957456532047044))
'''
#Check which of the features in these feature pairs best correlate with target
'''Using the correlation matrix from the cleaning file, here are notes about features highly correlated with each other and how it matches with target
----------------------------------------------------------------------------------------------------------------------------------------------------
('feelslike', 'temp') -> temp is just slightly more correlated with target
('dew', 'temp') -> temp is more correlated with target than dew
('dew', 'feelslike') -> either one or both will have been dropped already anyway so no need to compare
('windspeed', 'windgust') ----> windspeed is more correlated with target
('solarenergy', 'solarradiation')------------> solarradiation is a must keep
('uvindex', 'solarradiation') ----------------> solarradiation is a must keep
('uvindex', 'solarenergy') --------------------> solarenergy would have already been dropped
('hour', 'sunheight') --------------------------> sunheight was engineered, will keep both
'''

#Collinearity feature drops
features_to_drop_collinearity = ['feelslike', 'windgust', 'solarenergy', 'uvindex']
X_train_scaled = X_train_scaled.drop(columns = features_to_drop_collinearity)
X_val_scaled = X_val_scaled.drop(columns = features_to_drop_collinearity)
X_test_scaled = X_test_scaled.drop(columns = features_to_drop_collinearity)

#Check if features were dropped
print(X_train_scaled.shape) #Should return 4880 x 27
print(X_val_scaled.shape) #Should return 1046 x 27
print(X_test_scaled.shape) #Should return 1046 x 27

#Drop dew too, (mistakenly left out)
X_train_scaled = X_train_scaled.drop(columns = ['dew'])
X_val_scaled = X_val_scaled.drop(columns = ['dew'])
X_test_scaled = X_test_scaled.drop(columns = ['dew'])

#Check if features were dropped
print(X_train_scaled.shape) #Should return 4880 x 26
print(X_val_scaled.shape) #Should return 1046 x 26
print(X_test_scaled.shape) #Should return 1046 x 26

#%%----------------------------------------------------------------------Model and parameter grids---------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

models = {
    'lr': LinearRegression(),
    'rfr': RandomForestRegressor(),
    'XGB': XGBRegressor(),
    'svr': SVR(),
    'catboost': CatBoostRegressor(verbose = 0, allow_writing_files = False),
    'knn': KNeighborsRegressor()
}
#%%----------------------------------------------------------------------Pipelines---------------------------------------------------------------------------------------------
pipes = {}
for acronym, model in models.items():
    pipes[acronym] = Pipeline([('model', model)])

# %%----------------------------------------------------------------------Parameter grids----------------------------------------------------------------------------------------
param_grids = {}

#%%-------------------------------------------------------------------------Linear regression parameters----------------------------------------------------------------------------------- 

param_grids['lr'] = {
}

#%%-------------------------------------------------------------------------Random Forest Regressor parameters-----------------------------------------------------------------------------------
param_grids['rfr'] = {
     'model__n_estimators': [100, 200],
     'model__max_depth': [10, 20, None],
     'model__min_samples_split': [2, 5, 10]
     }


#%%--------------------------------------------------------------------------XGBoost parameters-----------------------------------------------------------------------------------
param_grids['XGB'] = {
     'model__learning_rate': [0.01, 0.1],
     'model__n_estimators': [100, 200],
     'model__max_depth': [3, 5, 7]
}
     

#%%--------------------------------------------------------------------------SVR parameters-----------------------------------------------------------------------------------
param_grids['svr'] = {
    'model__tol': [1e-3, 1e-4, 1e-5],
    'model__C': [0.1, 1, 10],
    'model__epsilon': [0.01, 0.1, 0.5],
    'model__kernel': ['linear', 'rbf'],
}

#%%--------------------------------------------------------------------------CatBoost parameters-----------------------------------------------------------------------------------
param_grids['catboost'] = {
    'model__learning_rate': [0.01, 0.1],
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5, 7]
}

#%%--------------------------------------------------------------------------KNN parameters-----------------------------------------------------------------------------------
param_grids['knn'] = {
    'model__n_neighbors': [3, 5, 7],
    'model__weights': ['uniform', 'distance'],
    'model__metric': ['euclidean', 'manhattan']
}
#%%--------------------------------------------------------------------------Grid search and model training-----------------------------------------------------------------------------------
#Cross validation and hyperparameter tuning will be done using GridSearchCV for each model in the pipeline
grid_searches = {}
model_predictions = {}
best_models = {}
best_parameters = {}

for acronym, pipe in pipes.items(): 
    grid_search = GridSearchCV(pipe, param_grids[acronym], cv=5, scoring= ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'], refit = 'neg_root_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_models[acronym] = grid_search.best_estimator_
    best_parameters[acronym] = grid_search.best_params_
    #Store grid search results in dictionary
    grid_searches[acronym] = grid_search

#%%--------------------------------------------------------------------------Model evaluation on validation-----------------------------------------------------------------------------------
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


for acronym, gs in grid_searches.items():
    print(f"\n--- Results for {acronym.upper()} ---")
    
    #Best parameters and CV score
    print(f"Best Parameters: {gs.best_params_}")
    print(f"Best CV RMSE Score: {abs(gs.best_score_):.4f}")
    
    #Predict
    y_pred_val = gs.best_estimator_.predict(X_val_scaled)
    model_predictions[acronym] = y_pred_val

    #Confirm prediction shape
    print(f"Prediction shape: {y_pred_val.shape} | True labels shape: {y_val.shape}")

    #Performance Report
    print("Metric scoring for the best model:")
    print("r2 Score:", r2_score(y_val, y_pred_val))
    print("Mean Absolute Error:", mean_absolute_error(y_val, y_pred_val))
    print("Root Mean Squared Error:", root_mean_squared_error(y_val, y_pred_val))

#Create dataframe showing all models and their performance metrics
results_df = pd.DataFrame(data = {
    'Model': list(grid_searches.keys()),
    'Best Parameters': [best_parameters[acronym] for acronym in grid_searches.keys()],
    'CV RMSE Score': [abs(grid_searches[acronym].best_score_) for acronym in grid_searches.keys()],
    'Validation R2 Score': [r2_score(y_val, model_predictions[acronym]) for acronym in grid_searches.keys()],
    'Validation MAE': [mean_absolute_error(y_val, model_predictions[acronym]) for acronym in grid_searches.keys()],
    'Validation RMSE': [root_mean_squared_error(y_val, model_predictions[acronym]) for acronym in grid_searches.keys()]
})

#%%--------------------------------------------------------------------------Model evaluation on test-----------------------------------------------------------------------------------
for acronym, gs in grid_searches.items():
    print(f"\n--- Results for {acronym.upper()} ---")
    
    #Best parameters and CV score
    print(f"Best Parameters: {gs.best_params_}")
    print(f"Best CV RMSE Score: {gs.best_score_:.4f}")
    
    #Predict
    y_pred_test = gs.best_estimator_.predict(X_test_scaled)
    model_predictions[acronym] = y_pred_test

    #Confirm prediction shape
    print(f"Prediction shape: {y_pred_test.shape} | True labels shape: {y_test.shape}")

    #Performance Report
    print("Metric scoring for the best model:")
    print("r2 Score:", r2_score(y_test, y_pred_test))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_test))
    print("Root Mean Squared Error:", root_mean_squared_error(y_test, y_pred_test))

#%%--------------------------------------------------------------------------Final importance analysis-----------------------------------------------------------------------------------
 #Derive feature importance for Catboost model
catboost_model = best_models['catboost']
feature_importance_catboost = catboost_model.named_steps['model'].get_feature_importance()

#Print feature importances and analyze
feature_importance_catboost_df = pd.DataFrame({
    'Features' : X_train_scaled.columns, 
    'Importances': feature_importance_catboost
})
print(feature_importance_catboost_df.sort_values(by = 'Importances', ascending = False))



#Derive things like feature importance, permutation importance and other analysis
xgb_model = best_models['XGB']
feature_importances = xgb_model.named_steps['model'].feature_importances_

#Print feature importances and analyze
feature_importance_df = pd.DataFrame({
    'Features' : X_train_scaled.columns, 
    'Importances': feature_importances
})

print(feature_importance_df.sort_values(by = 'Importances', ascending = False))

#%%--------------------------------------------------------------------------Create csvs for Tableau visualizations---------------------------------------------
#Create csv for actual vs predicted values for all models on test set
rows = []
for model_name, predictions in model_predictions.items():
    for i in range(len(y_test)):
        rows.append({
            'Dataset': 'Radekhiv',
            'Actual': y_test[i],
            'Predicted': predictions[i],
            'Model': model_name.upper(),
            'month': df_test.iloc[i]['month'],  
            'hour': df_test.iloc[i]['hour'],
            'Error': y_test[i] - predictions[i]
        })

test_results_df = pd.DataFrame(rows)
test_results_df.to_csv('test_results_Radekhiv.csv', index = False)

#Create csv for model performance on test
model_performance_df = pd.DataFrame({
    'Model': list(grid_searches.keys()),
    'Dataset': 'Radekhiv',
    'CV RMSE Score': [abs(grid_searches[acronym].best_score_) for acronym in grid_searches.keys()],
    'Validation R2 Score': [r2_score(y_val, model_predictions[acronym]) for acronym in grid_searches.keys()],
    'Validation MAE': [mean_absolute_error(y_val, model_predictions[acronym]) for acronym in grid_searches.keys()],
    'Validation RMSE': [root_mean_squared_error(y_val, model_predictions[acronym]) for acronym in grid_searches.keys()],
    'Test R2 Score': [r2_score(y_test, model_predictions[acronym]) for acronym in grid_searches.keys()],
    'Test MAE': [mean_absolute_error(y_test, model_predictions[acronym]) for acronym in grid_searches.keys()],
    'Test RMSE': [root_mean_squared_error(y_test, model_predictions[acronym]) for acronym in grid_searches.keys()]
})

model_performance_df.to_csv('model_performance_Radekhiv.csv', index = False)
#Create csv for feature importance for best performing model

feature_importance_catboost_df['Dataset'] = 'Radekhiv'
feature_importance_catboost_df['Model'] = 'CatBoost'
feature_importance_catboost_df.to_csv('feature_importance_catboost_Radekhiv.csv', index = False)


# %%
