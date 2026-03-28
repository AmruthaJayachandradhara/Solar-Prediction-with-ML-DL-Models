#%%----------------------------------------------------------------------Importing dataset and inspection----------------------------------------------------------------------------------##

'''This pipeline is for the Rajasthan, India dataset location '''

import pandas as pd
import os

os.listdir()
os.chdir("C:\\Users\\shime\\Downloads")
os.listdir()

df = pd.read_csv('preprocess_base.csv')

#Refamiliarize with dataset
print(df.head())
print(df.describe())
print(df.info())

#Inspect
print(df.info()) #Should return datetime
print(df.head()) #Should look okay when inspecting

#Check columns
print(df.columns)

#Check what index is
print(df.index) #We see it isn't datetime

# df = df.drop(columns = ['index']) #Reset index to be a rangeindex since the index isn't datetime and we don't need it to be

print(df.shape) #17544 x 24

#Make sure datetime is chronologic order
if df['datetime'].is_monotonic_increasing:
    print('datetime is in chronological order') #returns true

#Inspect datetime column to see if there are any issues with it
print(df['datetime'].head(25)) #Looks good but we just need y/m/d h:m:s

#Convert datetime into datetime dtype
df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)#Removing timezone since it's not necessary

#Check if i worked
print(df['datetime'].dtype) #Should return datetime64[ns]
print(df['datetime'].head()) #Should look the same but in datetime format

#Make sure period of time in the dataset doesn't have massive gaps or holes
print(df['datetime'].diff(periods=1).value_counts()) #Appears that there are no gaps larger than an hour, we also know dataset is hourly


'''





'''
#%%---------------------------------------------------------------------------Light preprocessing-----------------------------------------------------------------------
#Check columns
print(df.columns)

#Create season feature the same way I did in Radekhiv dataset, but adjusted for the seasons of Rajasthan
df['season'] = None

df['season'] = df['season'].fillna(df['datetime'].dt.month.isin([12, 1, 2]).map({True: 'Winter', False: None}))
df['season'] = df['season'].fillna(df['datetime'].dt.month.isin([3,4,5]).map({True: 'Summer', False: None}))
df['season'] = df['season'].fillna(df['datetime'].dt.month.isin([6,7,8, 9]).map({True: 'Monsoon', False: None}))
df['season'] = df['season'].fillna(df['datetime'].dt.month.isin([10,11]).map({True: 'Post-Monsoon/Fall', False: None}))

#Check if it worked
print(df['season'].value_counts()) #Looks good, the seasons align with what we can find about Rajasthan

#Check the hour value counts when is_daytime is 0 (nighttime hours)
print(df[df['is_daytime'] == 0]['datetime'].dt.hour.value_counts())
print(df[df['is_daytime'] == 0]['datetime'].dt.hour.value_counts().sum()) #8757 is considered night time (when G_i = 0)

#Check the hour value counts when is_daytime is 1 (daytime hours)
print(df[df['is_daytime'] == 1]['datetime'].dt.hour.value_counts())
print(df[df['is_daytime'] == 1]['datetime'].dt.hour.value_counts().sum()) #8787 is considered daytime (when G_i > 0)


#There seems to be something wrong with is_daytime feature so I will be doing a more aggressive filter similar to Radekhiv location
df.drop(df[df['G_i'] < 10].index, inplace = True) #This is a filter used in the Radekhiv location
print(len(df)) #Using same filter as Radekhiv has left us with 8672 rows now

#Check again 
print(df['datetime'].dt.hour.value_counts()) #Check there are less nighttime hours

'''Some of the features have different units than Radekhiv dataset, most differences won't matter because of scaling but below I'll fix it'''

#Fix pressure
df['SP'] = df['SP'] / 100 #Convert from Pa to hPa to match Radekhiv dataset

#Check min and max of target
print(df['P'].min(), df['P'].max()) #Min is 0 and max is 845.35

#Convert target to kWh
df['P'] = df['P'] / 1000 #Convert from W to kWh to match Radekhiv dataset

'''It's worth noting that windspeed and windgust here are m/s while in Radekhiv it was a bit unspecified, scaling should handle this anyway'''

#Rename columns so that pipeline needs less adjustments
df.rename(columns = {
    'datetime': 'Datetime', 
    'P': 'generation', 
    'SP': 'sealevelpressure',
    'T2': 'temperature', 
    'WS10m': 'windspeed',
    'WD10m': 'winddirection',
    'RH': 'humidity',
    'H_sun': 'sunheight'
    }, inplace = True)

# %%----------------------------------------------------------------------Splitting the dataset into training and testing sets---------------------------------------------------------------------------------##
#We will split for train test and validation
from sklearn.model_selection import train_test_split

#Random split is okay
df_train, df_test = train_test_split(df, test_size=0.15, random_state = 42)
#Split again for validation set, split will be 70/15/15 for train/validation/test
df_train, df_val = train_test_split(df_train, test_size = 0.1765, random_state = 42) 

#Check the shape of the splits for df
print(df_train.shape) #6070 x 25
print(df_val.shape) #1301 x 25
print(df_test.shape) #1301 x 25



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
        print(df[col]) #It only shows only datetime to be fitting this critera

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
print(df_train.shape) #6070 x 28
print(df_val.shape) #1301 x 28
print(df_test.shape) #1301 x 28

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

#Returns season(our own engineered feature) but I will double check by checking all columns and metadata
print(df_train.dtypes)

'''According to the metadata, these are the only non-numeric features before one hot encoding let's take a look at the unique value count to see if it'll be a good idea to one hot encode'''
print(df_train['season'].nunique()) #Season is four since I remember from engineering it


#Temporarily combine all splits to encode and then split again to avoid any issues with one hot encoding
len_train = len(df_train) #6070
len_val = len(df_val) #1301
len_test = len(df_test) #1301

df_combo = pd.concat([df_train, df_val, df_test], axis = 0, ignore_index = True)
print(df_combo.shape) #8762 x 28


#One hot encode season and conditions 
df_combo = pd.get_dummies(df_combo, columns = ['season'], drop_first = True)

#Check if it worked
print(df_combo.head())
print(df_combo.shape) #30 columns now

#Resplit the dataset back into train, val, and test
df_train = df_combo.iloc[:len_train, :]
df_val = df_combo.iloc[len_train:len_train+len_val, :]
df_test = df_combo.iloc[len_train+len_val:, :]

#Check if shape maintained
print(df_train.shape) #6070 x 30
print(df_val.shape) #1301 x 30
print(df_test.shape) #1301 x 30

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
print(X_train.shape) #6070 x 29
print(X_val.shape) #1301 x 29
print(X_test.shape) #1301 x 29

print(y_train.shape) #6070
print(y_val.shape) #1301
print(y_test.shape) #1301

#Check y splits wwere also made into numpy
print(type(y_train)) #numpy.ndarray
print(type(y_val)) #numpy.ndarray   
print(type(y_test)) #numpy.ndarray

#%%

'''Ignore'''

# import matplotlib.pyplot as plt
# #Going to try and log transform everything but the target, date features, and encoded features to see if it improves model performance
# #This is what was done in the Shakhovska paper

# features_for_transformation = ['temp', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob', 'snow',
#        'snowdepth', 'windgust', 'windspeed', 'winddir', 'sealevelpressure',
#        'cloudcover', 'visibility', 'solarradiation', 'solarenergy', 'uvindex',
#        'severerisk', 'sunheight']
# print(len(features_for_transformation)) 

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
'''We remember from that function that only season was categorical'''
#check
print(X_train.columns)

continuous_columns = [col for col in X_train.columns if not col.startswith('season_')]
print(continuous_columns)
print(len(continuous_columns)) #26 continuous features

#Other columns are binary thanks to one hot encoding done before
binary_columns = [col for col in X_train.columns if col not in continuous_columns]
print(binary_columns)
print(len(binary_columns)) #3 binary features, season_Summer, season_Winter, season_Post-Monsoon/Fall

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
print(X_train_scaled.shape) #Should return 6070 x 29
print(X_val_scaled.shape) #Should return 1301 x 29
print(X_test_scaled.shape)#Should return 1301x 29



#%%------------------------------------------------------------------------Feature selection and engineering----------------------------------------------------------------
from sklearn.feature_selection import VarianceThreshold

#As mentioned earlier, some features are obvously not useful like second and minute since the data is hourly, also year only has 3 unique
print(df_combo.nunique()) #We can see year only has 3 different values while features like second and minute only have one

#Drop the most obcious features from splits
X_train_scaled = X_train_scaled.drop(columns = ['minute', 'second', 'year', 'day']) # i understand that year has 3 observations, but year our data isn't uniform for every year, and we're focused on environmental features anyway
X_val_scaled = X_val_scaled.drop(columns = ['minute', 'second', 'year', 'day'])
X_test_scaled = X_test_scaled.drop(columns = ['minute', 'second', 'year', 'day'])

#Check shapes
print(X_train_scaled.shape) #6070 x 25
print(X_val_scaled.shape) #1301 x 25
print(X_test_scaled.shape) #1301 x 25
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
print(X_train_scaled.shape) #Should return 6070 x 23
print(X_val_scaled.shape) #Should return 1301 x 23  
print(X_test_scaled.shape)#They all return expected shape

#Safety check
print(X_train_scaled.columns)

#Multicollinearity check
'''For mutlicollinearity, we will check continuous features'''
#Create new list of binary columns since some were dropped
binary_columns_updated = [col for col in X_train_scaled.columns if col.startswith('season_')]
print(binary_columns_updated) 
print(len(binary_columns_updated)) #Now 3 features

#Make sure all continous ones are just the ones not in binary
Continuous_features = [col for col in X_train_scaled.columns if col not in binary_columns_updated] #Want to get all features besides binary features
print(Continuous_features)
print(len(Continuous_features)) #Now 20 features so totals line up perfect

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
'Gb_i', 'G_i', np.float64(0.9832594658048074)
('Gr_i', 'G_i', np.float64(0.9399815273812983))
('sunheight', 'G_i', np.float64(0.7302327084572351))
('hour_sin', 'G_i', np.float64(0.8563236610077146))
('is_peak_hours', 'G_i', np.float64(0.7532442021124596))
('G_i', 'Gb_i', np.float64(0.9832594658048074))
('Gr_i', 'Gb_i', np.float64(0.8962812807031522))
('hour_sin', 'Gb_i', np.float64(0.7876236085844002))
('is_peak_hours', 'Gb_i', np.float64(0.703788001365596))
('Gr_i', 'Gd_i', np.float64(0.7202675272048841))
('sunheight', 'Gd_i', np.float64(0.844565354363884))
('hour_sin', 'Gd_i', np.float64(0.7826492388163777))
('sunheight', 'Gr_i', np.float64(0.8724625342404718))
('hour_sin', 'Gr_i', np.float64(0.8357073596178372))
('is_peak_hours', 'Gr_i', np.float64(0.7380479790981974))
('G_i', 'sunheight', np.float64(0.7302327084572351))
('Gr_i', 'sunheight', np.float64(0.8724625342404718))
('hour_sin', 'sunheight', np.float64(0.80897849213449))
'''

#Collinearity feature drops
features_to_drop_collinearity = ['Gb_i', 'Gd_i', 'Gr_i', 'is_peak_hours', 'hour_cos', 'hour_sin', 'day_of_year', 'doy_sin', 'doy_cos',
                                 'doy_cos', 'month_sin', 'month_cos'] #Dropping features that are highly correlated with G_i and also with each other, also dropping is_peak_hours since it's a binary feature that is highly correlated with G_i and sunheight, and we already have hour_sin and hour_cos to capture the cyclical nature of hours
X_train_scaled = X_train_scaled.drop(columns = features_to_drop_collinearity)
X_val_scaled = X_val_scaled.drop(columns = features_to_drop_collinearity)
X_test_scaled = X_test_scaled.drop(columns = features_to_drop_collinearity)

#Check if features were dropped
print(X_train_scaled.shape) #Should return 6070 x 14
print(X_val_scaled.shape) #Should return 1301 x 14
print(X_test_scaled.shape) #Should return 1301 x 14

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

