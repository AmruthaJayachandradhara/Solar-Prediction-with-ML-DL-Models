#%%----------------------------------------------------Checking Radekhiv dataset from literature review-----------------------------------------------------------------$

#Dataset is from a paper by Shakovska et al. (2024) that we found in our literature review. 
#Dataset link : https://figshare.com/articles/dataset/Solar_power_station_data/26357059/1?file=47875348


#Check directory
import os 
print(os.getcwd())

#If the directory isn't wrong then change
os.chdir("C:\\Users\\shime\\Downloads")

#Load dataset
import pandas as pd
Radekhiv_df = pd.read_excel("radekhiv_to share (1).xlsx")
Radekhiv_df.head()
Radekhiv_df.tail()

#Check the columns in the dataset and the datatypes
print(Radekhiv_df.info()) #We observe 27 total columns and 13057 total rows
# INTEGERS: year, month, day, hour, solarradiation, uvindex, severrisk
# FLOATS: generation, temp, feelslike, dew, humidity, precip, precipprob, snow, snowdepth, windgust, windspeed, winddir, sealevelpressure, cloudcover, visibility, solarenergy are floats, 
# OBJECT: preciptype, conditions, icon, stations are all objects
#%%----------------------Explore and Check viability of dataset for our project--------------------------------------------------------------------------#

#Get a feel for dataset
print(Radekhiv_df.head()) #Looking at the dataset we see that the data is recorded at an hourly level, earliest date of observation is June 22, 2022
print(Radekhiv_df.tail()) #Last date of observation is Febrary 23, 2024. Good amount of data, at least a year to work with even if filtered
print(Radekhiv_df.shape) # 13057 by 27

#Check values of all columns in the dataset
print(Radekhiv_df.describe()) #Mostly full as every column but generation has exactly 13057 non-null values. Generation has 13054 total values

#Get column names
columns = Radekhiv_df.columns.tolist()
print(columns)

#Create loops to check viability of dataset to avoid any surprises
#Loop for variables that are not object datatypes
for column in columns:
    if Radekhiv_df[column].dtype != 'object': 
        print("Column name:", column)
        print("Data type:", Radekhiv_df[column].dtype)
        print("Number of unique values in column:", Radekhiv_df[column].nunique())
        print("Unique values in column:", Radekhiv_df[column].unique())
        print("Minimum value:", Radekhiv_df[column].min())
        print("Maximum value:", Radekhiv_df[column].max())
        print("----------------------------------------------------------------------------------------") #Nothing seems out the norm here except negative value as minimum for generation feature


#Replicate loop to check for columns with object datatypes
for column in columns:
    if Radekhiv_df[column].dtype == 'object':
        print("Column name:", column)
        print("Data type:", Radekhiv_df[column].dtype)
        print("Number of unique values in column:", Radekhiv_df[column].nunique())
        print("Unique values in column:", Radekhiv_df[column].unique())
        print("----------------------------------------------------------------------------------------")#Nothing  seems really out the norm except for nans in preciptype, maybe redundant values in conditionsfeature, icon seems too similar to conditions too, stations will be dropped
#Check for missing values in the dataset
print(Radekhiv_df.isnull().sum()) #Preciptype has 10375 missing values, generation has 3 missing values, stations has 569 missing values
print(Radekhiv_df.isna().sum()) # Same as before

#Check for negatives in dataset specifically generation column
print((Radekhiv_df['generation'] < 0).sum()) #It appears that Radekhiv has negative values 1091 times in the dataset

#%%-------------------------------------------------------Cleaning the file------------------------------------------------------------------------------------
#Make a copy of the dataset that we'll move forward with in manipulating and cleaning 
Radekhiv_df_copy = Radekhiv_df.copy()

#There isn't a need for seprate columns for all date related variables so let's create a timestamp 
Radekhiv_df_copy['Datetime'] = pd.to_datetime(Radekhiv_df_copy[['year', 'month', 'day', 'hour']])
print(Radekhiv_df_copy['Datetime'].dtype)
print(Radekhiv_df_copy['Datetime'].head())

#Drop previous columns that were combined into datetime column
new_columns_to_drop = ['year', 'month', 'day', 'hour']
Radekhiv_df_copy = Radekhiv_df_copy.drop(columns = new_columns_to_drop)

#Final check of columns in dataset
print(Radekhiv_df_copy.columns.tolist())
print(Radekhiv_df_copy.info()) #We have 24 columns now and 13057 rows, datetime is a datetime format, generation has 13054 non-null values, preciptype has 10375 non-null values, stations has 12488 non-null values

#Address missingness in the dataset
'''Remember in previous code we found a number of null values in preciptype and stations with just 3 in generation variable'''
columns_to_drop_from_missingness = ['preciptype', 'stations']

#Drop columns from missingness
Radekhiv_df_copy = Radekhiv_df_copy.drop(columns = columns_to_drop_from_missingness)

#Check dataset again
print(Radekhiv_df_copy.info()) #22 columns still 13056 observations

#Address negative values in generation variable
'''Remember it was found that generation had 1091 negative values, which doesn't seem to be possible because how can anything generate negative power or less than 0 power?
For this reason, it is important that we inspect the negative values further before doing something'''

#Check the negative values in generation variable
negative_values_generated = Radekhiv_df_copy[Radekhiv_df_copy['generation'] < 0]
print(negative_values_generated)#From this we can see the negative generation values are when solar radiation, solar energy, and uvindex are 0, icon also says 'night' quite often
print(negative_values_generated['generation'].unique()) #From this we can see that all 1091 negative values are either -1 or -2

'''We assume from knowledge of PV systems that these negative values that are very close to zero are likely some kind of sensor error which can be assumed to be 0 power generation or output'''

#Replace negative values with 0
Radekhiv_df_copy['generation'] = Radekhiv_df_copy['generation'].apply(lambda x: 0 if x < 0 else x)

#Check if negative values were replaced
print((Radekhiv_df_copy['generation'] < 0).sum()) #Output is 0 so we're succsesful in replacing negative values with 0

'''There are still 3 observations of missingness in the generation variable. We will decide what to do with it below'''
#Check when missing values in generation are
missing_generation = Radekhiv_df_copy[Radekhiv_df_copy['generation'].isnull()]
print(missing_generation) # We observe that the dates of the missing features are 2023/05/04, 2023/05/08, and 2023/06/10

#Recall start and end of dataset
print(Radekhiv_df_copy['Datetime'].min()) #2022-06-22
print(Radekhiv_df_copy['Datetime'].max()) #2024-02-23
#so since missing generation dates fall in between any full year of observation, they cannot be simply ignored

#Drop the 3 missing values
Radekhiv_df_copy = Radekhiv_df_copy.drop(missing_generation.index)

#Check dataset
print(Radekhiv_df_copy['generation'].isnull().sum())
print(Radekhiv_df_copy.shape) # We see the three have been dropped as the observation countis now 13054 instead of 13057

#%%---------------------------------------------------------------------Early Feature Engineering--------------------------------------------------------------------------------------------#
#Add a season feature for better visuals
'''The PV of the dataset is located in Western Ukraine, near the border of Ukraine and Poland. We know the seasons in Ukraine are as follows'''

Radekhiv_df_copy['season'] = None

Radekhiv_df_copy['season'] = Radekhiv_df_copy['season'].fillna(Radekhiv_df_copy['Datetime'].dt.month.isin([12, 1, 2]).map({True: 'Winter', False: None}))
Radekhiv_df_copy['season'] = Radekhiv_df_copy['season'].fillna(Radekhiv_df_copy['Datetime'].dt.month.isin([3,4,5]).map({True: 'Spring', False: None}))
Radekhiv_df_copy['season'] = Radekhiv_df_copy['season'].fillna(Radekhiv_df_copy['Datetime'].dt.month.isin([6,7,8]).map({True: 'Summer', False: None}))
Radekhiv_df_copy['season'] = Radekhiv_df_copy['season'].fillna(Radekhiv_df_copy['Datetime'].dt.month.isin([9,10,11]).map({True: 'Fall', False: None}))

#Check if this was successful
print(Radekhiv_df_copy.head())
print(Radekhiv_df_copy.tail())
print(Radekhiv_df_copy['season'].unique()) #We see that the season feature was successfully added to the dataset

#Engineer a feature for sun height 
import pvlib
from pvlib import solarposition

#We already have time stamp in dataset, and the coordinates of Radekhiv are 50.2797° N, 24.6369° E. We can use this information to calculate sun height

#Using the documentation of pvlib we can get sunheight from computing solar position 
solar_position_df = solarposition.get_solarposition(Radekhiv_df_copy['Datetime'], latitude=50.2797, longitude=24.6369, altitude= 231, method = 'nrel_numpy', pressure = Radekhiv_df_copy['sealevelpressure'], temperature = Radekhiv_df_copy['temp'])
print(solar_position_df.head())
print(solar_position_df['apparent_elevation'].unique())
#We take the column from solar position that represents sun height and add it to the dataset using datetime as the index
Radekhiv_df_copy = Radekhiv_df_copy.set_index('Datetime')
Radekhiv_df_copy['sunheight'] = solar_position_df['apparent_elevation'] 

#Check if it worked well
print(Radekhiv_df_copy.head())
print(Radekhiv_df_copy.tail())
print(Radekhiv_df_copy['sunheight'].unique())


#%%-------------------------------------------------------------------Final checks to make sure dataset isn't corrupt or unusable for the project-------------------------------------------------------------------#
#Reset index since we made datetime index in previous chunk
Radekhiv_df_copy = Radekhiv_df_copy.reset_index()

#Make sure period of time in the dataset doesn't have massive gaps or holes
print(Radekhiv_df_copy['Datetime'].diff(periods=1).value_counts()) #Mostly 1 hour or 2 hour gaps, but there is one large 68 day gap
#The 68 day gap is from 11/2023 to 1/30/2024, one way around this is to just have the period of observation not include this gap

#Check for unique datetimes and compare to number of rows in dataset
print(Radekhiv_df_copy['Datetime'].nunique()) #Output is 13052 
print(len(Radekhiv_df_copy))#Output = 13054 so there are 2 duplicates specifically for timestamp

#Check timestamp duplicates and remove duplicates
duplicates = Radekhiv_df_copy[Radekhiv_df_copy.duplicated(subset=['Datetime'], keep=False)]
print(duplicates) #We see that the duplicates are for 10-30-2022 and 10-29-2023. 

'''The duplicate time stamps come from rows 3121, 3122, 11857 and 11858'''
#Before dropping or doing anything, I checked the excel sheet and the duplicates are not a typo of the hour mark, so here I will address it
#Duplicate is likely due to daylight savings, I tried writing code to address without dropping but was not successful, so I will drop duplicates for now
Radekhiv_df_copy = Radekhiv_df_copy.drop_duplicates(subset=['Datetime'], keep='first')

#Check if resolved
print(Radekhiv_df_copy['Datetime'].nunique())#13052 again
print(len(Radekhiv_df_copy))#13052 so we're successful

#Final missingness check
print(Radekhiv_df_copy.isnull().sum()) #No missingness at all

#Checks of values
print(Radekhiv_df_copy.describe())#No glaring or unexplainable red flags in the data, values are in expected ranges and averages too


#%%-----------------------------------------------------------------Save cleaned dataset for EDA and modeling .py files--------------------------------------------------------------------------------#
Radekhiv_df_copy.to_csv("Radekhiv_Cleaned.csv", index = False)

# %%---------------------------------------------------------------Make additional copy and this time remove night time hours just in case it is needed for modeling--------------------------------------------------------------#

#Make another copy
Radekhiv_df_copy_2 = Radekhiv_df_copy.copy()

# Remove night time hours by observing solar radiation, solar energy, icon
Radekhiv_df_copy_2 = Radekhiv_df_copy_2[Radekhiv_df_copy_2['solarradiation'] > 0] #Remove night time hours by looking for instances where solar radiation is greater than 0

#Check unique icon values for new dataset without night time hours and compare with original with night time hours
print(Radekhiv_df_copy['icon'].unique()) #
print(Radekhiv_df_copy_2['icon'].unique()) # We should see that night isn't in the values anymore

#Check value counts too We should at least see a drop in night time hours
print(Radekhiv_df_copy['icon'].value_counts())
print(Radekhiv_df_copy_2['icon'].value_counts()) #We see a significant drop in the night related or adjacent icons

#Check lengths
print(len(Radekhiv_df_copy)) #13052 
print(len(Radekhiv_df_copy_2)) # 6972 so we removed a lot of hours

#Export this dataset as well for future use
Radekhiv_df_copy_2.to_csv("Radekhiv_Cleaned_Without_Nighttime.csv", index = False)
# %%-------------------------------------------------------Make Correlation matrix for dataset----------------------------------------------------------------------------------#
#For this we will use the dataset with the night time hours present

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Convert categorical variables to numeric for correlation matrix
categorical_columns = Radekhiv_df_copy.select_dtypes(include=['object']).columns
for column in categorical_columns:  
    Radekhiv_df_copy[column] = Radekhiv_df_copy[column].astype('category').cat.codes


numerical_columns = ['generation', 'temp', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob', 'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility', 'solarenergy', 'solarradiation', 'uvindex', 'severerisk']

#Create dataset with just numerical columns for correlation matrix
Radekhiv_df_copy_numerical = Radekhiv_df_copy[numerical_columns]

Radekhiv_df_copy_corr = Radekhiv_df_copy[numerical_columns].corr()
plt.figure(figsize = (20, 20))
sns.heatmap(Radekhiv_df_copy_corr, annot = True, cmap = 'RdBu', square = True)

print(Radekhiv_df_copy_corr) 

# %%
