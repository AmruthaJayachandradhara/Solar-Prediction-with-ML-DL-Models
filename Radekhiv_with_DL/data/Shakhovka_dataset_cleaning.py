#%%----------------------------------------------------Checking Shakhovska dataset from literature review-----------------------------------------------------------------$

#Dataset is from a paper by Shakovska et al. (2024) that we found in our literature review. 
#Dataset link : https://figshare.com/articles/dataset/Solar_power_station_data/26357059/1?file=47875348


#Check directory
import os 
print(os.getcwd())

#If the directory isn't wrong then change
#os.chdir("C:\\Users\\shime\\Downloads")

#Load dataset
import pandas as pd
Shakhovska_df = pd.read_excel("radekhiv_to share.xlsx")
Shakhovska_df.head()
Shakhovska_df.tail()

#Check the columns in the dataset and the datatypes
print(Shakhovska_df.info()) #We observe 27 total columns and 13057 total rows
# INTEGERS: year, month, day, hour, solarradiation, uvindex, severrisk
# FLOATS: generation, temp, feelslike, dew, humidity, precip, precipprob, snow, snowdepth, windgust, windspeed, winddir, sealevelpressure, cloudcover, visibility, solarenergy are floats, 
# OBJECT: preciptype, conditions, icon, stations are all objects
#%%----------------------Explore and Check viability of dataset for our project--------------------------------------------------------------------------#

#Get a feel for dataset
print(Shakhovska_df.head()) #Looking at the dataset we see that the data is recorded at an hourly level, earliest date of observation is June 22, 2022
print(Shakhovska_df.tail()) #Last date of observation is Febrary 23, 2024. Good amount of data, at least a year to work with even if filtered
print(Shakhovska_df.shape) # 13057 by 27

#Check values of all columns in the dataset
print(Shakhovska_df.describe()) #Mostly full as every column but generation has exactly 13057 non-null values. Generation has 13054 total values

#Get column names
columns = Shakhovska_df.columns.tolist()
print(columns)

#Create loops to check viability of dataset to avoid any surprises
#Loop for variables that are not object datatypes
for column in columns:
    if Shakhovska_df[column].dtype != 'object': 
        print("Column name:", column)
        print("Data type:", Shakhovska_df[column].dtype)
        print("Number of unique values in column:", Shakhovska_df[column].nunique())
        print("Unique values in column:", Shakhovska_df[column].unique())
        print("Minimum value:", Shakhovska_df[column].min())
        print("Maximum value:", Shakhovska_df[column].max())
    print("----------------------------------------------------------------------------------------") #Nothing seems out the norm here except negative value as minimum for generation feature


#Replicate loop to check for columns with object datatypes
for column in columns:
    if Shakhovska_df[column].dtype == 'object':
        print("Column name:", column)
        print("Data type:", Shakhovska_df[column].dtype)
        print("Number of unique values in column:", Shakhovska_df[column].nunique())
        print("Unique values in column:", Shakhovska_df[column].unique())
        print("----------------------------------------------------------------------------------------")#Nothing  seems really out the norm except for nans in preciptype, maybe redundant values in conditionsfeature, icon seems too similar to conditions too, stations will be dropped
#Check for missing values in the dataset
print(Shakhovska_df.isnull().sum()) #Preciptype has 10375 missing values, generation has 3 missing values, stations has 569 missing values
print(Shakhovska_df.isna().sum()) # Same as before

#Check for negatives in dataset specifically generation column
print((Shakhovska_df['generation'] < 0).sum()) #It appears that Shakhovska has negative values 1091 times in the dataset

#%%-------------------------------------------------------Cleaning the file------------------------------------------------------------------------------------
#Make a copy of the dataset that we'll move forward with in manipulating and cleaning 
Shakhovska_df_copy = Shakhovska_df.copy()

#There isn't a need for seprate columns for all date related variables so let's create a timestamp 
Shakhovska_df_copy['Datetime'] = pd.to_datetime(Shakhovska_df_copy[['year', 'month', 'day', 'hour']])
print(Shakhovska_df_copy['Datetime'].dtype)
print(Shakhovska_df_copy['Datetime'].head())

#Drop previous columns that were combined into datetime column
new_columns_to_drop = ['year', 'month', 'day', 'hour']
Shakhovska_df_copy = Shakhovska_df_copy.drop(columns = new_columns_to_drop)

#Final check of columns in dataset
print(Shakhovska_df_copy.columns.tolist())
print(Shakhovska_df_copy.info()) #We have 24 columns now and 13057 rows, datetime is a datetime format, generation has 13054 non-null values, preciptype has 10375 non-null values, stations has 12488 non-null values

#Address missingness in the dataset
'''Remember in previous code we found a number of null values in preciptype and stations with just 3 in generation variable'''
columns_to_drop_from_missingness = ['preciptype', 'stations']

#Drop columns from missingness
Shakhovska_df_copy = Shakhovska_df_copy.drop(columns = columns_to_drop_from_missingness)

#Check dataset again
print(Shakhovska_df_copy.info()) #22 columns still 13056 observations

#Address negative values in generation variable
'''Remember it was found that generation had 1091 negative values, which doesn't seem to be possible because how can anything generate negative power or less than 0 power?
For this reason, it is important that we inspect the negative values further before doing something'''

#Check the negative values in generation variable
negative_values_generated = Shakhovska_df_copy[Shakhovska_df_copy['generation'] < 0]
print(negative_values_generated)#From this we can see the negative generation values are when solar radiation, solar energy, and uvindex are 0, icon also says 'night' quite often
print(negative_values_generated['generation'].unique()) #From this we can see that all 1091 negative values are either -1 or -2

'''We assume from knowledge of PV systems that these negative values that are very close to zero are likely some kind of sensor error which can be assumed to be 0 power generation or output'''

#Replace negative values with 0
Shakhovska_df_copy['generation'] = Shakhovska_df_copy['generation'].apply(lambda x: 0 if x < 0 else x)

#Check if negative values were replaced
print((Shakhovska_df_copy['generation'] < 0).sum()) #Output is 0 so we're succsesful in replacing negative values with 0

'''There are still 3 observations of missingness in the generation variable. We will decide what to do with it below'''
#Check when missing values in generation are
missing_generation = Shakhovska_df_copy[Shakhovska_df_copy['generation'].isnull()]
print(missing_generation) # We observe that the dates of the missing features are 2023/05/04, 2023/05/08, and 2023/06/10 

#Recall start and end of dataset
print(Shakhovska_df_copy['Datetime'].min()) #2022-06-22
print(Shakhovska_df_copy['Datetime'].max()) #2024-02-23
#so since missing generation dates fall in between any full year of observation, they cannot be simply ignored

#Drop the 3 missing values
Shakhovska_df_copy = Shakhovska_df_copy.drop(missing_generation.index)

#Check dataset
print(Shakhovska_df_copy['generation'].isnull().sum())
print(Shakhovska_df_copy.shape) # We see the three have been dropped as the observation countis now 13054 instead of 13057


#%%-------------------------------------------------------------------Final checks to make sure dataset isn't corrupt or unusable for the project-------------------------------------------------------------------#
#Make sure period of time in the dataset doesn't have massive gaps or holes
print(Shakhovska_df_copy['Datetime'].diff(periods=1).value_counts()) #Mostly 1 hour or 2 hour gaps, but there is one large 68 day gap
#The 68 day gap is from 11/2023 to 1/30/2024, one way around this is to just have the period of observation not include this gap


#Check for unique datetimes and compare to number of rows in dataset
print(Shakhovska_df_copy['Datetime'].nunique()) #Output is 13052 
print(len(Shakhovska_df_copy))#Output = 13054 so there are 2 duplicates specifically for timestamp

#Check timestamp duplicates and remove duplicates
duplicates = Shakhovska_df_copy[Shakhovska_df_copy.duplicated(subset=['Datetime'], keep=False)]
print(duplicates) #We see that the duplicates are for 10-30-2022 and 10-29-2023. 

'''The duplicate time stamps come from rows 3121, 3122, 11857 and 11858'''
#Before dropping or doing anything, I checked the excel sheet and the duplicates are not a typo of the hour mark, so here I will address it
#Duplicate is likely due to daylight savings, I tried writing code to address without dropping but was not successful, so I will drop duplicates for now
Shakhovska_df_copy = Shakhovska_df_copy.drop_duplicates(subset=['Datetime'], keep='first')

#Check if resolved
print(Shakhovska_df_copy['Datetime'].nunique())#13052 again
print(len(Shakhovska_df_copy))#13052 so we're successful

#Final missingness check
print(Shakhovska_df_copy.isnull().sum()) #No missingness at all

#Checks of values
print(Shakhovska_df_copy.describe())#No glaring or unexplainable red flags in the data, values are in expected ranges and averages too


#%%-----------------------------------------------------------------Save cleaned dataset for EDA and modeling .py files--------------------------------------------------------------------------------#
Shakhovska_df_copy.to_csv("Shakhovska_Cleaned.csv", index = False)
