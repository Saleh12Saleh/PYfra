# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -LanguageId
#     formats: ipynb,py:light
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
# ---

# Notebook 2
# ==============
# Data Cleaning and Feature Engineering

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pyfra
import seaborn as sns

# # Data import & pulling
#
# We have imported the data and separated them into four different categories: characteristics,places,users,vehicles.

french_categories = {'characteristics': 'caracteristiques', 'places':'lieux', 'users':'usagers', 'vehicles':'vehicules'}
data_categories = french_categories.keys()
categories_dict = dict(zip(data_categories, [0,0,0,0]))


# +
# Define the function that reads the raw data for the specified time range
def read_csv_of_year(start_year, end_year, separators, name_separator='_'):
    if len (separators)<4:
        separators = [separators]*4
        
    df_dict = {}
    for year in range(start_year,end_year+1):
        this_year_str = str(year)
        # Data Category
        this_df_dict = {}        
        for this_category, this_sep in zip(data_categories, separators):
            # We need the French name of the category for the filename
            this_french_category = french_categories[this_category]
            this_file_path_and_name = '../data/'+this_year_str+'/' + this_french_category+name_separator+this_year_str+'.csv'
            this_df_dict[this_category] = pd.read_csv(this_file_path_and_name, encoding='latin-1', sep=this_sep, low_memory=False)
        df_dict[year] = this_df_dict
    return df_dict

# Import years
df_dict = {}
df_dict.update(read_csv_of_year(2005, 2008, separators=','))
df_dict.update(read_csv_of_year(2009,2009, separators=['\t', ',', ',', ',']))
df_dict.update(read_csv_of_year(2010, 2016, separators=','))
df_dict.update(read_csv_of_year(2017, 2018, separators=',', name_separator='-'))
df_dict.update(read_csv_of_year(2019, 2021, separators=';', name_separator='-'))


# +
dict_of_category_dfs = {}
for this_category in data_categories:
    dict_of_category_dfs[this_category] = pd.concat([df_dict[year][this_category] for year in range(2005,2022)], ignore_index=True)

characteristics = dict_of_category_dfs['characteristics']
places = dict_of_category_dfs['places']
users = dict_of_category_dfs['users']
vehicles = dict_of_category_dfs['vehicles']


# -

# # Data Cleaning
# We will perform some of the cleaning of the data on the individual datasets. Not all cleaning is possible before merging the datasets, so there will be a second round of cleaning.

# ## Calculate the percentage of missing values for each dataframe

def na_percentage(df):
  return df.isna().sum() *100 / len(df)


for this_category, df in dict_of_category_dfs.items():
    print(this_category+'\n', na_percentage(df),'\n')

# ## Users Dataset

#Grav
users.grav.replace(to_replace=-1,value=1,inplace=True)

# +
#Place

users.place.value_counts()
users.place.fillna(1,inplace=True) #replace is with mode
users.place.replace(to_replace=-1,value=0,inplace=True) #-1 is unassigned , will put 0 unknown #Same result


# +
#Trajet

users.trajet.replace(to_replace=-1,value=0,inplace=True) #-1 is unassigned , will put 0 unknown #Same result
users.trajet.fillna(5,inplace=True) #replace is with mode

# +
#locp

users.locp.replace(to_replace=-1,value=0,inplace=True) #-1 is unassigned , will put 0 unknown #Same result
users.locp.value_counts()
users.locp.fillna(0,inplace=True) #replace is with mode

# +
#actp

users.actp.replace(to_replace=['B'],value=0,inplace=True)#-1,B is unassigned , will put 0 unknown #Same result
users.actp.replace(to_replace=' -1',value=0,inplace=True)
users.actp.replace(to_replace=['A'],value=8,inplace=True) #A is coming in/out of vehicule , will put 8 instead (int)
users.actp.fillna(0,inplace=True) #replace is with mode

users.actp.replace(to_replace='0',value=0,inplace=True)
users.actp.replace(to_replace='1',value=1,inplace=True)
users.actp.replace(to_replace='2',value=2,inplace=True)
users.actp.replace(to_replace='3',value=3,inplace=True)
users.actp.replace(to_replace='4',value=4,inplace=True)
users.actp.replace(to_replace='5',value=5,inplace=True)
users.actp.replace(to_replace='6',value=6,inplace=True)
users.actp.replace(to_replace='7',value=7,inplace=True)
users.actp.replace(to_replace='8',value=8,inplace=True)
users.actp.replace(to_replace='9',value=9,inplace=True)

# +
#etatp

users.etatp.replace(to_replace=-1,value=0,inplace=True) #-1, is unassigned , will put 0 unknown #Same result
users.etatp.isna().sum() #119291
users.etatp.value_counts() 
users.etatp.fillna(0,inplace=True) #replace is with mode

# +
#an_nais

users.an_nais.isna().sum() #10080
users.an_nais.value_counts() 
users.an_nais.fillna(1986.0,inplace=True)
#replace is with mode #the first 5 values of value_counts are close to each other

# -

#Sex 
users.sexe.replace(to_replace=-1,value=1,inplace=True)

# ### Fixing incoherency of 'secu' Variable
# Safety equipment until 2018 was in 2 variables: existence and use.
#
# From 2019, it is the use with up to 3 possible equipments for the same user
# (especially for motorcyclists whose helmet and gloves are mandatory).
#
# #secu1
# The character information indicates the presence and use of the safety equipment:
# -1 - No information
# 0 - No equipment
# 1 - Belt
# 2 - Helmet
# 3 - Children device
# 4 - Reflective vest
# 5 - Airbag (2WD/3WD)
# 6 - Gloves (2WD/3WD)
# 7 - Gloves + Airbag (2WD/3WD)
# 8 - Non-determinable
# 9 - Other
#
# #secu2
# The character information indicates the presence and use of the safety equipment
#
# #secu3
# The character information indicates the presence and use of safety equipment

#Security 
users.drop(columns='secu3',inplace=True)

#Searching for the exact index where the year 2019 starts to accuratly study secu factors
X = users.loc[users['Num_Acc'].astype(str).str.startswith("2019")].index[0]

# +
#secu has some missing values for older years , must fill with mode before continuing

users.secu[:X].fillna(value=1,inplace=True)

# +
users['SecuA'] = ((users.secu - users.secu%10)/10) #Type of security Used
users['SecuB']=   users.secu%10  # Was the security used or No

#SecuA is not very important we will focus on secuB if the item was used or not to facilitate study
# -

# 0 is unknown change to others 9 
users.SecuA.replace(to_replace=0,value=9,inplace=True)

# 2-No 3-UnDeterminable 0-Unknown , change all to 0 not used
users.SecuB.replace(to_replace=3,value=0,inplace=True)
users.SecuB.replace(to_replace=2,value=0,inplace=True)

# For Secu1-secu2 
# We will gather all usage for security variable with 1 value {1} , and if there is no safety we will use {0} , No need to take multiple security parameters (secu2) we will noly take into consideration 1 security variable for comparibility with earlier years.

users.secu1.replace(to_replace=[2,3,4,5,6,7,8,9],value=1,inplace=True)
users.secu1.replace(to_replace=[-1],value=0,inplace=True)

users.secu1.value_counts() 

users.SecuB.value_counts()

# Now both secu1 and secuB are same format we need to merge them into 1 column (for all years)

#must add iloc
users['Security']=0
users['Security'][:X] = users.SecuB[:X]
users['Security'][X+1:] = users.secu1[X+1:]

# +
#To drop unneeded columns
# #copy this to begining of df 3 before modeling for df
#users = users.drop(columns=['secu','secu1','secu2','SecuA','SecuB'])
# -

na_percentage(users)

# ### Translating the variable names from French to English

# +
users = users.rename(columns = {'catu' : 'User_category',
                                'grav' : 'Severity' , #Severity of accident
                                'sexe' : 'Sex' , #Sex of Driver
                                'trajet' : 'Trajectory' , 
                                'locp' : 'LOCP' , #localisation of pedestrian
                                'actp' : 'ACTP' , #action of pedestrian
                                'etatp' : 'StateP' , #State of pedestrian during accident
                                'an_nais' : 'YoB' , #Year of Birth
                               })
users.columns

#change type to int
users.place = users.place.astype(int)
users.Trajectory = users.Trajectory.astype(int)
users.LOCP = users.LOCP.astype(int)
users.ACTP = users.ACTP.astype(int)
users.StateP = users.StateP.astype(int)
users.YoB = users.YoB.astype(int)
# -

# ## Places Dataset

# ### Dropping unwanted columns , which are v1 , v2, vma, voie, env1

# +
# Droped 'Unnamed: 0','v1','v2','vma', because they contained no information.

places = places.drop(['v1','v2','vma','voie','env1'], axis = 1)
# -

# ### French to English Variables

# +
# Change french names against english names.

places = places.rename(columns = {'catr' : 'Rd_Cat', 'circ' : 'Traf_Direct' , 'nbv' : 'Lanes' ,
                           'pr' : 'Landmark' , 'pr1' : 'Dist_to_Landmark', 'vosp' : 'Add_Lanes', 'prof' : 'Rd_Prof' ,
                          'plan' : 'Rd_Plan' , 'lartpc' : 'Gre_Verge' , 'larrout' : 'Rd_Width', 'surf' : 'Rd_Cond',
                          'infra' : 'Envinmt' , 'situ' : 'Pos_Acc'})
places.head()
# -


# ### Changing Nans with Zeros

# +
# There is no value = 0 assigned to information in the places data set. 
# Zeros are used in the cleaned data set as a feature to identify original Nans
# and to keep the data set with as much information as possible.

places = places.fillna({'Rd_Cat':0, 'Traf_Direct': 0, 'Lanes':0, 'Add_Lanes':0, 'Rd_Prof':0,'Rd_Plan':0,
                        'Rd_Cond':0, 'Envinmt':0, 'Pos_Acc':0})
# -

# ### Changing needed "object" Variables to "int" Variables

# +
# Convert 'object' Variables to 'float' Variables

object_list = ['Landmark', 'Dist_to_Landmark', 'Gre_Verge', 'Rd_Width']

places[object_list] = places[object_list].apply(pd.to_numeric, errors='coerce', axis=1)

# Replace empty cells with 'Nans'

places.replace('', np.nan).copy()

# Fill 'Nans' with 0

places = places.fillna({'Landmark':0, 'Dist_to_Landmark': 0, 'Gre_Verge':0, 'Rd_Width':0})

# Convert 'float' Variables to 'int' Variables

float_list = ['Rd_Cat', 'Traf_Direct', 'Lanes', 'Landmark','Dist_to_Landmark', 'Add_Lanes', 'Rd_Prof', 'Rd_Plan',
              'Gre_Verge', 'Rd_Width', 'Rd_Cond', 'Envinmt','Pos_Acc']

places[float_list] = places[float_list].astype(int, errors = 'raise')

print(places.isna().sum())
print()
print(places.info())
print()
print(places.shape)
# -

# ## Characteristics Dataset

# ### Translating the variable names from French to English

# +
# Translation of the variable nacmes from French to English, also improving the names so that it becomes clearer, what they are about
characteristics.rename(columns={'an': 'year', 'mois':'month', 'jour': 'day', 'hrmn':'hhmm', 
                                'lum': 'daylight', 'agg': 'built-up_area', 'int':'intersection_category', 'atm': 'atmospheric_conditions',
                                'col': 'collision_category', 'com': 'municipality', 'adr':'adress', 'gps': 'gps_origin', 'lat': 'latitude',
                                'long': 'longitude', 'dep': 'department'}, inplace=True)

# Change the values for 'built-up_area' to make it more understandable, 1 means the accident happened in a built-up area and 0 means happened elsewhere. 
characteristics['built-up_area'].replace({1:0, 2:1}, inplace=True)
# -

# ### Fixing incoherent format of year variable

characteristics['year'].value_counts()

# The year format is inconsistent. Until 2018, the year was relative to the year 2000, e.g. "5" for 2005. This changed, however, in 2019 which was labeled as 2019.
# We will change the year format to YYYY.

characteristics['year'].replace({5:2005, 6:2006, 7:2007, 8:2008, 9:2009, 10:2010, 11:2011,
                                                         12:2012, 13:2013, 14:2014, 15:2015, 16:2016, 17:2017, 18:2018}, inplace=True)

# ### Fix inconsistent time format

# The time format inconsistent, sometimes it is hhmm, and sometimes hh:mm. We will therefore remove any ":" from the column 

#remove ':' from hhmm
characteristics['hhmm'] = characteristics['hhmm'].apply(lambda s: int(str(s).replace(':','')))


# ### Get weekday and weekend feature

characteristics['date'] = pd.to_datetime({'year':characteristics['year'],
                                                                 'month':dict_of_category_dfs['characteristics']['month'],
                                                                 'day':dict_of_category_dfs['characteristics']['day']})

# +
# New variable: weekday, integer from 0 to 6 representing the weekdays from monday to sunday.
characteristics['day_of_week'] = dict_of_category_dfs['characteristics']['date'].apply(lambda x: x.day_of_week)

# New binary variable: is_weekend, 0 for monday to friday and 1 for saturday and sunday
characteristics['is_weekend'] = (dict_of_category_dfs['characteristics']['day_of_week'] > 4).astype('int')


# -

# ### Remove trailing zeroes from Department variable
# The Department codes are followed by a zero for the years 2005--2018, which has no practical use for us. We will therefore eliminate these trailing zeroes.
# Also, since 2019 all the data is saved as strings. We will convert everything to strings, as this is nominal data, we will not make any calculations with it.

# +
def department_converter(dep):
    # Takes in a department code as int and returns a string
    # e.g. 750 will be '75' for Paris
    # and 201 will be '2B'
    if dep == 201:
        return '2A'
    elif dep == 202:
        return '2B'
    elif dep>970:
        return str(dep)
    else:
        return str(dep).rstrip('0')

characteristics.loc[(np.less(characteristics['year'],2019)),'department'] = \
    characteristics[(np.less(characteristics['year'],2019))]['department'].apply(department_converter)
# -

# ### Remove leading zeros from department code
# The dataset from 2021 contains leading zeroes for the department codes 1 to 9. These have to be replaced.

characteristics['department'] = characteristics['department'].apply(lambda code: code.lstrip('0'))

# ### Fill missing values in atmospheric conditions variable

characteristics['atmospheric_conditions'] = characteristics['atmospheric_conditions'].fillna(
    characteristics['atmospheric_conditions'].mode()[0])

# ### Fill missing values in collision category variable

characteristics['collision_category'] = characteristics['collision_category'].fillna(
    characteristics['collision_category'].mode()[0])

# ## Vehicles dataset

# ### Translating the variable names from French to English

vehicles = vehicles.rename(columns = {'id_vehicule' : 'id_veh' , 'num_veh' : 'num_veh' ,
                           'senc' : 'direction' , 'catv' : 'cat_veh', 'obs' : 'obstacle', 'obsm' : 'obstacle_movable' ,
                          'choc' : 'initial_point' , 'manv' : 'principal_maneuver' , 'motor' : 'motor_veh', 'occutc' : 'num_occupants'})
vehicles.columns

# ### Check of the variables with the most missing values

# Variable num_occupants is representing amount of passangers being victims of an accident when they used public transport system. Missing values are caused by not recording value 0 and keeping the cell empty. For this reason we decided to replace the missing values by 0.

vehicles["num_occupants"] = vehicles["num_occupants"].fillna(0)
vehicles['num_occupants'].isna().sum()

vehicles['num_occupants'].value_counts()

# The variable motor_veh represents the type of the motorisation of the vehicle. There are 85 % missing values in this column. Some of the values of this variable don't specificate an exact type but are tracked as unspecified, unknown, or other. We have decided to drop this variable as it doesn't have any significant influence on the target variable. 

vehicles = vehicles.drop(columns=['motor_veh'])

# 8 Variables have <= 1% missing information, so for those it should be fine to set the missing information just to zero.

vehicles[['Num_Acc', 'direction', 'cat_veh', 'obstacle', 'obstacle_movable', 'initial_point', 'principal_maneuver']] = vehicles[['Num_Acc', 'direction', 'cat_veh', 'obstacle', 'obstacle_movable', 'initial_point', 'principal_maneuver']].fillna(0)
vehicles.isna().sum()

# # Merge all datasets

# ## Ensure Correct Attribution of Users to Vehicles

users['id_vehicule'].fillna(users['num_veh'], inplace=True)
users.drop(columns=['num_veh'], inplace=True)
users.rename(columns={'id_vehicule': 'id_veh'}, inplace=True)
users.set_index(['Num_Acc', 'id_veh'], inplace=True)

# ## Left Join for further investigations
# We will continue working with the left join of the data, as the missing lines miss the most important variables anyway.

# +
df = users.merge(vehicles, how='left', left_index=True, right_on=['Num_Acc', 'id_veh']) \
     .merge(characteristics, how='left', on='Num_Acc') \
     .merge(places, how='left', on='Num_Acc')

print(na_percentage(df))
# -

df['Age'] = df['year'] - df['YoB']

del characteristics, places, vehicles, users, dict_of_category_dfs

# ## One-Hot Encoding of Categorical Features

df = pd.get_dummies(df.sample(frac=0.1, random_state=23),
       columns=['daylight', 'built-up_area', 'intersection_category', 
              'atmospheric_conditions', 'collision_category', 'department',
              'Rd_Cat', 'Traf_Direct', 'Add_Lanes', 'Rd_Prof', 'Rd_Plan', 
              'Rd_Cond', 'Envinmt', 'Pos_Acc', 'place', 
              'User_category', 'Sex', 'Trajectory', 'LOCP', 'ACTP', 
              'StateP', 'direction', 
              'cat_veh', 'obstacle', 'obstacle_movable', 'initial_point', 'principal_maneuver'])

# # Export DataFrame to Pickle 
# This step is necessary to be able to work with the data in another notebook.

df.to_pickle('../data/df.p')

# The pickle file is too big to track on github, we will therefore create a second file which contains the output of the describe-method as well as the number of nans for each column and the dtypes of the DataFrame.

df_check_info = pyfra.df_testing_info(df)
df_check_info.to_csv('../data/df_check_info.csv')
