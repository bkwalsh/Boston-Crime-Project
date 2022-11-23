#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import math
import geopandas as gpd
import numpy as np
import folium
from folium.plugins import HeatMap
import matplotlib as plt
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import interpolate
get_ipython().run_line_magic('matplotlib', 'inline')

spreadsheet = pd.read_csv('/Users/bayardwalsh/Downloads/Boston_Crime_Data.csv', low_memory=False)


# In[2]:


# Dataframe with crimes listed by count
crimes_by_count = pd.DataFrame({'Count': spreadsheet.OFFENSE_CODE_GROUP.value_counts().sort_values(ascending = False)})
crimes_by_count['Index']=[i for i in range(1,crimes_by_count.size+1)]
crimes_by_count


# In[3]:


# Scatterplot with crime amount indicating size, distributed by index
graph = crimes_by_count.plot.scatter(x='Index', y='Count',s=crimes_by_count["Count"] * .01, c='Count',cmap="viridis")
median= math.floor(len(spreadsheet)/len(crimes_by_count)) #Average line
graph.axhline(y=median)


# In[4]:


#Heat map with sample size 10,000 crimes 
location_of_crimes = spreadsheet[spreadsheet.OFFENSE_CODE_GROUP.isin(crimes_by_count.index)].loc[:, ['Lat', 'Long']].dropna()

m=folium.Map(location = [42.320,-71.05], 
                  zoom_start = 11,
                  min_zoom = 11
)

HeatMap(data=location_of_crimes.sample(10000), radius=16).add_to(m)

m


# In[5]:


#List crimes with Greater than 10,000 recorded cases, or 'most common crimes', which is 15 total offense code
most_common_crimes = pd.DataFrame({'Count': spreadsheet.OFFENSE_CODE_GROUP.value_counts().sort_values(ascending = False).head(15)}) 
most_common_crimes


# In[6]:


#Heat Map for Overall Common Crimes, 10,000 samples 
location_common_crimes = spreadsheet[spreadsheet.OFFENSE_CODE_GROUP.isin(most_common_crimes.index)].loc[:, ['Lat', 'Long']].dropna()

m=folium.Map(location = [42.320,-71.05],
                  zoom_start = 11,
                  min_zoom = 11
)

HeatMap(data=location_common_crimes.sample(10000), radius=16).add_to(m)

m


# In[7]:


#Crimes By Year Histogram
years=[2015,2016,2017,2018,2019,2020]
crimes_per_year = pd.DataFrame({'Count': spreadsheet['YEAR'].value_counts().sort_index().values}, index = years)
plt.figure(figsize = (14, 8))
sns.barplot(x = crimes_per_year.index, y = 'Count', data = crimes_per_year, palette = 'rocket')
sns.set(font_scale=1.25)
plt.title('Crimes By Year in Boston', fontsize = 24)


# In[8]:


# Observation: significantly lower recorded info for 2015 and 2020
# Yearly and Monthly Graph Breakdown
for x in years:
    curr = spreadsheet[spreadsheet.YEAR == x]    
    curr=curr.groupby(["YEAR", "MONTH"]).size().reset_index(name="Counts")
    graph = curr.plot.bar(x='MONTH', y='Counts',colormap='summer', rot=0)
    plt.title('Boston Monthly Crime Reports in year '+str(x))
    plt.ylabel("Crimes Commited")
    plt.xlabel("Month")

# Indicates that 2015 and 2020 both don't have complete datasets 


# In[9]:


# Overall Monthly Crimes in Boston
# Keep in mind that 1-5 are unrecorded and 6 is paritally recorded for 2015, 5-12 are unrecorded 
#    and 4 is partially recorded for 2020. Because each month is missing roughly a year of data, the numbers won't
#    be incredibly skewed, but worth considering
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
crimes_per_month = pd.DataFrame({'Count': spreadsheet['MONTH'].value_counts().sort_index().values}, index = months)
plt.figure(figsize = (14, 8))
sns.barplot(x = crimes_per_month.index, y = 'Count', data = crimes_per_month, palette = 'magma')
plt.tick_params(labelsize = 12)
plt.title('Monthly Crimes in Boston', fontsize = 24)


# In[10]:


#District Histogram
# Note:
#few cases of external where long/lat is given
    # of these, some are in the city (under 10), however vast majority are outside city lines
    # for example, found crime at 42.328662, -71.085634 which wasn't recorded in a certain district
    # analysis will treat these as unrecorded data 
# 67 case of external wehre location is unrecorded -> will group external into unrecorded location

district=spreadsheet.copy()
Unrecorded = district["DISTRICT"].isna().sum()
district['Counts'] = district.groupby(['DISTRICT'])['DISTRICT'].transform('count')
district=district[['DISTRICT', 'Counts']].drop_duplicates(subset=['DISTRICT'])
Unrecorded+= district.iloc[13]['Counts'] #add 'external' into Not Listed set
district["Counts"].fillna(Unrecorded, inplace = True)
district=district.drop(labels=426891) #remove external edge case
district["DISTRICT"].fillna('NL', inplace = True)
district=district.sort_values(by=['Counts'])
district.plot.bar(x='DISTRICT', y='Counts', rot=0)
plt.title('Boston Crime Reports By District')
plt.ylabel("Crimes Commited")
plt.xlabel("District Code")


# In[11]:


# Bubble Map where size of bubble corresponds to amount of crimes committed
m = folium.Map(location=[42.3601,-71.0589], tiles="OpenStreetMap", zoom_start=12)

df = district.set_index('DISTRICT')
data = pd.DataFrame({
   'lon':[-71.0594, -71.0649, -71.0616, -71.0706, -71.1321, -71.0254, -71.1600, -71.091375,-71.1203,-71.0855,-71.049495, -71.1213],
   'lat':[42.3555, 42.2995, 42.3787, 42.3407, 42.3529, 42.3800, 42.2782, 42.315198, 42.3098,42.2700,42.333431, 42.2533],
   'name':['A1', 'C11', 'A15', 'D4', 'D14', 'A7', 'E5', 'B2', 'E13', 'B3', 'C6', 'E18'],
   'value':[df.at['A1', 'Counts'], df.at['C11', 'Counts'], df.at['A15', 'Counts'], df.at['D4', 'Counts'], df.at['D14', 'Counts'], 
            df.at['A7', 'Counts'], df.at['E5', 'Counts'], df.at['B2', 'Counts'],df.at['E13', 'Counts'],df.at['B3', 'Counts']
            ,df.at['C6', 'Counts'],df.at['E18', 'Counts']]
}, dtype=str)

for i in range(0,len(data)):
   folium.Circle(
      location=[data.iloc[i]['lat'], data.iloc[i]['lon']],
      popup=data.iloc[i]['name'],
      radius=float(data.iloc[i]['value'])*.025,
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(m)

m


# In[12]:


#Violent Crime Analysis
    #some cases in OFFENSE_CODE_GROUP where crimes is non-negative- Missing Person Located, Property Found
    # Note: no cases of sexual assault or rape
#Methodology
#In the FBI’s Uniform Crime Reporting (UCR) Program, violent crime is composed of four offenses: 
#murder and nonnegligent manslaughter, forcible rape, robbery, and aggravated assault. 
#Violent crimes are defined in the UCR Program as those offenses which involve force or threat of force.

#Robbery, Defined as:
#Larceny From Motor Vehicle
#Larceny 
#Auto Theft
#Residential Burglary
#Other Burglary
#Commercial Burglary
#Robbery
#Burglary - No Property Taken
#HOME INVASION


#Assault, Defined As:
#Simple Assault
#Aggravated Assault 


#Murder, Defined As:
#Homicide
#Manslaughter
murder=spreadsheet.copy()
assault=spreadsheet.copy()
robbery=spreadsheet.copy()
nonviolent=spreadsheet.copy()

murder = murder[murder['OFFENSE_CODE_GROUP'].isin(['Homicide','Manslaughter'])]
assault =assault[assault['OFFENSE_CODE_GROUP'].isin(['Aggravated Assault','Simple Assault'])]
robbery =robbery[robbery['OFFENSE_CODE_GROUP'].isin(['Larceny From Motor Vehicle','Larceny','Auto Theft',
                                                             'Residential Burglary','Other Burglary',
                                                             'Commercial Burglary','Robbery','Burglary - No Property Taken',
                                                             'HOME INVASION'])]
nonviolent=nonviolent[nonviolent['OFFENSE_CODE_GROUP'].isin(['Larceny From Motor Vehicle','Larceny','Auto Theft',
                                        'Residential Burglary','Other Burglary',
                                        'Commercial Burglary','Robbery','Burglary - No Property Taken',
                                        'HOME INVASION','Homicide','Manslaughter',
                                        'Aggravated Assault','Simple Assault'])==False]

murder['Violent Crime Type']='Murder'
robbery['Violent Crime Type']='Robbery'
assault['Violent Crime Type']='Assault'

frames = [murder, robbery, assault]

vc=pd.concat(frames)

vc


# In[13]:


# Percentage Breakdown within Violent Crimes 
# compares types of violent crimes in Boston by Percentage

# Function to get percentages for graph
def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{:.1f}%\n({v:d})'.format(pct, v=val)
        return my_format

vc_counts=vc['Violent Crime Type'].value_counts()
plt.pie(vc_counts,labels = vc_counts.index, autopct=autopct_format(vc_counts))
plt.title('Violent Crimes by Type, Boston', fontsize = 12)


# In[14]:


# Violent Crime Mapping
# Bubble map for Boston districts based on violent crime

vcmap = folium.Map(location=[42.3601,-71.0589], tiles="OpenStreetMap", zoom_start=12)

vcdistrict=vc.copy()
vcdistrict['Counts'] = vcdistrict.groupby(['DISTRICT'])['DISTRICT'].transform('count')
vcdistrict=vcdistrict[['DISTRICT', 'Counts']].drop_duplicates(subset=['DISTRICT'])

df = vcdistrict.set_index('DISTRICT')
data = pd.DataFrame({
   'lon':[-71.0594, -71.0649, -71.0616, -71.0706, -71.1321, -71.0254, -71.1600, -71.091375,-71.1203,-71.0855,-71.049495, -71.1213],
   'lat':[42.3555, 42.2995, 42.3787, 42.3407, 42.3529, 42.3800, 42.2782, 42.315198, 42.3098,42.2700,42.333431, 42.2533],
   'name':['A1', 'C11', 'A15', 'D4', 'D14', 'A7', 'E5', 'B2', 'E13', 'B3', 'C6', 'E18'],
   'value':[df.at['A1', 'Counts'], df.at['C11', 'Counts'], df.at['A15', 'Counts'], df.at['D4', 'Counts'], df.at['D14', 'Counts'], 
            df.at['A7', 'Counts'], df.at['E5', 'Counts'], df.at['B2', 'Counts'],df.at['E13', 'Counts'],df.at['B3', 'Counts']
            ,df.at['C6', 'Counts'],df.at['E18', 'Counts']]
}, dtype=str)

for i in range(0,len(data)):
   folium.Circle(
      location=[data.iloc[i]['lat'], data.iloc[i]['lon']],
      popup=data.iloc[i]['name'],
      radius=float(data.iloc[i]['value'])*.09, #Note, size of circle increased from previous bubble map with goal of showing proportion
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(vcmap)

vcmap


# In[15]:


#Comparing Violent To Non-Violent Crimes Percentages
frames = [vc,nonviolent]
violentcrimecomp=pd.concat(frames)
violentcrimecomp=violentcrimecomp.where(~violentcrimecomp.notna(), 'Violent Crimes')
violentcrimecomp['Violent Crime Type'].fillna('Non-Violent Crimes', inplace = True)

def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{:.1f}%\n({v:d})'.format(pct, v=val)
        return my_format
comp_counts=violentcrimecomp['Violent Crime Type'].value_counts(dropna=False)
plt.pie(comp_counts,labels = comp_counts.index, autopct=autopct_format(comp_counts))
plt.title('Non-Violent vs. Violent Crimes, Boston', fontsize = 12)


# In[16]:


# Boston crimes per hour histogram
per_hour = pd.DataFrame({'Count': spreadsheet['HOUR'].value_counts().sort_index()})
plt.figure(figsize = (12,10))
sns.barplot(x = per_hour.index, y = per_hour['Count'], data = per_hour, color = 'seagreen')
plt.ylabel("Crime Count",fontsize=16)
plt.xlabel("Time of Day, Hours",fontsize=16)
plt.tick_params(labelsize = 14)
plt.title('Boston Crimes per Hour', fontsize = 20)

