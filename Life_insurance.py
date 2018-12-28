# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 20:11:52 2018

@author: gmacharia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import sklearn
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


filepath='C:/Users/gmacharia/Grace/Personal files/Freelance/Analysis/I & M/Raw data/Obituaries_Dataset.csv'
obituaries=pd.read_csv(filepath)

##sort by name and check duplicates
obituaries = obituaries.sort_values(['Name'])
a=obituaries.loc[obituaries[['Name','Gender','Death_to_Announce']].duplicated()]


###check for missing data
obituaries[['Name', 'Announcement', 'Death', 'Burial', 'Burial_Day', 'Burial_Week',
       'Gender', 'Age', 'Color', 'Size', 'Word_Count', 'No_of_Children',
       'Significant_Children', 'Significant_Relatives', 'Fundraising',
       'Death_to_Announce', 'Death_to_Burial', 'Announce_to_Burial',
       'No_of_Relatives', 'County_Burial', 'County_Death', 'County_Morgue',
       'Distance_Death', 'Distance_Morgue', 'Cause_of_Death', 'Married',
       'Spouse_Alive', 'Spouse_gender', 'Hospital', 'Morgue', 'Same_Morgue',
       'Cost_Morgue', 'Occupation', 'Repetition', 'Corporate',
       'Corporate_Name', 'Residence', 'Residence_Name', 'Residence_Category']].count()


## Kaplan-meier survival curve using gender
kmf = KaplanMeierFitter() 

###create censored =1 since death has occured in all cases
obituaries['Censored']=1

##create a subset for non-null gender values
obituaries_age=obituaries.loc[(pd.notnull(obituaries['Age']))]

T = obituaries_age['Age'] #duration
C = obituaries_age['Censored'] #censorship - 1 if death/ is seen, 0 if censored

kmf.fit(T, event_observed=C)

#recode Gender
obituaries_age['Gender'] = obituaries_age.Gender.apply(lambda x: 1 if x == "Male" else 2) 


output={}
fig, ax= plt.subplots(figsize=[10,5])
for i, gender in enumerate(set(obituaries_age['Gender'])):
    ix=obituaries_age['Gender']==gender
    output[f' kmf {gender}']= kmf.fit(T.loc[ix], C.loc[ix], label=gender)
    ax=kmf.plot(ax=ax)

##create hazard ratio
male = (obituaries_age["Gender"] == 1)
output = logrank_test(T[male], T[~male], 
                       C[male], C[~male], alpha=0.99 )
output.print_summary()

Z = output.test_statistic
D = C.sum()

hazard_ratio = np.exp(Z*np.sqrt(4/D))
print(hazard_ratio)



###create binary variable for significant children and significant relative
obituaries['Significant_Children_bin']=1
obituaries['Significant_Relatives_bin']=1

obituaries.loc[obituaries['Significant_Children']==0, 'Significant_Children_bin' ]=0
obituaries.loc[obituaries['Significant_Relatives']==0, 'Significant_Relatives_bin' ]=0

###create list of variables to be used
obs_cols= ['Fundraising', 'Gender', 'Age', 'Married', 'Spouse_Alive', 'Spouse_gender','Significant_Relatives_bin', 'Significant_Children_bin','Significant_Children', 'Significant_Relatives']

##drop all null cases in the variables
obituaries = obituaries.dropna(subset=obs_cols)
obituaries = obituaries[obs_cols]
obituaries.info()

###drop significant children and significant relative 
obituaries.drop(['Significant_Children', 'Significant_Relatives'],axis=1,inplace=True)
obituaries.head()


 #recode Gender, Fundraising, Married and Spouse alive

for var in ['Gender', 'Spouse_gender',]:
    obituaries[var] = obituaries[var].apply(lambda x: 1 if x == 'Male' else 2)


for var in ['Fundraising', 'Spouse_Alive','Married']:
    obituaries[var] = obituaries[var].apply(lambda x: 1 if x== 'Yes' else 0)


X = obituaries.ix[:,(1,2,3,4,5,6,7)].values
Y = obituaries.ix[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .3, random_state=25)

LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)

y_pred = LogReg.predict(X_test)


###create confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix


print(classification_report(y_test, y_pred))


f1_score = sklearn.metrics.f1_score(y_test, y_pred)
tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print (f"F1 score = {f1_score}")
print (f"Sensitivity = {sensitivity}")
print (f"Specificity = {specificity}")


