"""
Complete Football Match Prediction System
All code from notebook in a single file
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime as dt
import itertools
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from pandas.plotting import scatter_matrix

# ============================================================================
# DATA LOADING
# ============================================================================

folder = '../data/Datasets/'

# Read all season data
raw_data_1 = pd.read_csv(folder + '2000-01.csv')
raw_data_2 = pd.read_csv(folder + '2001-02.csv')
raw_data_3 = pd.read_csv(folder + '2002-03.csv')
raw_data_4 = pd.read_csv(folder + '2003-04.csv')
raw_data_5 = pd.read_csv(folder + '2004-05.csv')
raw_data_6 = pd.read_csv(folder + '2005-06.csv')
raw_data_7 = pd.read_csv(folder + '2006-07.csv')
raw_data_8 = pd.read_csv(folder + '2007-08.csv')
raw_data_9 = pd.read_csv(folder + '2008-09.csv')
raw_data_10 = pd.read_csv(folder + '2009-10.csv')
raw_data_11 = pd.read_csv(folder + '2010-11.csv')
raw_data_12 = pd.read_csv(folder + '2011-12.csv')
raw_data_13 = pd.read_csv(folder + '2012-13.csv')
raw_data_14 = pd.read_csv(folder + '2013-14.csv')
raw_data_15 = pd.read_csv(folder + '2014-15.csv')
raw_data_16 = pd.read_csv(folder + '2015-16.csv')
raw_data_17 = pd.read_csv(folder + '2016-17.csv')
raw_data_18 = pd.read_csv(folder + '2017-18.csv')

# Extract required columns
columns_req = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']

playing_statistics_1 = raw_data_1[columns_req]
playing_statistics_2 = raw_data_2[columns_req]
playing_statistics_3 = raw_data_3[columns_req]
playing_statistics_4 = raw_data_4[columns_req]
playing_statistics_5 = raw_data_5[columns_req]
playing_statistics_6 = raw_data_6[columns_req]
playing_statistics_7 = raw_data_7[columns_req]
playing_statistics_8 = raw_data_8[columns_req]
playing_statistics_9 = raw_data_9[columns_req]
playing_statistics_10 = raw_data_10[columns_req]
playing_statistics_11 = raw_data_11[columns_req]
playing_statistics_12 = raw_data_12[columns_req]
playing_statistics_13 = raw_data_13[columns_req]
playing_statistics_14 = raw_data_14[columns_req]
playing_statistics_15 = raw_data_15[columns_req]
playing_statistics_16 = raw_data_16[columns_req]
playing_statistics_17 = raw_data_17[columns_req]
playing_statistics_18 = raw_data_18[columns_req]

# ============================================================================
# GOALS SCORED AND CONCEDED FUNCTIONS
# ============================================================================

def get_goals_scored(playing_stat):
    """Gets the goals scored aggregated by teams and matchweek"""
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean(numeric_only=True).T.columns:
        teams[i] = []
    
    for i in range(len(playing_stat)):
        HTGS = playing_stat.iloc[i]['FTHG']
        ATGS = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGS)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS)
    
    GoalsScored = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T
    GoalsScored[0] = 0
    for i in range(2, 39):
        GoalsScored[i] = GoalsScored[i] + GoalsScored[i-1]
    return GoalsScored


def get_goals_conceded(playing_stat):
    """Gets the goals conceded aggregated by teams and matchweek"""
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean(numeric_only=True).T.columns:
        teams[i] = []
    
    for i in range(len(playing_stat)):
        ATGC = playing_stat.iloc[i]['FTHG']
        HTGC = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGC)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGC)
    
    GoalsConceded = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T
    GoalsConceded[0] = 0
    for i in range(2, 39):
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i-1]
    return GoalsConceded


def get_gss(playing_stat):
    """Add goals scored and conceded statistics"""
    GC = get_goals_conceded(playing_stat)
    GS = get_goals_scored(playing_stat)
   
    j = 0
    HTGS = []
    ATGS = []
    HTGC = []
    ATGC = []

    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTGS.append(GS.loc[ht][j])
        ATGS.append(GS.loc[at][j])
        HTGC.append(GC.loc[ht][j])
        ATGC.append(GC.loc[at][j])
        
        if ((i + 1) % 10) == 0:
            j = j + 1
        
    playing_stat['HTGS'] = HTGS
    playing_stat['ATGS'] = ATGS
    playing_stat['HTGC'] = HTGC
    playing_stat['ATGC'] = ATGC
    
    return playing_stat

# Apply to all datasets
playing_statistics_1 = get_gss(playing_statistics_1)
playing_statistics_2 = get_gss(playing_statistics_2)
playing_statistics_3 = get_gss(playing_statistics_3)
playing_statistics_4 = get_gss(playing_statistics_4)
playing_statistics_5 = get_gss(playing_statistics_5)
playing_statistics_6 = get_gss(playing_statistics_6)
playing_statistics_7 = get_gss(playing_statistics_7)
playing_statistics_8 = get_gss(playing_statistics_8)
playing_statistics_9 = get_gss(playing_statistics_9)
playing_statistics_10 = get_gss(playing_statistics_10)
playing_statistics_11 = get_gss(playing_statistics_11)
playing_statistics_12 = get_gss(playing_statistics_12)
playing_statistics_13 = get_gss(playing_statistics_13)
playing_statistics_14 = get_gss(playing_statistics_14)
playing_statistics_15 = get_gss(playing_statistics_15)
playing_statistics_16 = get_gss(playing_statistics_16)
playing_statistics_17 = get_gss(playing_statistics_17)
playing_statistics_18 = get_gss(playing_statistics_18)

# ============================================================================
# POINTS CALCULATION FUNCTIONS
# ============================================================================

def get_points(result):
    """Convert match result to points"""
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0


def get_cuml_points(matchres):
    """Get cumulative points"""
    matchres_points = matchres.map(get_points)
    for i in range(2, 39):
        matchres_points[i] = matchres_points[i] + matchres_points[i-1]
    matchres_points.insert(column=0, loc=0, value=[0*i for i in range(20)])
    return matchres_points


def get_matchres(playing_stat):
    """Get match results for each team"""
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean(numeric_only=True).T.columns:
        teams[i] = []

    for i in range(len(playing_stat)):
        if playing_stat.iloc[i].FTR == 'H':
            teams[playing_stat.iloc[i].HomeTeam].append('W')
            teams[playing_stat.iloc[i].AwayTeam].append('L')
        elif playing_stat.iloc[i].FTR == 'A':
            teams[playing_stat.iloc[i].AwayTeam].append('W')
            teams[playing_stat.iloc[i].HomeTeam].append('L')
        else:
            teams[playing_stat.iloc[i].AwayTeam].append('D')
            teams[playing_stat.iloc[i].HomeTeam].append('D')
            
    return pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T


def get_agg_points(playing_stat):
    """Get aggregated points for teams"""
    matchres = get_matchres(playing_stat)
    cum_pts = get_cuml_points(matchres)
    HTP = []
    ATP = []
    j = 0
    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTP.append(cum_pts.loc[ht][j])
        ATP.append(cum_pts.loc[at][j])

        if ((i + 1) % 10) == 0:
            j = j + 1
            
    playing_stat['HTP'] = HTP
    playing_stat['ATP'] = ATP
    return playing_stat

# Apply to all datasets
playing_statistics_1 = get_agg_points(playing_statistics_1)
playing_statistics_2 = get_agg_points(playing_statistics_2)
playing_statistics_3 = get_agg_points(playing_statistics_3)
playing_statistics_4 = get_agg_points(playing_statistics_4)
playing_statistics_5 = get_agg_points(playing_statistics_5)
playing_statistics_6 = get_agg_points(playing_statistics_6)
playing_statistics_7 = get_agg_points(playing_statistics_7)
playing_statistics_8 = get_agg_points(playing_statistics_8)
playing_statistics_9 = get_agg_points(playing_statistics_9)
playing_statistics_10 = get_agg_points(playing_statistics_10)
playing_statistics_11 = get_agg_points(playing_statistics_11)
playing_statistics_12 = get_agg_points(playing_statistics_12)
playing_statistics_13 = get_agg_points(playing_statistics_13)
playing_statistics_14 = get_agg_points(playing_statistics_14)
playing_statistics_15 = get_agg_points(playing_statistics_15)
playing_statistics_16 = get_agg_points(playing_statistics_16)
playing_statistics_17 = get_agg_points(playing_statistics_17)
playing_statistics_18 = get_agg_points(playing_statistics_18)

# ============================================================================
# TEAM FORM FUNCTIONS
# ============================================================================

def get_form(playing_stat, num):
    """Get form for last num matches"""
    form = get_matchres(playing_stat)
    form_final = form.copy()
    for i in range(num, 39):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i-j]
            j += 1
    return form_final


def add_form(playing_stat, num):
    """Add form columns to dataset"""
    form = get_form(playing_stat, num)
    h = ['M' for i in range(num * 10)]
    a = ['M' for i in range(num * 10)]
    
    j = num
    for i in range((num*10), 380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        
        past = form.loc[ht][j]
        h.append(past[num-1])
        
        past = form.loc[at][j]
        a.append(past[num-1])
        
        if ((i + 1) % 10) == 0:
            j = j + 1

    playing_stat['HM' + str(num)] = h
    playing_stat['AM' + str(num)] = a
    return playing_stat


def add_form_df(playing_statistics):
    """Add all form columns"""
    playing_statistics = add_form(playing_statistics, 1)
    playing_statistics = add_form(playing_statistics, 2)
    playing_statistics = add_form(playing_statistics, 3)
    playing_statistics = add_form(playing_statistics, 4)
    playing_statistics = add_form(playing_statistics, 5)
    return playing_statistics

# Apply to all datasets
playing_statistics_1 = add_form_df(playing_statistics_1)
playing_statistics_2 = add_form_df(playing_statistics_2)
playing_statistics_3 = add_form_df(playing_statistics_3)
playing_statistics_4 = add_form_df(playing_statistics_4)
playing_statistics_5 = add_form_df(playing_statistics_5)
playing_statistics_6 = add_form_df(playing_statistics_6)
playing_statistics_7 = add_form_df(playing_statistics_7)
playing_statistics_8 = add_form_df(playing_statistics_8)
playing_statistics_9 = add_form_df(playing_statistics_9)
playing_statistics_10 = add_form_df(playing_statistics_10)
playing_statistics_11 = add_form_df(playing_statistics_11)
playing_statistics_12 = add_form_df(playing_statistics_12)
playing_statistics_13 = add_form_df(playing_statistics_13)
playing_statistics_14 = add_form_df(playing_statistics_14)
playing_statistics_15 = add_form_df(playing_statistics_15)
playing_statistics_16 = add_form_df(playing_statistics_16)
playing_statistics_17 = add_form_df(playing_statistics_17)
playing_statistics_18 = add_form_df(playing_statistics_18)

# Rearrange columns
cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 
        'HTP', 'ATP', 'HM1', 'HM2', 'HM3', 'HM4', 'HM5', 'AM1', 'AM2', 'AM3', 'AM4', 'AM5']

playing_statistics_1 = playing_statistics_1[cols]
playing_statistics_2 = playing_statistics_2[cols]
playing_statistics_3 = playing_statistics_3[cols]
playing_statistics_4 = playing_statistics_4[cols]
playing_statistics_5 = playing_statistics_5[cols]
playing_statistics_6 = playing_statistics_6[cols]
playing_statistics_7 = playing_statistics_7[cols]
playing_statistics_8 = playing_statistics_8[cols]
playing_statistics_9 = playing_statistics_9[cols]
playing_statistics_10 = playing_statistics_10[cols]
playing_statistics_11 = playing_statistics_11[cols]
playing_statistics_12 = playing_statistics_12[cols]
playing_statistics_13 = playing_statistics_13[cols]
playing_statistics_14 = playing_statistics_14[cols]
playing_statistics_15 = playing_statistics_15[cols]
playing_statistics_16 = playing_statistics_16[cols]
playing_statistics_17 = playing_statistics_17[cols]
playing_statistics_18 = playing_statistics_18[cols]

# ============================================================================
# ADD MATCHWEEK
# ============================================================================

def get_mw(playing_stat):
    """Add matchweek column"""
    j = 1
    MatchWeek = []
    for i in range(380):
        MatchWeek.append(j)
        if ((i + 1) % 10) == 0:
            j = j + 1
    playing_stat['MW'] = MatchWeek
    return playing_stat

playing_statistics_1 = get_mw(playing_statistics_1)
playing_statistics_2 = get_mw(playing_statistics_2)
playing_statistics_3 = get_mw(playing_statistics_3)
playing_statistics_4 = get_mw(playing_statistics_4)
playing_statistics_5 = get_mw(playing_statistics_5)
playing_statistics_6 = get_mw(playing_statistics_6)
playing_statistics_7 = get_mw(playing_statistics_7)
playing_statistics_8 = get_mw(playing_statistics_8)
playing_statistics_9 = get_mw(playing_statistics_9)
playing_statistics_10 = get_mw(playing_statistics_10)
playing_statistics_11 = get_mw(playing_statistics_11)
playing_statistics_12 = get_mw(playing_statistics_12)
playing_statistics_13 = get_mw(playing_statistics_13)
playing_statistics_14 = get_mw(playing_statistics_14)
playing_statistics_15 = get_mw(playing_statistics_15)
playing_statistics_16 = get_mw(playing_statistics_16)
playing_statistics_17 = get_mw(playing_statistics_17)
playing_statistics_18 = get_mw(playing_statistics_18)

# ============================================================================
# COMBINE ALL SEASONS
# ============================================================================

playing_stat = pd.concat([
    playing_statistics_1, playing_statistics_2, playing_statistics_3,
    playing_statistics_4, playing_statistics_5, playing_statistics_6,
    playing_statistics_7, playing_statistics_8, playing_statistics_9,
    playing_statistics_10, playing_statistics_11, playing_statistics_12,
    playing_statistics_13, playing_statistics_14, playing_statistics_15,
    playing_statistics_16, playing_statistics_17, playing_statistics_18
], ignore_index=True)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def get_form_points(string):
    """Calculate form points from result string"""
    sum = 0
    for letter in string:
        sum += get_points(letter)
    return sum

playing_stat['HTFormPtsStr'] = (playing_stat['HM1'] + playing_stat['HM2'] + 
                                playing_stat['HM3'] + playing_stat['HM4'] + 
                                playing_stat['HM5'])
playing_stat['ATFormPtsStr'] = (playing_stat['AM1'] + playing_stat['AM2'] + 
                                playing_stat['AM3'] + playing_stat['AM4'] + 
                                playing_stat['AM5'])

playing_stat['HTFormPts'] = playing_stat['HTFormPtsStr'].apply(get_form_points)
playing_stat['ATFormPts'] = playing_stat['ATFormPtsStr'].apply(get_form_points)

# Win/Loss Streaks
def get_3game_ws(string):
    return 1 if string[-3:] == 'WWW' else 0

def get_5game_ws(string):
    return 1 if string == 'WWWWW' else 0

def get_3game_ls(string):
    return 1 if string[-3:] == 'LLL' else 0

def get_5game_ls(string):
    return 1 if string == 'LLLLL' else 0

playing_stat['HTWinStreak3'] = playing_stat['HTFormPtsStr'].apply(get_3game_ws)
playing_stat['HTWinStreak5'] = playing_stat['HTFormPtsStr'].apply(get_5game_ws)
playing_stat['HTLossStreak3'] = playing_stat['HTFormPtsStr'].apply(get_3game_ls)
playing_stat['HTLossStreak5'] = playing_stat['HTFormPtsStr'].apply(get_5game_ls)

playing_stat['ATWinStreak3'] = playing_stat['ATFormPtsStr'].apply(get_3game_ws)
playing_stat['ATWinStreak5'] = playing_stat['ATFormPtsStr'].apply(get_5game_ws)
playing_stat['ATLossStreak3'] = playing_stat['ATFormPtsStr'].apply(get_3game_ls)
playing_stat['ATLossStreak5'] = playing_stat['ATFormPtsStr'].apply(get_5game_ls)

# Goal Difference
playing_stat['HTGD'] = playing_stat['HTGS'] - playing_stat['HTGC']
playing_stat['ATGD'] = playing_stat['ATGS'] - playing_stat['ATGC']

# Point Differences
playing_stat['DiffPts'] = playing_stat['HTP'] - playing_stat['ATP']
playing_stat['DiffFormPts'] = playing_stat['HTFormPts'] - playing_stat['ATFormPts']

# Scale by Matchweek
cols = ['HTGD', 'ATGD', 'DiffPts', 'DiffFormPts', 'HTP', 'ATP']
playing_stat.MW = playing_stat.MW.astype(float)

for col in cols:
    playing_stat[col] = playing_stat[col] / playing_stat.MW

# Convert to binary classification
def only_hw(string):
    return 'H' if string == 'H' else 'NH'

playing_stat['FTR'] = playing_stat.FTR.apply(only_hw)

# Test set
playing_stat_test = playing_stat[5700:]

# Save datasets
playing_stat.to_csv('final_dataset.csv')
playing_stat_test.to_csv("test_set.csv")

# ============================================================================
# DATA ANALYSIS
# ============================================================================

print("="*80)
print("DATASET SUMMARY")
print("="*80)

dataset = pd.read_csv('final_dataset.csv')
print(f"Dataset shape: {dataset.shape}")
print(f"\nFirst few rows:")
print(dataset.head())
print(f"\nColumns: {list(dataset.keys())}")

# Win rate analysis
n_matches = dataset.shape[0]
n_features = dataset.shape[1] - 1
n_homewins = len(dataset[dataset.FTR == 'H'])
win_rate = (float(n_homewins) / n_matches) * 100

print("\n" + "="*80)
print("MATCH STATISTICS")
print("="*80)
print(f"Total number of matches: {n_matches}")
print(f"Number of features: {n_features}")
print(f"Number of matches won by home team: {n_homewins}")
print(f"Win rate of home team: {win_rate:.2f}%")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

# Remove columns
dataset2 = dataset.copy().drop(columns=[
    'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
    'HTGS', 'ATGS', 'HTGC', 'ATGC',
    'HM4', 'HM5', 'AM4', 'AM5', 'MW', 'HTFormPtsStr',
    'ATFormPtsStr', 'HTFormPts', 'ATFormPts', 'HTWinStreak3',
    'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5', 'ATWinStreak3',
    'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5', 'DiffPts'
])

# Separate features and target
X_all = dataset2.drop(['FTR'], axis=1)
y_all = dataset2['FTR']

# Standardize data
cols_scale = [['HTGD', 'ATGD', 'HTP', 'ATP']]
for col in cols_scale:
    X_all[col] = scale(X_all[col])

# Convert form columns to string
X_all.HM1 = X_all.HM1.astype('str')
X_all.HM2 = X_all.HM2.astype('str')
X_all.HM3 = X_all.HM3.astype('str')
X_all.AM1 = X_all.AM1.astype('str')
X_all.AM2 = X_all.AM2.astype('str')
X_all.AM3 = X_all.AM3.astype('str')

# Preprocess features
def preprocess_features(X):
    """Convert categorical variables to dummy variables"""
    output = pd.DataFrame(index=X.index)
    
    for col, col_data in X.items():
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print(f"\nProcessed feature columns ({len(X_all.columns)} total features)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.3, random_state=2, stratify=y_all
)

# ============================================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================================

print("\n" + "="*80)
print("LOGISTIC REGRESSION MODEL")
print("="*80)

lr_classifier = LogisticRegression(random_state=0, max_iter=1000)
lr_classifier.fit(X_train, y_train)
lr_pred = lr_classifier.predict(X_test)

lr_cm = confusion_matrix(y_test, lr_pred)
print("\nConfusion Matrix:")
print(lr_cm)
print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

# ============================================================================
# MODEL 2: SVM
# ============================================================================

print("\n" + "="*80)
print("SVM MODEL")
print("="*80)

svm_classifier = SVC(kernel='rbf', random_state=0)
svm_classifier.fit(X_train, y_train)
svm_pred = svm_classifier.predict(X_test)

svm_cm = confusion_matrix(y_test, svm_pred)
print("\nConfusion Matrix:")
print(svm_cm)
print("\nClassification Report:")
print(classification_report(y_test, svm_pred))

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)