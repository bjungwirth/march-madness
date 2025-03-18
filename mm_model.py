import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import KFold
import random
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM

# Load data
regresults = pd.read_csv("data/kaggle/MRegularSeasonDetailedResults.csv")
results = pd.read_csv("data/kaggle/MNCAATourneyDetailedResults.csv")
sub = pd.read_csv("data/kaggle/sample_submission.csv")
seeds = pd.read_csv("data/kaggle/MNCAATourneySeeds.csv")
seeds_2024 = pd.read_csv("data/kaggle/2024_tourney_seeds.csv")

# Clean seed data - remove non-numeric characters
seeds['Seed'] = seeds['Seed'].str.replace(r'[^0-9]', '', regex=True).astype(int)

### Collect regular season results - double the data by swapping team positions
r1 = regresults[['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 
                'WFGA', 'WAst', 'WBlk', 'LFGA', 'LAst', 'LBlk', 'WLoc']]
r2 = regresults[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'NumOT', 
                'LFGA', 'LAst', 'LBlk', 'WFGA', 'WAst', 'WBlk', 'WLoc']]

r1.columns = ['Season', 'DayNum', 'T1', 'T1_Points', 'T2', 'T2_Points', 'NumOT', 
             'T1_fga', 'T1_ast', 'T1_blk', 'T2_fga', 'T2_ast', 'T2_blk', 'WLoc']
r2.columns = ['Season', 'DayNum', 'T1', 'T1_Points', 'T2', 'T2_Points', 'NumOT', 
             'T1_fga', 'T1_ast', 'T1_blk', 'T2_fga', 'T2_ast', 'T2_blk', 'WLoc']

regular_season = pd.concat([r1, r2], ignore_index=True)

### Collect tourney results - double the data by swapping team positions
t1 = results[['Season', 'DayNum', 'WTeamID', 'LTeamID', 'WScore', 'LScore', 'WLoc']].copy()
t1['ResultDiff'] = t1['WScore'] - t1['LScore']

t2 = results[['Season', 'DayNum', 'LTeamID', 'WTeamID', 'LScore', 'WScore', 'WLoc']].copy()
t2['ResultDiff'] = t2['LScore'] - t2['WScore']

t1.columns = ['Season', 'DayNum', 'T1', 'T2', 'T1_Points', 'T2_Points', 'WLoc', 'ResultDiff']
t2.columns = ['Season', 'DayNum', 'T1', 'T2', 'T1_Points', 'T2_Points', 'WLoc', 'ResultDiff']

tourney = pd.concat([t1, t2], ignore_index=True)

### Fit model on regular season data - extract team quality
# In Python we'll use a different approach since pymer4 is complex to install
# We'll use a simpler logistic regression model per season

march_teams = seeds[['Season', 'TeamID']].rename(columns={'TeamID': 'Team'})

X = pd.merge(
    regular_season,
    march_teams,
    left_on=['Season', 'T1'],
    right_on=['Season', 'Team']
).drop(columns=['Team'])

X = pd.merge(
    X,
    march_teams,
    left_on=['Season', 'T2'],
    right_on=['Season', 'Team']
).drop(columns=['Team'])

X = X[['Season', 'T1', 'T2', 'T1_Points', 'T2_Points', 'NumOT']].drop_duplicates()

# Convert team IDs to categorical for modeling
X['T1'] = X['T1'].astype(str)
X['T2'] = X['T2'].astype(str)

# Create team quality measures using win probabilities
quality = []
for season in X['Season'].unique():
    season_data = X[(X['Season'] == season) & (X['NumOT'] == 0)].copy()
    season_data['win'] = (season_data['T1_Points'] > season_data['T2_Points']).astype(int)
    
    # Get win rates for each team
    t1_wins = season_data.groupby('T1')['win'].mean().reset_index()
    t1_wins.columns = ['Team_Id', 'win_rate']
    
    t2_losses = 1 - season_data.groupby('T2')['win'].mean().reset_index()
    t2_losses.columns = ['Team_Id', 'loss_rate']
    
    # Combine to get overall quality
    team_quality = pd.merge(t1_wins, t2_losses, on='Team_Id', how='outer').fillna(0.5)
    team_quality['quality'] = (team_quality['win_rate'] + team_quality['loss_rate']) / 2
    team_quality['Season'] = season
    team_quality['Team_Id'] = team_quality['Team_Id'].astype(int)
    
    quality.append(team_quality[['Season', 'Team_Id', 'quality']])

quality = pd.concat(quality, ignore_index=True)

### Regular season statistics
def calc_season_summary(df):
    return df.assign(
        win14days=lambda x: ((x['DayNum'] > 118) & (x['T1_Points'] > x['T2_Points'])).astype(int),
        last14days=lambda x: (x['DayNum'] > 118).astype(int)
    ).groupby(['Season', 'T1']).agg(
        WinRatio14d=('win14days', lambda x: x.sum() / x.count() if x.count() > 0 else 0),
        PointsMean=('T1_Points', 'mean'),
        PointsMedian=('T1_Points', 'median'),
        PointsDiffMean=pd.NamedAgg(column='T1_Points', aggfunc=lambda x, y: (x - y).mean()),
        FgaMean=('T1_fga', 'mean'),
        FgaMedian=('T1_fga', 'median'),
        FgaMin=('T1_fga', 'min'),
        FgaMax=('T1_fga', 'max'),
        AstMean=('T1_ast', 'mean'),
        BlkMean=('T1_blk', 'mean'),
        OppFgaMean=('T2_fga', 'mean'),
        OppFgaMin=('T2_fga', 'min')
    ).reset_index()

# For the PointsDiffMean calculation which is more complex in pandas
regular_season['PointsDiff'] = regular_season['T1_Points'] - regular_season['T2_Points']
season_summary = regular_season.assign(
    win14days=lambda x: ((x['DayNum'] > 118) & (x['T1_Points'] > x['T2_Points'])).astype(int),
    last14days=lambda x: (x['DayNum'] > 118).astype(int)
).groupby(['Season', 'T1']).agg(
    WinRatio14d=('win14days', lambda x: x.sum() / x.count() if x.count() > 0 else 0),
    PointsMean=('T1_Points', 'mean'),
    PointsMedian=('T1_Points', 'median'),
    PointsDiffMean=('PointsDiff', 'mean'),
    FgaMean=('T1_fga', 'mean'),
    FgaMedian=('T1_fga', 'median'),
    FgaMin=('T1_fga', 'min'),
    FgaMax=('T1_fga', 'max'),
    AstMean=('T1_ast', 'mean'),
    BlkMean=('T1_blk', 'mean'),
    OppFgaMean=('T2_fga', 'mean'),
    OppFgaMin=('T2_fga', 'min')
).reset_index()

# Prepare team statistics for both T1 and T2
season_summary_X1 = season_summary.copy()
season_summary_X2 = season_summary.copy()

# Rename columns to indicate team 1 features
rename_cols = {col: f'X1_{col}' for col in season_summary_X1.columns if col not in ['Season', 'T1']}
season_summary_X1 = season_summary_X1.rename(columns=rename_cols)

# Rename columns to indicate team 2 features
rename_cols = {col: f'X2_{col}' for col in season_summary_X2.columns if col not in ['Season', 'T1']}
season_summary_X2 = season_summary_X2.rename(columns=rename_cols)
season_summary_X2 = season_summary_X2.rename(columns={'T1': 'T2'})

### Combine all features into a data frame
data_matrix = tourney.merge(
    season_summary_X1, on=['Season', 'T1']
).merge(
    season_summary_X2, on=['Season', 'T2']
).merge(
    seeds[['Season', 'TeamID', 'Seed']].rename(columns={'TeamID': 'T1', 'Seed': 'Seed1'}), 
    on=['Season', 'T1']
).merge(
    seeds[['Season', 'TeamID', 'Seed']].rename(columns={'TeamID': 'T2', 'Seed': 'Seed2'}), 
    on=['Season', 'T2']
).assign(
    SeedDiff=lambda x: x['Seed1'] - x['Seed2']
).merge(
    quality[['Season', 'Team_Id', 'quality']].rename(columns={'Team_Id': 'T1', 'quality': 'quality_march_T1'}),
    on=['Season', 'T1']
).merge(
    quality[['Season', 'Team_Id', 'quality']].rename(columns={'Team_Id': 'T2', 'quality': 'quality_march_T2'}),
    on=['Season', 'T2']
)

# One-hot encode categorical variables
data_matrix = pd.get_dummies(data_matrix, columns=['WLoc'], drop_first=True)

### Prepare xgboost
features = [col for col in data_matrix.columns if col not in 
           ['Season', 'DayNum', 'T1', 'T2', 'T1_Points', 'T2_Points', 'ResultDiff']]

X_train = data_matrix[features].values
y_train = data_matrix['ResultDiff'].values

# Custom Cauchy objective function for XGBoost
def cauchyobj(preds, dtrain):
    labels = dtrain.get_label()
    c = 5000
    x = preds - labels
    grad = x / (x**2/c**2 + 1)
    hess = -c**2 * (x**2 - c**2) / (x**2 + c**2)**2
    return grad, hess

# XGBoost parameters
xgb_parameters = {
    'booster': 'gbtree',
    'eta': 0.02,
    'subsample': 0.35,
    'colsample_bytree': 0.7,
    'num_parallel_tree': 10,
    'min_child_weight': 40,
    'gamma': 10,
    'max_depth': 3,
    'objective': 'reg:squarederror',  # Will be overridden by custom objective
    'eval_metric': 'mae'
}

# Create 5-fold cross-validation indices
N = len(data_matrix)
fold5list = np.concatenate([
    np.ones(N//5) * 1,
    np.ones(N//5) * 2,
    np.ones(N//5) * 3,
    np.ones(N//5) * 4,
    np.ones(N - 4*(N//5)) * 5
]).astype(int)

### Build cross-validation model, repeated 10-times
iteration_count = []
smooth_model = []

for i in range(1, 11):
    # Set seed for reproducibility
    np.random.seed(i)
    random.seed(i)
    
    # Resample fold split
    folds = []
    fold_list = np.random.permutation(fold5list)
    for k in range(1, 6):
        folds.append(np.where(fold_list == k)[0])
    
    # Create dataset for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Run cross-validation
    np.random.seed(120)
    cv_results = xgb.cv(
        params=xgb_parameters,
        dtrain=dtrain,
        num_boost_round=3000,
        folds=folds,
        obj=cauchyobj,
        early_stopping_rounds=25,
        verbose_eval=False,
        seed=120
    )
    
    # Get best iteration
    best_iteration = cv_results.shape[0]
    iteration_count.append(best_iteration)
    
    # Train model with best iteration
    xgb_model = xgb.train(
        params=xgb_parameters,
        dtrain=dtrain,
        num_boost_round=best_iteration,
        obj=cauchyobj
    )
    
    # Get predictions
    preds = xgb_model.predict(dtrain)
    
    # Fit smooth spline model for probability conversion
    labels = (data_matrix['ResultDiff'] > 0).astype(int)
    smooth_fit = UnivariateSpline(preds, labels, s=len(preds)/2)
    smooth_model.append(smooth_fit)

### Build submission models
submission_model = []

for i in range(1, 11):
    np.random.seed(i)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Train final model with slightly more iterations
    model = xgb.train(
        params=xgb_parameters,
        dtrain=dtrain,
        num_boost_round=round(iteration_count[i-1] * 1.05),
        obj=cauchyobj
    )
    
    submission_model.append(model)

### Run predictions
sub['Season'] = 2024

# Get 2024 seeds
seeds_2024 = seeds[seeds['Season'] == 2024]

# Create all possible matchups
team_ids = seeds_2024['TeamID'].tolist()
matchups = []
for i, t1 in enumerate(team_ids):
    for t2 in team_ids[i+1:]:
        matchups.append({
            'T1': min(t1, t2),
            'T2': max(t1, t2),
            'ID': f"2024_{min(t1, t2)}_{max(t1, t2)}",
            'Season': 2024
        })

matchups = pd.DataFrame(matchups)

# Prepare feature data for predictions
Z = matchups.merge(
    season_summary_X1, on=['Season', 'T1'], how='left'
).merge(
    season_summary_X2, on=['Season', 'T2'], how='left'
).merge(
    seeds[['Season', 'TeamID', 'Seed']].rename(columns={'TeamID': 'T1', 'Seed': 'Seed1'}), 
    on=['Season', 'T1'], how='left'
).merge(
    seeds[['Season', 'TeamID', 'Seed']].rename(columns={'TeamID': 'T2', 'Seed': 'Seed2'}), 
    on=['Season', 'T2'], how='left'
).assign(
    SeedDiff=lambda x: x['Seed1'] - x['Seed2']
).merge(
    quality[['Season', 'Team_Id', 'quality']].rename(columns={'Team_Id': 'T1', 'quality': 'quality_march_T1'}),
    on=['Season', 'T1'], how='left'
).merge(
    quality[['Season', 'Team_Id', 'quality']].rename(columns={'Team_Id': 'T2', 'quality': 'quality_march_T2'}),
    on=['Season', 'T2'], how='left'
)

# One-hot encode categorical variables
# Add dummy WLoc variables (these will be zeros as we don't know location for future games)
for col in ['WLoc_N', 'WLoc_H', 'WLoc_A']:
    if col not in Z.columns:
        Z[col] = 0

# Ensure Z has all the feature columns needed
for feature in features:
    if feature not in Z.columns:
        Z[feature] = 0  # Default value

# Create test data matrix
dtest = xgb.DMatrix(Z[features].values)

# Make predictions
probs = []
point_diff_preds = []

for i in range(10):
    preds = submission_model[i].predict(dtest)
    probs.append(smooth_model[i](preds))
    point_diff_preds.append(preds)

Z['Pred'] = np.mean(probs, axis=0)
Z['Pt_Diff'] = np.mean(point_diff_preds, axis=0)

# Save predictions
Z[['ID', 'Pred', 'Pt_Diff']].to_csv("sub.csv", index=False) 