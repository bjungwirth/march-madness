import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.interpolate import UnivariateSpline

# Custom Cauchy objective function
def cauchyobj(preds, dtrain):
    labels = dtrain.get_label()
    c = 5000
    x = preds - labels
    grad = x / (x**2/c**2 + 1)
    hess = -c**2 * (x**2 - c**2) / (x**2 + c**2)**2
    return grad, hess

class TournamentPredictor:
    def __init__(self):
        # Load parameters
        self.params = self.load_parameters()
        self.features = self.load_features()
        # Load trained models
        self.models = []
        self.splines = []
        for i in range(1, 11):
            model_path = f"data/xgb_model_{i}.model"
            self.models.append(xgb.Booster(model_file=model_path))
            # Load spline data and recreate splines
            spline_data = pd.read_csv(f"data/spline_{i}.csv")
            self.splines.append(UnivariateSpline(spline_data['x'], spline_data['y']))
            
    def load_parameters(self):
        params_df = pd.read_csv("data/xgb_parameters.csv")
        params = {row['parameter']: row['value'] for _, row in params_df.iterrows()}
        # Convert numeric parameters
        for param in ['eta', 'subsample', 'colsample_bytree', 'min_child_weight', 'gamma', 'max_depth']:
            if param in params:
                params[param] = float(params[param])
        # Set custom objective
        params['objective'] = cauchyobj
        return params
    
    def load_features(self):
        # Load feature names used in the model
        feature_df = pd.read_csv("data/xgb_features.csv", nrows=1)
        return [col for col in feature_df.columns if col not in ["T1", "T2", "Season"]]
    
    def retrain_models(self, updated_teams):
        """Retrain XGBoost models with updated team strengths"""
        # Load original data
        data = pd.read_csv("data/xgb_features.csv")
        
        # Update team strengths in the data
        for team_id, strength in updated_teams.items():
            # Update where T1 matches team_id
            data.loc[data['T1'] == team_id, 'quality_march_T1'] = strength
            # Update where T2 matches team_id
            data.loc[data['T2'] == team_id, 'quality_march_T2'] = strength
        
        # Prepare training data
        X = data[self.features].values
        y = data['ResultDiff'].values if 'ResultDiff' in data.columns else np.zeros(len(data))
        dtrain = xgb.DMatrix(X, label=y)
        
        # Retrain models
        self.models = []
        for i in range(10):
            np.random.seed(i+1)
            model = xgb.train(
                params=self.params,
                dtrain=dtrain,
                num_boost_round=200,  # Fewer rounds for quick updates
                obj=cauchyobj
            )
            self.models.append(model)
            
            # Update splines too
            preds = model.predict(dtrain)
            labels = (y > 0).astype(int)
            self.splines[i] = UnivariateSpline(preds, labels, s=len(preds)/2)
        
    def predict_matchup(self, matchup_df):
        """Predict win probability and point differential for matchups"""
        # Ensure matchup_df has all required features
        for feature in self.features:
            if feature not in matchup_df.columns:
                matchup_df[feature] = 0
        
        # Create DMatrix
        dtest = xgb.DMatrix(matchup_df[self.features].values)
        
        # Make predictions with all models
        probs = []
        point_diffs = []
        
        for i in range(len(self.models)):
            preds = self.models[i].predict(dtest)
            probs.append(self.splines[i](preds))
            point_diffs.append(preds)
        
        # Average predictions
        matchup_df['Pred'] = np.mean(probs, axis=0)
        matchup_df['Pt_Diff'] = np.mean(point_diffs, axis=0)
        
        return matchup_df[['ID', 'Pred', 'Pt_Diff']] 