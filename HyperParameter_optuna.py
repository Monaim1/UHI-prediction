import optuna
from catboost import CatBoostRegressor, Pool
import pandas as pd
from functools import partial
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
import json



def objective(trial, X, y):

    params = {
        'iterations': 12000,  # Fixed number of iterations
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'depth': trial.suggest_int('depth', 5, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 20),
        'loss_function': 'RMSE',
        'early_stopping_rounds':150,
        'eval_metric': 'R2',
        'random_seed': 42,
        'verbose': False,  # Suppress CatBoost's internal logs
    }
    
    model = CatBoostRegressor(**params)
    scores = cross_val_score(model, X, y, cv=4, scoring='r2', n_jobs=-1)
    
    return np.mean(scores)

def main():
    with open('_Models/CatBoost/Z_Train_Satellite_Weather_PolyLandCover500Circles_Buildings14_Ecostress_HVSR.json') as f:
        top_features_cbm = json.load(f)

    train_FEdata = pd.read_csv("Train_Data/Final/Z_Train_Satellite_Weather_PolyLandCover500Circles_Buildings14_Ecostress_HVSR.csv")
    CNN_Preds= pd.read_csv("Train_Data/Final/_Train_CNN_Preds.csv")
    loaded_data = CNN_Preds.merge(train_FEdata, on=["Latitude", "Longitude"])
    loaded_data = loaded_data[['UHI Index',"LandCover_CNN"]+ top_features_cbm]
    loaded_data

    X, y = loaded_data.drop(columns=['UHI Index']), loaded_data['UHI Index']

    print("Data Loaded, Starting Tuning...")
    study = optuna.create_study(direction='maximize')
    objective_with_data = partial(objective, X=X, y=y)
    study.optimize(objective_with_data, n_trials=25, n_jobs=3)
    
    print("Best parameters:", study.best_params)
    print("Best Validation R2:", study.best_value)

if __name__ == '__main__':
    main()
