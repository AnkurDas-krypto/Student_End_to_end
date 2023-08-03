import os
from dataclasses import dataclass
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

@dataclass
class ModelTrainerConfig:
    model_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    try:
        def __init__(self):
            self.model_trainer_config = ModelTrainerConfig()


        def evaluate_models(self, X_train, y_train,X_test,y_test,models,param):
            try:
                report = {}

                for i in range(len(list(models))):
                    model = list(models.values())[i]
                    para=param[list(models.keys())[i]]

                    gs = GridSearchCV(model,para,cv=3)
                    gs.fit(X_train,y_train)

                    model.set_params(**gs.best_params_)
                    model.fit(X_train,y_train)

                    #model.fit(X_train, y_train)  # Train model

                    y_train_pred = model.predict(X_train)

                    y_test_pred = model.predict(X_test)

                    train_model_score = r2_score(y_train, y_train_pred)

                    test_model_score = r2_score(y_test, y_test_pred)

                    report[list(models.keys())[i]] = test_model_score

                return report

            except Exception as e:
                raise e
        
        def save_object(self, file_path, obj):
            try:
                dir_path = os.path.dirname(file_path)

                os.makedirs(dir_path, exist_ok=True)

                with open(file_path, "wb") as file_obj:
                    pickle.dump(obj, file_obj)

            except Exception as e:
                raise e

        def initiate_model_trainer(self, train_array, test_array):
            X_train,y_train,X_test,y_test=(
                    train_array[:,:-1],
                    train_array[:,-1],
                    test_array[:,:-1],
                    test_array[:,-1]
                )
            models = {
                    "Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Linear Regression": LinearRegression(),
                    "XGBRegressor": XGBRegressor(),
                    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                    "AdaBoost Regressor": AdaBoostRegressor(),
                }

            params={
                    "Decision Tree": {
                        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        # 'splitter':['best','random'],
                        # 'max_features':['sqrt','log2'],
                    },
                    "Random Forest":{
                        # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    
                        # 'max_features':['sqrt','log2',None],
                        'n_estimators': [8,16,32,64,128,256]
                    },
                    "Gradient Boosting":{
                        # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                        'learning_rate':[.1,.01,.05,.001],
                        'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                        # 'criterion':['squared_error', 'friedman_mse'],
                        # 'max_features':['auto','sqrt','log2'],
                        'n_estimators': [8,16,32,64,128,256]
                    },
                    "Linear Regression":{},
                    "XGBRegressor":{
                        'learning_rate':[.1,.01,.05,.001],
                        'n_estimators': [8,16,32,64,128,256]
                    },
                    "CatBoosting Regressor":{
                        'depth': [6,8,10],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'iterations': [30, 50, 100]
                    },
                    "AdaBoost Regressor":{
                        'learning_rate':[.1,.01,0.5,.001],
                        # 'loss':['linear','square','exponential'],
                        'n_estimators': [8,16,32,64,128,256]
                    }
                    
                }
            
            model_report:dict=self.evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                    raise ("No best model")

            self.save_object(file_path= self.model_trainer_config.model_path, obj= best_model)

            predicted = best_model.predict(X_test)
            r2_square = r2_score(predicted, y_test)
            return r2_square
        
    except Exception as e:
        raise e