from src.logger import logging
from src.exception import CustomException
import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import(
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVR

from src.utils import saved_obj,evaluate_metrics

@dataclass

class ModelTrainerConfig:
    ModelTrainer_path_obj:str = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array, test_array):
        try:
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]
            
            
            ## initiate the model
            models = {
                "LinearRegression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "SupportVectorRegressor":SVR()
                
            }
            
            params ={
                "LinearRegression":{},
                "Ridge":{
                    'alpha':[0.1,0.2,1]
                },
                "Lasso":{
                    'alpha':[0.1,0.2,1]
                    
                },
                "DecisionTreeRegressor":{
                    'max_depth':[3,5,None],
                    'max_features':[3,5,10],
                    'random_state':[None]
                    
                },
                'RandomForestRegressor':{
                    'n_estimators':[10,50,100],
                    'max_depth':[3,5,10]
                },
                "AdaBoostRegressor":{
                    "n_estimators":[50,100,200],
                    'learning_rate':[0.1,0.2,0.01],
                    'random_state':[0]
                },
                'GradientBoostingRegressor':{
                    'n_estimators':[50,100,200],
                    'max_depth':[3,5,10],
                    'learning_rate':[0.1,0.01,0.02]
                },
                "SupportVectorRegressor":{
                    'kernel':['rbf'],
                    'degree':[3],
                    'C':[1.0]
                }
                
            }
            
            
            report,best_model_name, best_model, best_score = evaluate_metrics(X_train,y_train,X_test,y_test,models,params)
            
            saved_obj(
                file_path = self.model_trainer_config.ModelTrainer_path_obj,
                obj= best_model)
            
            logging.info(f"Best Model: {best_model_name} saved at {self.model_trainer_config.ModelTrainer_path_obj}")
            
            return best_model_name,best_model

            
            
            
        except Exception as e:
            raise CustomException(e,sys)