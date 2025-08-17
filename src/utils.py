from src.logger import logging
from src.exception import CustomException
import sys
import os
import pickle

from sklearn.model_selection  import GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import pandas as pd
import numpy as np



def saved_obj(file_path, obj):
    try:
        # Ensure directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save object to file
        with open(file_path, 'wb') as file_obj:  # <-- use full file path here
            pickle.dump(obj, file_obj)          # <-- dump object to file

    except Exception as e:
        raise CustomException(e, sys)
    
    
    
def evaluate_metrics(X_train, y_train, X_test, y_test, models, params):
    try:
        report = []
        best_score= float('-inf')
        best_model_name = None
        best_model = None
        
        for model_name,model in models.items():
            logging.info(f"evaluate model: {model_name}")
            param = params.get(model_name, {})
            if param:
                gs = GridSearchCV(model,param_grid=param,cv=3,scoring='r2',n_jobs=-1)
                gs.fit(X_train,y_train)
                current_model = gs.best_estimator_
            else:
                model.fit(X_train,y_train)
                current_model = model
            y_pred = current_model.predict(X_test)
                
            # evaluate the model
            mse = mean_squared_error(y_test,y_pred)
            mae = mean_absolute_error(y_test,y_pred)
            r2score = r2_score(y_test,y_pred)
            rmse = np.sqrt(mse)
                
            if r2score > best_score:
                best_score = r2score
                best_model_name = model_name
                best_model = current_model
            report.append({
                'model name':model_name,
                'mean squared error':mse,
                'mean absolute error':mae,
                'r2 score':r2score,
                'root mean square error':rmse
                })
        logging.info(f"Best Model: {best_model_name} with R2 Score: {best_score}")
        return report, best_model_name, best_model, best_score
    except Exception as e:
        raise CustomException(e,sys)

def load_obj(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)