from src.logger import logging
from src.exception import CustomException
import os
import sys
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from src.utils import saved_obj

@dataclass

class DataTransformationConfig:
    preprocessor_path_obj:str = os.path.join("artifacts","preprocessor.pkl")
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformer_config = DataTransformationConfig()
        
    
    def get_data_transformer_obj(self):
        try:
            numerical_feature = ['reading_score','writing_score']
            categorical_feature = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            
            num_pipeline =Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore'))
                ]
            )
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipe",num_pipeline,numerical_feature),
                    ("cat_pipe",cat_pipeline,categorical_feature)
                    
                ]
            )
            logging.info(f"numerical feature:{numerical_feature}")
            logging.info(f"categorical feature:{categorical_feature}")
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)#
        
        
        
    def initiate_data_transformer(self,train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("read the data as data frame")
            logging.info(f"obtaining preprocessor path obj")
            
            preprocessor_obj = self.get_data_transformer_obj()
            
            target_feature = 'math_score'
            
            
            input_feature_train_df = train_df.drop(columns='math_score')
            target_feature_train_df = train_df['math_score']
            
            input_feature_test_df = test_df.drop(columns='math_score')
            target_feature_test_df = test_df['math_score']
            
            logging.info(f"applying preprocessor path on training and testing data set")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr,target_feature_train_df.to_numpy()]
            test_arr = np.c_[input_feature_test_arr,target_feature_test_df.to_numpy()]
            
            
            saved_obj(
                file_path=self.data_transformer_config.preprocessor_path_obj,
                obj=preprocessor_obj
            )
            
            return(
                train_arr,test_arr, self.data_transformer_config.preprocessor_path_obj,
                
            )
            
        except Exception as e:
            raise CustomException(e,sys) 