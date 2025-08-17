from src.logger import logging
from src.exception import CustomException
import os
import sys
import pandas as pd


class DataValidation:
    def __init__(self,train_data,test_data):
        self.train_data = train_data
        self.test_data = test_data
        
    def initiate_data_validation(self):
        try:
            train_df = pd.read_csv(self.train_data)
            test_df = pd.read_csv(self.test_data) 
            
            # shape of dataset
            logging.info(f"train data shape: \n{train_df.shape}")
            logging.info(f"test data shape: \n{test_df.shape}")
            
            # checking null value
            logging.info(f"train data null values: \n{train_df.isnull().sum()}")
            logging.info(f"test data null values: \n{test_df.isnull().sum()}")
            
            # checking duplicated value
            logging.info(f"train data duplicate values: \n{train_df.duplicated().sum()}")
            logging.info(f"test data duplicate values: \n{test_df.duplicated().sum()}")
            
            # # data information
            # logging.info(f"train data information: \n{train_df.info()}")
            # logging.info(f"test data information: \n{test_df.info()}")
            
            # statatics test
            logging.info(f"train data statatics info: \n{train_df.describe()}")
            logging.info(f"test data statatics info: \n{test_df.describe()}")#
            
            
            expected_columns = ['gender','race_ethnicity','parental_level_of_education',
                    'lunch','test_preparation_course','reading_score','writing_score']
            missing_columns = [col for col in expected_columns if col not in train_df.columns]
            
            if missing_columns:
                raise CustomException(F"missing columns in train data: {missing_columns}",sys)
        except Exception as e:
            raise CustomException(e,sys)