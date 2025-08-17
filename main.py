from src.logger import logging
from src.exception import CustomException
import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer

def main():
    try:
        # step: 1 Data Ingestion
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")
        
        # step: 2 Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr,_=data_transformation.initiate_data_transformer(train_data,test_data)
        
        # step: 3 data validation 
        data_validation = DataValidation(train_data, test_data)
        data_validation.initiate_data_validation()
        logging.info("data validation completed")
        
        # step: 4 model trainer
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr,test_arr)
        
        
        
    except Exception as e:
        raise CustomException(e,sys)
    
    
    
if __name__=="__main__":
    main()

# logging.info("logging has started")



# try:
#     result = 1/0
#     logging.info("1 divided by 0")
# except Exception as e:
#     raise CustomException(e,sys)
    