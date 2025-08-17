from src.logger import logging
from src.exception import CustomException
import os
import sys
from src.utils import load_obj
import pandas as pd



transformer = load_obj("artifacts/preprocessor.pkl")
model = load_obj("artifacts/model.pkl")



input_data = pd.DataFrame ({
    'gender':['Male'],
    'race_ethnicity':['group C'],
    'parental_level_of_education':['high school'],
    'lunch':['standard'],
    'test_preparation_course': ['none'],
    'reading_score':[40],
    'writing_score':[60]
    
})

new_transformer = transformer.transform(input_data)
predict = model.predict(new_transformer)

print(f"Model Prediction:{predict}")
