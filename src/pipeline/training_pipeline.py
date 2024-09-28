# Step 9
import os # to create file path --> in linux server path created only this module
## Assign path of both train and test files
from src.logger import logging
import sys # is for system error
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
## Data classes
from dataclasses import dataclass
# from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.inititate_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.inititate_model_training(train_arr,test_arr)