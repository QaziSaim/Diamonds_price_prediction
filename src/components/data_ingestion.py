# Step 3
import os # to create file path --> in linux server path created only this module
## Assign path of both train and test files
from src.logger import logging
import sys # is for system error
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
## Data classes
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
"""
Dataclass --> directly created variable name
just want to create class variable and don't want to create functionality within it then we use data class
"""



"""
path need to be store some seprate class
data class store only the path of dataset 
data class - class variable

artifact all the file name store in this folder 
"""
### Initialize the  data ingestion configuration
@dataclass # constructor use 
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')
### create class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig() # This called above three path
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion methods Starts")

        try:
            # df=pd.read_csv('notebooks/data/gemstone.csv')
            df=pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))
            logging.info("Dataset read as pandas Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of Data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Exception at data ingestion stage")
            raise CustomException(e,sys)



# Shift it to training_pipeline after created model_trainer
# if __name__ == '__main__':
#     obj = DataIngestion()
#     train_data_path,test_data_path=obj.initiate_data_ingestion()
#     data_transformation = DataTransformation()
#     train_arr,test_arr,_=data_transformation.inititate_data_transformation(train_data_path,test_data_path)