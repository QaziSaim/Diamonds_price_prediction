## Step 10
import sys
import os
from src.logger import logging
from src.exception import CustomException
from src.utils import load_objects
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl') # deployement happend in linux and this wil run linux as well
            # preprocessor_path = 'artifacts/preprocessor.pkl' this will only run in windows
            model_path = os.path.join('artifacts','model.pkl')


            ## How do we load our file
            preprocessor=load_objects(preprocessor_path)
            model=load_objects(model_path)

            # Scaled the data
            data_scaled = preprocessor.transform(features)

            pred=model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
class CustomData:
    def __init__(self,
                carat: float,
                depth: float,
                table: float,
                x: float,
                y: float,
                z: float,
                cut:  str,
                color:  str,
                clarity: str):
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut=cut
        self.color=color
        self.clarity=clarity
    def get_data_as_dataframe(self):
        try:
            custome_data_input_dict = {
                                        'carat': [self.carat],
                                        'cut': [self.cut],
                                        'color': [self.color],
                                        'clarity': [self.clarity],
                                        'depth': [self.depth],
                                        'table': [self.table],
                                        'x': [self.x],
                                        'y': [self.y],
                                        'z': [self.z]
                                    }
            df = pd.DataFrame(custome_data_input_dict)
            logging.info("Dataframe Gatherd")
            return df

        except Exception as E:
            logging.info("Exception occured in prediction pipeline")
            raise CustomException(E,sys)

        
    
        