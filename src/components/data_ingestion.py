import os
import pandas as pd
from sklearn.model_selection import train_test_split
#from src.exception import CustomException
#from src.logger import logging
import sys
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_path = os.path.join("artifact", "train.csv")
    test_path = os.path.join("artifact", "test.csv")
    raw_path = os.path.join("artifact", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):

        try:
            #logging.info("Starting data ingestion")
            df = pd.read_csv("C:/Users/User/Desktop/practice_OOPS/Student_End_to_end/data.csv")

            

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_path), exist_ok=True)
            #logging.info("artifact directory created")

            df.to_csv(self.data_ingestion_config.raw_path, index = False,header = True)
            #logging.info("splitting in progress")
            train_df, test_df = train_test_split(df, test_size = 0.2, random_state =42)

            train_df.to_csv(self.data_ingestion_config.train_path, index = False, header = True)
            test_df.to_csv(self.data_ingestion_config.test_path, index = False, header = True)

            #logging.info("splitting completed")

            return (
                self.data_ingestion_config.train_path,
                self.data_ingestion_config.test_path
            )

        except Exception as e:
            raise e

if __name__ == "__main__":
    data_ingestion_config = DataIngestionConfig()
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()

