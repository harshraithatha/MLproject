import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(
        self, source_path: str = "Wholesale+Customers+Data.csv"
    ) -> tuple[str, str]:
        logging.info("Entered the data ingestion method")
        try:
            dataframe = pd.read_csv(source_path)
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            dataframe.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw dataset stored at %s", self.ingestion_config.raw_data_path)

            train_set, test_set = train_test_split(
                dataframe,
                test_size=0.2,
                random_state=42,
                stratify=dataframe["Channel"],
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test split completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as error:
            raise CustomException(error, sys) from error
