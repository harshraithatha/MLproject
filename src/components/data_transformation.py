import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


TARGET_COLUMN = "Channel"
DROP_COLUMNS = ["Buyer/Spender"]


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    target_encoder_file_path: str = os.path.join("artifacts", "target_encoder.pkl")


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            numeric_columns = [
                "Fresh",
                "Milk",
                "Grocery",
                "Frozen",
                "Detergents_Paper",
                "Delicatessen",
            ]
            categorical_columns = ["Region"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numeric_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor
        except Exception as error:
            raise CustomException(error, sys) from error

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> tuple[np.ndarray, np.ndarray, str, str]:
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data for transformation")

            preprocessing_obj = self.get_data_transformer_object()
            target_encoder = LabelEncoder()

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN] + DROP_COLUMNS)
            target_feature_train_df = target_encoder.fit_transform(train_df[TARGET_COLUMN])

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN] + DROP_COLUMNS)
            target_feature_test_df = target_encoder.transform(test_df[TARGET_COLUMN])

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )
            save_object(
                file_path=self.data_transformation_config.target_encoder_file_path,
                obj=target_encoder,
            )

            logging.info("Saved preprocessing artifacts")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.target_encoder_file_path,
            )
        except Exception as error:
            raise CustomException(error, sys) from error
