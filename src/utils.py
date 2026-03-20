import os
import sys
from typing import Dict

import dill
from sklearn.metrics import accuracy_score, f1_score

from src.exception import CustomException


def save_object(file_path: str, obj) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as error:
        raise CustomException(error, sys) from error


def load_object(file_path: str):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as error:
        raise CustomException(error, sys) from error


def evaluate_classification_models(
    x_train,
    y_train,
    x_test,
    y_test,
    models: Dict[str, object],
) -> Dict[str, Dict[str, float]]:
    try:
        report = {}

        for model_name, model in models.items():
            model.fit(x_train, y_train)

            train_predictions = model.predict(x_train)
            test_predictions = model.predict(x_test)

            report[model_name] = {
                "train_accuracy": accuracy_score(y_train, train_predictions),
                "test_accuracy": accuracy_score(y_test, test_predictions),
                "test_f1": f1_score(y_test, test_predictions, average="weighted"),
            }

        return report
    except Exception as error:
        raise CustomException(error, sys) from error
