import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_classification_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> dict:
        try:
            logging.info("Splitting transformed training and test arrays")

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "logistic_regression": LogisticRegression(max_iter=1000),
                "random_forest": RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                ),
                "gradient_boosting": GradientBoostingClassifier(random_state=42),
                "knn": KNeighborsClassifier(n_neighbors=5),
            }

            model_report = evaluate_classification_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
            )

            best_model_name = max(
                model_report,
                key=lambda name: model_report[name]["test_accuracy"],
            )
            best_model = models[best_model_name]
            best_score = model_report[best_model_name]["test_accuracy"]

            if best_score < 0.6:
                raise CustomException("No suitable model found for the dataset", sys)

            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            logging.info("Best model selected: %s", best_model_name)

            predictions = best_model.predict(x_test)
            test_accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions, output_dict=True)

            return {
                "best_model_name": best_model_name,
                "test_accuracy": test_accuracy,
                "classification_report": report,
                "all_model_scores": model_report,
            }
        except Exception as error:
            raise CustomException(error, sys) from error
