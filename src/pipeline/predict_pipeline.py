import pandas as pd
from typing import Optional

from src.components.data_transformation import DROP_COLUMNS
from src.utils import load_object


class PredictPipeline:
    def __init__(self) -> None:
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"
        self.target_encoder_path = "artifacts/target_encoder.pkl"

    def predict(self, features: pd.DataFrame):
        model = load_object(self.model_path)
        preprocessor = load_object(self.preprocessor_path)
        target_encoder = load_object(self.target_encoder_path)

        model_ready_features = features.drop(
            columns=[column for column in DROP_COLUMNS if column in features.columns],
            errors="ignore",
        )
        transformed_features = preprocessor.transform(model_ready_features)
        predictions = model.predict(transformed_features)
        return target_encoder.inverse_transform(predictions.astype(int))


class CustomData:
    def __init__(
        self,
        region: str,
        fresh: float,
        milk: float,
        grocery: float,
        frozen: float,
        detergents_paper: float,
        delicatessen: float,
        buyer_spender: Optional[int] = None,
    ) -> None:
        self.buyer_spender = buyer_spender
        self.region = region
        self.fresh = fresh
        self.milk = milk
        self.grocery = grocery
        self.frozen = frozen
        self.detergents_paper = detergents_paper
        self.delicatessen = delicatessen

    def get_data_as_data_frame(self) -> pd.DataFrame:
        data = {
            "Buyer/Spender": [self.buyer_spender if self.buyer_spender is not None else 0],
            "Region": [self.region],
            "Fresh": [self.fresh],
            "Milk": [self.milk],
            "Grocery": [self.grocery],
            "Frozen": [self.frozen],
            "Detergents_Paper": [self.detergents_paper],
            "Delicatessen": [self.delicatessen],
        }

        return pd.DataFrame(data)
