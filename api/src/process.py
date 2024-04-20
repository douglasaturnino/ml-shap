import pickle

import numpy as np
import pandas as pd
from config.config import config
from src.models import ShapOutput


class DataProcess:
    def __init__(self) -> None:
        self.model = pickle.load(open(config["model"]["ml_path"], "rb"))
        self.shap_model = pickle.load(open(config["model"]["shap_path"], "rb"))
        self.th = config["model"]["threshold"]
        self.columns = config["model_columns"]["x_columns"]

    def rename_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        data_df = data.copy()
        data_df.columns = self.columns
        return data_df

    def predict_and_prepare_results(self, data_df):
        prob = self.model.predict_proba(data_df)[:, 1].astype(float)
        data_df["predict"] = (
            self.model.predict_proba(data_df)[:, 1] > self.th
        ).astype(int)

        data_df["resultado"] = data_df["predict"].apply(
            lambda x: "Adimplente" if x == 0 else "Inadimplente"
        )

        data_df["prob_inadimplencia"] = prob
        data_df["predict"] = data_df.apply(
            lambda x: x["resultado"]
            + "-"
            + str(round(x["prob_inadimplencia"], 2)),
            axis=1,
        )
        return data_df

    def transform_data(self, data):
        data_pp1 = self.model.named_steps["rare_encoder"].transform(data)
        data_pp2 = self.model.named_steps["categorical_encoder"].transform(
            data_pp1
        )
        return data_pp2

    def calculate_shap_values(self, data_pp):
        shap_values = self.shap_model(data_pp)
        values_list = (
            shap_values.values.tolist()
            if isinstance(shap_values.values, np.ndarray)
            else shap_values.values
        )
        data_list = (
            shap_values.data.tolist()
            if isinstance(shap_values.data, np.ndarray)
            else shap_values.data
        )
        shap_client = ShapOutput(
            values=values_list,
            base_values=shap_values.base_values,
            data=data_list,
        )
        return shap_client

    def prepare_output(self, data_df, shap_client, data_pp):
        if data_df.shape[0] == 1:
            test = {
                "resultado": data_df["resultado"][0],
                "probabilidade": round(data_df["prob_inadimplencia"][0], 2),
                "shap_output": shap_client.model_dump(),
                "feature_names_values": data_pp.to_dict(orient="records")[0],
            }
            return test
        else:
            return data_df["predict"].tolist()

    def process_data(self, data):
        data_df = self.rename_columns(data)
        data_pp = self.transform_data(data)
        data_df = self.predict_and_prepare_results(data_df)
        shap_client = self.calculate_shap_values(data_pp)
        return self.prepare_output(data_df, shap_client, data_pp)
