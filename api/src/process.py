import os
import pickle
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from config.config import config
from src.models import ShapOutput


class DataProcess:
    """
    Classe para processamento de dados e geração de previsões.

    Attributes:
        model (sklearn.base.BaseEstimator): Modelo de aprendizado de máquina para previsão.
        shap_model (callable): Modelo SHAP para geração de feature importance explanations.
        th (float): Threshold para classificação.
        columns (list): Lista de nomes de colunas usados no modelo.

    Methods:
        __init__: Inicializa o DataProcess com modelo, modelo SHAP, Threshold e colunas.
        rename_columns: Renomeia as colunas dos dados de entrada.
        predict_and_prepare_results: Prevê classes e prepara resultados.
        transform_data: Transforma os dados de entrada para a previsão do modelo.
        calculate_shap_values: Calcula os valores SHAP para os dados de entrada.
        prepare_output: Prepara os dados de saída contendo previsões e explicações do modelo SHAP.
        process_data: Processa os dados de entrada para gerar previsões e explicações do modelo SHAP.
    """

    def __init__(self) -> None:
        """
        Inicializa o DataProcess com modelo, modelo SHAP, Threshold e colunas.
        """
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        self.model = pickle.load(open(path + config["model"]["ml_path"], "rb"))
        self.shap_model = pickle.load(
            open(path + config["model"]["shap_path"], "rb")
        )
        self.th = config["model"]["threshold"]
        self.columns = config["model_columns"]["x_columns"]

    def rename_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Renomeia as colunas dos dados de entrada.

        Args:
            data (pandas.DataFrame): Dados de entrada.

        Returns:
            pandas.DataFrame: Dados com colunas renomeadas.
        """
        data_df = data.copy()
        data_df.columns = self.columns
        return data_df

    def predict_and_prepare_results(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prevê classes e prepara resultados.

        Args:
            data (pandas.DataFrame): Dados de entrada com colunas renomeadas.

        Returns:
            pandas.DataFrame: Dados com classes previstas e probabilidades.
        """
        prob = self.model.predict_proba(data)[:, 1].astype(float)
        data["predict"] = (
            self.model.predict_proba(data)[:, 1] > self.th
        ).astype(int)

        data["resultado"] = data["predict"].apply(
            lambda x: "Adimplente" if x == 0 else "Inadimplente"
        )

        data["prob_inadimplencia"] = prob
        data["predict"] = data.apply(
            lambda x: x["resultado"]
            + "-"
            + str(round(x["prob_inadimplencia"], 2)),
            axis=1,
        )
        return data

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma os dados de entrada para a previsão do modelo.

        Args:
            data (pandas.DataFrame): Dados a serem transformados

        Returns:
            pandas.DataFrame: Dados transformados para a previsão do modelo.
        """
        data_pp1 = self.model.named_steps["rare_encoder"].transform(data)
        data_pp2 = self.model.named_steps["categorical_encoder"].transform(
            data_pp1
        )
        return data_pp2

    def calculate_shap_values(self, data_pp: pd.DataFrame) -> ShapOutput:
        """
        Calcula os valores SHAP para os dados de entrada.

        Args:
            data_pp (numpy.ndarray): Dados transformados para a previsão do modelo.

        Returns:
            ShapOutput: Valores SHAP a base SHAP e os dados
        """
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

    def prepare_output(
        self,
        data_df: pd.DataFrame,
        shap_client: ShapOutput,
        data_pp: pd.DataFrame,
    ) -> Union[Dict[str, Any], List[Any]]:
        """
        Prepara os dados de saída contendo previsões e explicações.

        Args:
            data_df (pandas.DataFrame): Dados de entrada com classes previstas e probabilidades.
            shap_client (ShapOutput): Valores SHAP e informações relacionadas.
            data_pp (pandas.DataFrame): Dados transformados para a previsão do modelo.

        Returns:
            dict or list: Saída contendo previsões e explicações.
        """
        if data_df.shape[0] == 1:
            dict_shap = {
                "resultado": data_df["resultado"][0],
                "probabilidade": round(data_df["prob_inadimplencia"][0], 2),
                "shap_output": shap_client.model_dump(),
                "feature_names_values": data_pp.to_dict(orient="records")[0],
            }
            return dict_shap
        else:
            return data_df["predict"].tolist()

    def process_data(self, data: pd.DataFrame) -> Union[Dict[str, Any], List[Any]]:
        """
        Processa os dados de entrada para gerar previsões e explicações do modelo SHAP.

        Args:
            data (pandas.DataFrame): Dados de entrada.

        Returns:
            dict or list: Saída contendo previsões e explicações.
        """
        data_df = self.rename_columns(data)
        data_pp = self.transform_data(data)
        data_df = self.predict_and_prepare_results(data_df)
        shap_client = self.calculate_shap_values(data_pp)
        return self.prepare_output(data_df, shap_client, data_pp)
