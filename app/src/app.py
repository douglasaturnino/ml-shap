import json
import os

import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st
from config.config import config
from matplotlib import pyplot as plt

URL_PREDICT = os.getenv(
    "URL_PREDICT", config["url_request"]["path_predict_localhost"]
)
URL_PREDICT_FILE = os.getenv(
    "URL_PREDICT_FILE", config["url_request"]["path_predict_file_localhost"]
)

st.set_page_config(page_title="The Machine Learning App", layout="wide")


def input_cliente() -> dict:
    """
    Função para coletar as informações do cliente a partir do sidebar.

    Returns:
        Dict: Um objeto contendo as informações do cliente.
    """
    st.sidebar.header("Input informações do cliente")
    dependentes = st.sidebar.selectbox("Tem dependentes?", ["S", "N"])
    estado_civil = st.sidebar.selectbox(
        "Qual Estado Civil?",
        [
            "solteiro ",
            "casado(a) com comunhao de bens",
            "casado(a) com comunhao parcial de bens",
            "casado(a) com separacao de bens",
            "divorciado",
            "separado judicialmente",
            "viuvo(a)",
            "outros",
        ],
    )
    idade = st.sidebar.number_input("Idade", min_value=18, max_value=130)
    cheque_sem_fundo = st.sidebar.selectbox(
        "Já passou cheque sem fundo?", ["S", "N"]
    )
    valor_emprestimo = st.sidebar.number_input("Valor do Empréstimo")
    return {
        "dependentes": dependentes,
        "estado_civil": estado_civil,
        "idade": idade,
        "cheque_sem_fundo": cheque_sem_fundo,
        "valor_emprestimo": valor_emprestimo,
    }


def predict(data: dict) -> tuple:
    """
    Função para fazer a previsão usando os dados fornecidos.

    Args:
        data (dict): Um dicionário contendo as informações do cliente.

    Returns:
        tuple: Uma tupla contendo a previsão, a probabilidade de inadimplência e a saída SHAP.
    """
    filtered_data = {
        col: data[col] for col in config["model_columns"]["x_columns"]
    }
    response = requests.post(URL_PREDICT, json=filtered_data)
    prediction = response.json()["resultado"]
    probability = response.json()["probabilidade"]
    shap_output = response.json()["shap_output"]
    return prediction, probability, shap_output


def display_shap_plot(shap_output: dict, data: pd.DataFrame) -> None:
    """
    Função para exibir o gráfico SHAP.

    Args:
        shap_output (dict): A saída SHAP do modelo.

    Returns:
        None
    """
    st.subheader("Gráfico SHAP")
    values = np.array(shap_output["values"][0])
    base_value = shap_output["base_values"]
    feature_values = np.array(shap_output["data"][0])
    display_data = list(data.values())
    feature_names = list(data.keys())
    explainer = shap.Explanation(
        values=values,
        base_values=base_value,
        data=feature_values,
        feature_names=feature_names,
        display_data=display_data,
    )
    fig, ax = plt.subplots()
    shap.plots.waterfall(explainer, show=False)
    st.pyplot(fig)


def input_csv() -> pd.DataFrame:
    """
    Função para receber e processar um arquivo CSV de entrada.

    Returns:
    - DataFrame: Um DataFrame contendo os dados do arquivo CSV.
    """
    uploaded_file = st.sidebar.file_uploader(
        "Upload de CSV com informações clientes", key="2"
    )
    if uploaded_file is not None:
        df_csv = pd.read_csv(uploaded_file, sep=",")
        json_df = json.dumps(
            df_csv[config["model_columns"]["x_columns"]].to_dict(
                orient="records"
            )
        )
        response = requests.post(URL_PREDICT_FILE, json=json_df)
        df_csv["predict"] = json.loads(response.text)
        df_csv[["predict", "prob_inadimplente"]] = df_csv["predict"].str.split(
            "-", expand=True
        )
        return df_csv


def display_predictions(df_csv: pd.DataFrame) -> None:
    """
    Função para exibir as previsões em uma tabela.

    Args:
        df_csv (DataFrame): O DataFrame contendo os dados e as previsões.

    Returns:
        None
    """
    st.subheader("Tabela de Previsões")
    st.dataframe(df_csv, 2000, 1000)
    st.download_button(
        label="Download CSV",
        data=df_csv.to_csv(index=False),
        file_name="predic.csv",
        mime="text/csv",
    )


def run() -> None:
    """
    Função principal que executa o aplicativo.

    Returns:
        None
    """
    st.title("APP Previsão de Inadimplência Bancária")
    data = input_cliente()

    if st.sidebar.button("Predict"):
        prediction, probability, shap_output = predict(data)
        st.sidebar.success(
            f"A predição do modelo: {prediction} / probabilidade de Inadimplência = {probability}"
        )
        display_shap_plot(shap_output, data)

    df_csv = input_csv()
    if df_csv is not None:
        display_predictions(df_csv)
