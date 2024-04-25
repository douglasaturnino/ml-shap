import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import yaml
from matplotlib import pyplot as plt
import shap
import os

def import_config(path_yaml):
    with open(path_yaml) as config_file:
        config = yaml.safe_load(config_file)
    return config

path = os.path.dirname(os.path.abspath(__file__))
config = import_config(path + '/config.yaml')

URL_PREDICT =  os.getenv('URL_PREDICT', config['url_request']['path_predict_localhost'])
URL_PREDICT_FILE = os.getenv('URL_PREDICT_FILE', config['url_request']['path_predict_file_localhost'])

st.set_page_config(page_title='The Machine Learning App', layout='wide')

def run():
    st.title('APP Previsão de Inadimplencia Bancário')
    st.sidebar.header('Input informações do cliente')

    # Inputs do usuário no sidebar
    id = st.sidebar.text_input("ID do cliente")
    dependentes = st.sidebar.selectbox("Tem dependestes?", ["S", "N"])
    estado_civil = st.sidebar.selectbox('Qual EStado Cívil?', ['solteiro ', 'casado(a) com comunhao de bens', 'casado(a) com comunhao parcial de bens', 'casado(a) com separacao de bens', 'divorciado', 'separado judicialmente', 'viuvo(a)', 'outros'])
    idade = st.sidebar.text_input("idade")
    cheque_sem_fundo = st.sidebar.selectbox("Já passou cheque sem fundo?", ['S', 'N'])
    valor_emprestimo = st.sidebar.text_input("Valor do Emprestimo")

    data= {
        'id': id,
        'dependentes': dependentes,
        'estado_civil': estado_civil,
        'idade': idade,
        'cheque_sem_fundo': cheque_sem_fundo,
        'valor_emprestimo': valor_emprestimo
    }

    # container para o gráfico SHAP
    container_grafico = st.container()

    if st.sidebar.button('Predict'):
        filtered_data = {col: data[col] for col in config['model_columns']['x_columns']}
        response = requests.post(URL_PREDICT, json=filtered_data)
        # response = requests.post(config['url_request']['path_predict_docker_compose'], json=filtered_data)
        
        prediction = str(response.json()['resultado']) + ' / probabilidade de Inadimplencia = ' + str(response.json()['probabilidade'])
        st.sidebar.success(f'The prediction from model: {prediction}')

        shap_output = response.json()['shap_output']
        values = np.array(shap_output['values'][0])
        base_value = shap_output['base_values']
        feature_values = np.array(shap_output['data'][0])

        feature_names_values = response.json()['feature_names_values']
        feature_names = [name for name, value in sorted(feature_names_values.items(),key=lambda item: feature_values.tolist().index(item[1]))]

        explainer = shap.Explanation(values=values, base_values=base_value, data=feature_values, feature_names=feature_names)
        
        with container_grafico:
            st.subheader("Grafico SHAP")
            fig, ax = plt.subplots()
            shap.plots.waterfall(explainer, show=False)
            st.pyplot(fig)

        
    # container para a tebela de previsões
    container_tabela = st.container()
    uploaded_file = st.sidebar.file_uploader("Upload de CSV com informações clientes", key='2')
    if uploaded_file is not None:
        df_csv = pd.read_csv(uploaded_file, sep=',')
        json_df = json.dumps(df_csv[config['model_columns']['x_columns']].to_dict(orient='records'))
        
        # response = requests.post(config['url_request']['path_predict_file_docker_compose'], json=json_df)
        response = requests.post(URL_PREDICT_FILE, json=json_df)

        df_csv['predict'] = json.loads(response.text)

        df_csv[['predict', 'prob_inadimplente']] = df_csv['predict'].str.split('-', expand=True)

        with container_tabela:
            st.subheader("Tabela de Previsões")
            st.dataframe(df_csv, 2000, 1000)
            st.download_button(label='Download CSV', data=df_csv.to_csv(index=False), file_name='predic.csv', mime='text/csv')

if __name__ == '__main__':
    run()