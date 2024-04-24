
class DataProcess:
    """
    Class for processing data and generating predictions and explanations.

    Attributes:
        model (sklearn.base.BaseEstimator): Machine learning model for prediction.
        shap_model (callable): SHAP model for generating feature importance explanations.
        th (float): Threshold for classification.
        columns (list): List of column names used in the model.

    Methods:
        __init__: Initializes DataProcess with model, SHAP model, threshold, and columns.
        rename_columns: Renames columns of the input data.
        predict_and_prepare_results: Predicts classes and prepares results.
        transform_data: Transforms input data for model prediction.
        calculate_shap_values: Calculates SHAP values for input data.
        prepare_output: Prepares output data containing predictions and explanations.
        process_data: Processes input data to generate predictions and explanations.
    """

    def __init__(self) -> None:
        """
        Initializes DataProcess with model, SHAP model, threshold, and columns.
        """
        self.model = pickle.load(open(config["model"]["ml_path"], "rb"))
        self.shap_model = pickle.load(open(config["model"]["shap_path"], "rb"))
        self.th = config["model"]["threshold"]
        self.columns = config["model_columns"]["x_columns"]

    def rename_columns(self, data):
        """
        Renames columns of the input data.

        Args:
            data (pandas.DataFrame): Input data.

        Returns:
            pandas.DataFrame: Data with renamed columns.
        """
        data_df = data.copy()
        data_df.columns = self.columns
        return data_df

    def predict_and_prepare_results(self, data_df):
        """
        Predicts classes and prepares results.

        Args:
            data_df (pandas.DataFrame): Input data with renamed columns.

        Returns:
            pandas.DataFrame: Data with predicted classes and probabilities.
        """
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
        """
        Transforms input data for model prediction.

        Args:
            data (pandas.DataFrame): Input data.

        Returns:
            numpy.ndarray: Transformed data for model prediction.
        """
        data_pp1 = self.model.named_steps["rare_encoder"].transform(data)
        data_pp2 = self.model.named_steps["categorical_encoder"].transform(
            data_pp1
        )
        return data_pp2

    def calculate_shap_values(self, data_pp):
        """
        Calculates SHAP values for input data.

        Args:
            data_pp (numpy.ndarray): Transformed data for model prediction.

        Returns:
            ShapOutput: SHAP values and related information.
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

    def prepare_output(self, data_df, shap_client, data_pp):
        """
        Prepares output data containing predictions and explanations.

        Args:
            data_df (pandas.DataFrame): Input data with predicted classes and probabilities.
            shap_client (ShapOutput): SHAP values and related information.
            data_pp (numpy.ndarray): Transformed data for model prediction.

        Returns:
            dict or list: Output containing predictions and explanations.
        """
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
        """
        Processes input data to generate predictions and explanations.

        Args:
            data (pandas.DataFrame): Input data.

        Returns:
            dict or list: Output containing predictions and explanations.
        """
        data_df = self.rename_columns(data)
        data_pp = self.transform_data(data)
        data_df = self.predict_and_prepare_results(data_df)
        shap_client = self.calculate_shap_values(data_pp)
        return self.prepare_output(data_df, shap_client, data_pp)



class DataProcess:
    """
    Classe para processamento de dados e geração de previsões e explicações.

    Atributos:
        model (sklearn.base.BaseEstimator): Modelo de aprendizado de máquina para previsão.
        shap_model (callable): Modelo SHAP para geração de explicações de importância de recursos.
        th (float): Limiar para classificação.
        columns (list): Lista de nomes de colunas usados no modelo.

    Métodos:
        __init__: Inicializa DataProcess com modelo, modelo SHAP, limiar e colunas.
        rename_columns: Renomeia colunas dos dados de entrada.
        predict_and_prepare_results: Prevê classes e prepara resultados.
        transform_data: Transforma os dados de entrada para a previsão do modelo.
        calculate_shap_values: Calcula os valores SHAP para os dados de entrada.
        prepare_output: Prepara os dados de saída contendo previsões e explicações.
        process_data: Processa os dados de entrada para gerar previsões e explicações.
    """

    def __init__(self) -> None:
        """
        Inicializa DataProcess com modelo, modelo SHAP, limiar e colunas.
        """
        self.model = pickle.load(open(config["model"]["ml_path"], "rb"))
        self.shap_model = pickle.load(open(config["model"]["shap_path"], "rb"))
        self.th = config["model"]["threshold"]
        self.columns = config["model_columns"]["x_columns"]

    def rename_columns(self, data):
        """
        Renomeia colunas dos dados de entrada.

        Args:
            data (pandas.DataFrame): Dados de entrada.

        Retorna:
            pandas.DataFrame: Dados com colunas renomeadas.
        """
        data_df = data.copy()
        data_df.columns = self.columns
        return data_df

    def predict_and_prepare_results(self, data_df):
        """
        Prevê classes e prepara resultados.

        Args:
            data_df (pandas.DataFrame): Dados de entrada com colunas renomeadas.

        Retorna:
            pandas.DataFrame: Dados com classes previstas e probabilidades.
        """
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
        """
        Transforma os dados de entrada para a previsão do modelo.

        Args:
            data (pandas.DataFrame): Dados de entrada.

        Retorna:
            numpy.ndarray: Dados transformados para a previsão do modelo.
        """
        data_pp1 = self.model.named_steps["rare_encoder"].transform(data)
        data_pp2 = self.model.named_steps["categorical_encoder"].transform(
            data_pp1
        )
        return data_pp2

    def calculate_shap_values(self, data_pp):
        """
        Calcula os valores SHAP para os dados de entrada.

        Args:
            data_pp (numpy.ndarray): Dados transformados para a previsão do modelo.

        Retorna:
            ShapOutput: Valores SHAP e informações relacionadas.
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

    def prepare_output(self, data_df, shap_client, data_pp):
        """
        Prepara os dados de saída contendo previsões e explicações.

        Args:
            data_df (pandas.DataFrame): Dados de entrada com classes previstas e probabilidades.
            shap_client (ShapOutput): Valores SHAP e informações relacionadas.
            data_pp (numpy.ndarray): Dados transformados para a previsão do modelo.

        Retorna:
            dict ou list: Saída contendo previsões e explicações.
        """
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
        """
        Processa os dados de entrada para gerar previsões e explicações.

        Args:
            data (pandas.DataFrame): Dados de entrada.

        Retorna:
            dict ou list: Saída contendo previsões e explicações.
        """
        data_df = self.rename_columns(data)
        data_pp = self.transform_data(data)
        data_df = self.predict_and_prepare_results(data_df)
        shap_client = self.calculate_shap_values(data_pp)
        return self.prepare_output(data_df, shap_client, data_pp)
