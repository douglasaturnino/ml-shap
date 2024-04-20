import os

import yaml


def import_config(path_yaml):
    """
    Função para importar as configurações de um arquivo YAML.

    Parâmetros:
    - path_yaml (str): O caminho para o arquivo YAML de configuração.

    Retorna:
    - config (dict): Um dicionário contendo as configurações do arquivo YAML.
    """
    with open(path_yaml) as config_file:
        config = yaml.safe_load(config_file)
    return config


path = os.path.dirname(os.path.abspath(__file__))
config = import_config(path + "/config.yaml")
