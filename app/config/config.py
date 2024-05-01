import os
from typing import Any, Dict

import yaml


def import_config(path_yaml: str) -> Dict[str, Any]:
    """
    Função para importar as configurações de um arquivo YAML.

    Args:
    - path_yaml (str): O caminho para o arquivo YAML de configuração.

    Returns:
    - config (Dict[str, Any]): Um dicionário contendo as configurações do arquivo YAML.
    """
    with open(path_yaml) as config_file:
        config = yaml.safe_load(config_file)
    return config


path = os.path.dirname(os.path.abspath(__file__))
config = import_config(path + "/config.yaml")
