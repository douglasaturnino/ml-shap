from typing import List, Union

from pydantic import BaseModel
"""
Definições de modelos Pydantic para representar a saída de um modelo SHAP e os dados de entrada.

Classes:
    - ShapOutput: Uma classe Pydantic para representar a saída de um modelo SHAP, incluindo os valores SHAP,
      valores base e os dados de entrada.
    - Data: Uma classe Pydantic para representar os dados de entrada do modelo.

Exemplo de uso:
    # Criando uma instância do modelo ShapOutput
    shap_output = ShapOutput(values=[[0.1, 0.2], [0.3, 0.4]],
                             base_values=[0.349, 0.349],
                             data=[[0.099, 0.275, 0.472, 30, 1000]])

    # Criando uma instância do modelo Data
    data_input = Data(dependentes='S', estado_civil='casado', idade=30,
                      cheque_sem_fundo='N', valor_emprestimo=1000.0)
"""

class ShapOutput(BaseModel):
    values: List[List[float]]
    base_values: Union[List[float], float]
    data: List[List[float]]


class Data(BaseModel):
    dependentes: str
    estado_civil: str
    idade: int
    cheque_sem_fundo: str
    valor_emprestimo: float
