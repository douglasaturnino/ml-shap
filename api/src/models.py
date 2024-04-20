from typing import List, Union

from pydantic import BaseModel


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
