import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from requests import request
import uvicorn
from pydantic import BaseModel
import json
import pickle
import yaml

from typing import List

def import_config(path_yaml):
    with open(path_yaml) as config_file:
        config = yaml.safe_load(config_file)
    return config

config = import_config("config.yaml")

# Create the app
app = FastAPI()

class ShapOutput(BaseModel):
    values: List[List[float]]
    base_values: float
    data: List[List[float]]


class Data(BaseModel):
    dependentes: str
    estado_civil: str
    idade: int
    cheque_sem_fundo: str
    valor_emprestimo: float

# Local trained Pipeline
xgb_model = pickle.load(open(config["model"]["ml_path"], "rb"))
th = config["model"]["threshold"]

shap_model = pickle.load(open(config["model"]["shap_path"], "rb"))

# Define predict function
@app.post(config["api"]["path_predict"])
def predict(data: Data):
    try:
        data_dict = data.dict()
        data_df = pd.DataFrame.from_dict([data_dict])
        data_df.columns = config["model_columns"]["x_columns"]

        predictions = (xgb_model.predict_proba(data_df)[:, 1] > th).astype(int)
        probabilidade = round(xgb_model.predict_proba(data_df)[:,1].astype(float)[0],2)
        resultado = 'Adimplete' if predictions == 0 else 'Inadimplente'

        data_pp1 = xgb_model.named_steps["rare_encoder"].transform(data_df)
        data_pp2 = xgb_model.named_steps["categorical_encoder"].transform(data_pp1)
        shap_values = shap_model(data_pp2)
        
        # Convert Numpy array to lists
        values_list =  shap_values.values.tolist() if isinstance(shap_values.values, np.ndarray) else shap_values.values
        data_list = shap_values.data.tolist() if isinstance(shap_values.data, np.ndarray) else shap_values.data
        shap_client = ShapOutput(values=values_list, base_values=shap_values.base_values, data=data_list)

        # Return simple dictionary
        return {
                "resultado": resultado,
                "probabilidade": probabilidade,
                "shap_output": shap_client.dict(),
                "feature_names_values": data_pp2.to_dict(orient="records")[0],
            }
    except Exception as e:
        return {"error": str(e)}

@app.post(config["api"]["path_predict_file"])
async def predict_file(request: Request):
    try:
        data = await request.json()
        data = await request.json()
        data = json.loads(data)
        data = pd.json_normalize(data)
        data.columns = config["model_columns"]["x_columns"]

        prob = xgb_model.predict_proba(data)[:, 1]
        data['predict'] = (xgb_model.predict_proba(data)[:, 1] > th).astype(int)

        data["predict"] = data["predict"].apply(lambda x: "Adimplente" if x == 0 else "Inadimplente" )
        
        data["prob_inadimplencia"] = prob

        data["predict"] = data.apply( lambda x: x["predict"] + "-" + str(round(x["prob_inadimplencia"], 2)), axis=1)
        return data['predict'].tolist()
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host=config["api"]["host"], port=config["api"]["porta"])