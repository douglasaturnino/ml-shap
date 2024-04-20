import json

import pandas as pd
import uvicorn
from config.config import config
from fastapi import FastAPI, Request
from src.models import Data
from src.process import DataProcess

app = FastAPI()


@app.post(config["api"]["path_predict"])
def predict(data: Data):
    try:
        data = pd.DataFrame.from_dict([data.model_dump()])
        return DataProcess().process_data(data)
    except Exception as e:
        return {"error": str(e)}


@app.post(config["api"]["path_predict_file"])
async def predict_file(request: Request):
    try:
        data = await request.json()
        data = json.loads(data)
        data = pd.json_normalize(data)

        results = DataProcess().process_data(data)

        return results
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host=config["api"]["host"], port=config["api"]["porta"])
