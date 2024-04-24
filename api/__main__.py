import uvicorn
from config.config import config
from src.main import app

uvicorn.run(app, host=config["api"]["host"], port=config["api"]["porta"])
