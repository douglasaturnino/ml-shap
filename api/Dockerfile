FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

ADD requirements.txt /app

RUN pip install -r requirements.txt

ADD . /app

EXPOSE 8001

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]