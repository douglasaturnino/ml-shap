FROM python:3.10

WORKDIR /app

ADD requirements.txt /app

RUN pip3 install -r requirements.txt 

ADD . /app

EXPOSE 8501

CMD streamlit run __main__.py