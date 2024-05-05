## Executando a aplicação

Há duas maneiras de executar a aplicação: usando Docker Compose ou executando diretamente os arquivos \_\_main\_\_.py entro das pastas api e app.

### Utilizando Docker Compose

Antes de tudo, certifique-se de ter o Docker instalado em sua máquina. Você pode instalá-lo seguindo a documentação oficial do [Docker Desktop](https://docs.docker.com/desktop/).

Com o Docker instalado, basta executar o comando `docker compose up` no terminal, dentro da pasta raiz do projeto. Isso construirá a API e a aplicação no Streamlit. Em seguida, em um navegador de sua preferência, acesse `http://localhost:8501` para abrir a aplicação no Streamlit. 

Para encerrar a aplicação, pressione `Ctrl + C` no terminal ou feche-o.

Para saber como utilizar, consulte a seção [Modo de uso](utilizacao.md).

### Executando o arquivo .py

Para executar a aplicação, crie um ambiente virtual usando sua ferramenta preferida. A ferramenta utilizada neste caso foi o Poetry, com a versão do Python sendo 3.10.14.

Após a instalação do Poetry, na raiz do projeto, execute o comando `poetry install` para instalar todas as dependências do projeto.

Em um terminal, execute o comando `python api/__main_.py` ou `python api`. Isso iniciará o FastAPI. Mantenha este terminal aberto e, em um novo terminal, execute o comando `streamlit run app/__main__.py`. Se a página não abrir automaticamente no seu navegador, acesse `http://localhost:8501`. 

Para encerrar a aplicação, pressione `Ctrl + C` no terminal ou feche-o. 

Para saber como utilizar, consulte a seção [Modo de uso](utilizacao.md).