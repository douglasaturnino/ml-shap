Essa sessão mostra como utilizar a aplicação no streamlit e algumas informações sobre a aplicação.

Existe duas formas de utlizar a aplicação uma preenchendo as informações do cliente na barra lateral 
para verificar a adimplencia individualmente e outra fazendo o upload de um arquivo csv, vamos mostrar como utilizar as duas formas abaixo.

## Informações necessarias

As informações necessaria para funcionar a aplicação são:

 - **dependentes**
      - **Descrição**: Informa se o cliente possui dependentes.
      - **Tipo de Dados**: String.
      - **Valores Permitidos**: ['S', 'N'].

 - **estado civil**
      - **Descrição**: O estado civil do cliente.
      - **Tipo de Dados**: String.
      - **Valores Permitidos**: ['solteiro', 'casado(a) com comunhao de bens', 'casado(a) com comunhao parcial de bens', 'casado(a) com separacao de bens', 'divorciado', 'separado judicialmente', 'viuvo(a)', 'outros'].

 - **idade**
      - **Descrição**: A idade do cliente na data do pedido de empréstimo.
      - **Tipo de Dados**: Integer.
      - **Valores Permitidos**: > 18

 - **cheque sem fundo**
      - **Descrição**: Informa se houve cheque sem fundo.
      - **Tipo de Dados**: String.
      - **Valores Permitidos**: ['S', 'N'].

 - **valor emprestimo**
      - **Descrição**: Valor do empréstimo solicitado pelo cliente.
      - **Tipo de Dados**: Float.
      - **Valores Permitidos**: > 0.0


## Preenchendo as Informações

Após inserir os valores e acionar o botão de previsão, os resultados da predição serão exibidos abaixo do botão em verde, indicando se o indivíduo é considerado adimplente ou inadimplente, juntamente com a probabilidade de inadimplência. É importante destacar que consideramos como inadimplentes aqueles com uma probabilidade de inadimplência superior a 75%.

Além disso, será apresentado um gráfico SHAP, o qual demonstrará a importância das variáveis para a decisão da predição, ordenadas da mais relevante para a menos relevante. Para uma compreensão mais aprofundada do gráfico SHAP, recomendamos a leitura do [artigo](https://www.linkedin.com/pulse/interpreta%2525C3%2525A7%2525C3%2525A3o-de-modelos-machine-learning-usando-shap-saturnino-ntjnf/?trackingId=4o9T47%2FZRfeYCPimVHj6JA%3D%3D).

<video src="https://github.com/douglasaturnino/ml-shap/assets/95532957/2e6250bd-4fd2-4cee-a087-c547b25e9885" width="640" height="360" controls></video>

## Fazendo upload de um arquivo csv

Para realizar o upload do arquivo, basta clicar em "Browse files" e selecionar o arquivo CSV desejado. Um exemplo de conjunto de dados pode ser encontrado no repositório [aqui](https://github.com/douglasaturnino/ml-shap/blob/main/data/test_data/test_data.csv).
Assim como as informações inseridas manualmente, as colunas necessárias são as mesmas na seção de [informações necessárias](#informacoes-necessarias).

<video src="https://github.com/douglasaturnino/ml-shap/assets/95532957/9c15c236-1508-480d-8122-aa6ad951bfc5" width="640" height="360" controls></video>
