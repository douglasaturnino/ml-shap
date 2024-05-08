<div align='center'>
<img src="https://github.com/douglasaturnino/ml-shap/assets/95532957/6e0a8482-fde0-43b4-b144-4947ae360591"  width=700px/>
</div>

Este é um projeto fictício. A empresa, o contexto e as perguntas de negócios não são reais. Este portfólio está seguindo as recomendações da Comunidade DS.
Ao enviar uma mensagem para a API, a resposta pode haver uma lentidão para aparecer, pois, o Render depois de um tempo sem uso desliga a aplicação.

O Banco UniiFance, uma instituição financeira líder e confiável, destaca-se no
mercado pela sua dedicação em fornecer soluções de crédito acessíveis e sob
medida para empresários do setor comercial. Com foco em empréstimos
flexíveis e acessíveis, nossa equipe altamente qualificada trabalha em estreita
colaboração com os clientes para atender às suas necessidades financeiras
específicas.

Atualmente o banco está passando por uma revisão em como ele empresta o
dinheiro para os seus clientes, assim o objetivo é criar processos inteligentes
para a previsão de que alguém pode vim a passar por dificuldades financeiras
nos próximos dois anos.

## 1 Problema de negócios
### 1.1 Problema

Um Banco Brasileiro contratou você como consultor em ciência de dados. O diretor da área de empréstimos para pessoa física, percebeu a diminuição do ROI(Return Over Investment Retorno sobre Investimento) devido a alta taxa de inadimplência.

### 1.2 Motivação
O diretor da área de empréstimos para pessoa física está com uma alta taxa percentual de inadimplência, afetando diretamente o ROI da empresa.

### 1.3 Demandas de negócio

- Qual perfil de um cliente adimplente?
- Qual variável que mais impacta para o cliente ser inadimplente?
- Qual a contribuição individual de cada variável para cada cliente ser inadimplente?

## 2 Premissas de negócio
- Todos os produtos de dados entregues devem ser acessíveis via internet.
- O planejamento da solução será validado com os times de negócio, visando garantir que as soluções desenvolvidas sejam úteis na sua tomada de decisão.
- Após responder as perguntas, o Diretor gostaria de construir uma aplicação que retorne a probabilidade do cliente ser inadimplente para que os analistas de empréstimo possam simular alguns cenários de empréstimos e tomar melhores decisões.

As variaveis do dataset original são:

| Variavel | Definição |
--------- | ---------
| id | Um Id que representa uma linha dentro do conjunto de dados |
| escolaridade | Nivel de escolaridade do cliente|
| renda_mensal | Renda bruta mensal do cliente |
| dependentes | Informa se o cliente possui dependentes  |
| estado_civil | O estado civil do cliente |
| idade | A idade do cliente na data do pedido de emprestimo |
| conta_poupanca | Informa se a conta é poupança |
| conta_salario | Informa se a conta é salario |
| qtd_fonte_renda | A quantidade de fonte de renda do cliente |
| cheque_sem_fundo | Informa se o cliente já passou cheque sem fundo |
| conta_conjunta | Informa se é uma conta conjunta |
| valor_conta_corrente | A quantidade de dinheiro na conta corrente do cliente |
| valor_conta_poupanca | A quantidade de dineiro na conta poupança do cliente |
| valor_emprestimo | A quantia que o cliente pediu o emprestimo
| multa | A multa aplica no emprestimo |
| juros | O juros aplicado no emprestimo |
| valor_emprestimo_atualizado | A soma do valor do emprestimo + multa + juros |
| pago | Informa se o cliente pagou o emprestimo |
| genero | Informa o genero do cliente |
| data | Data que o dataset foi fornecido |
| estado | Estado do cliente |

## 3 Planejamento da solução
### 3.1 Produto final
O que será entregue efetivamente?

- A uma API que recebe os dados do cliente e retorna a probabilidade de Inadimplencia.
- Um painel para verificar se o cliente vai ser adimplente.

### 3.2 Ferramentas
Quais ferramentas serão usadas no processo?

- Python;
- Visual Studio code;
- Jupyter Notebook;
- Git, Github;
- Streamlit;
- Docker, Docker compose;
- Cloud Render.

### 3.3 Metodo SAPE

#### Saida

- Aplicativo que recebe os dados de clientes novos e retorna a probabilidade
de Inadimplência junto com o impacto individual de cada variável para
tomada de decisão.
- Relatório com todas as resposta das perguntas feitas pelo Diretor da área de
empréstimos.

#### Processo

- Baixar os dados e ler os dados

- Executar o processo de limpeza de dados como:
   - renomear colunas
   - checagem de valores NA
   - imputação de valores NA
   - renomear categorias
   - mudar de tipo de variável, caso necessário

- Estatística descritiva com visualização

- Criação de hipóteses e validação das mesmas

- Análise exploratória
   - Univariada
   - Bivariada

- Preparação dos dados para treinamento de modelos de Machine Learning (feature engineering)

- Seleção das variáveis mais importantes
- Treinamento de algoritmos de ML
- Tunagem de hyperparâmetros e cross validation
- Interpretação dos modelos utilizando SHAP
- Comparação da performance dos modelos
- Escolha do modelo final
- Uso do modelo final para fazer predições de novos dados em um aplicativo na web

#### Entrada

- Tabela de dados de clientes em .csv fornecidad pela empresa.

## 4. Os 3 principais insights dos dados

#### 1. Clientes com Dependentes são em média 15% mais ADIMPLENTES.

* **Verdadeiro** Clientes com dependentes são em média 18% mais ADIMPLENTES.

![alt text](https://github.com/douglasaturnino/ml-shap/assets/95532957/2aae8624-7ef8-4843-be22-110c333afc69)

#### 2. Clientes que pegaram empréstimos de mais de RS$20000 tem tendência a ser INADIMPLENTES.

* **Verdadeiro** Clientes que pegaram empréstimos de mais de RS$20000 tem tendência a ser INADIMPLENTES.

![alt text](https://github.com/douglasaturnino/ml-shap/assets/95532957/e084ec1f-e96b-46d7-a170-788c629e41d8)


### 3. Clientes que são solteiros tem uma taxa percentual maior de INADIMPLÊNCIA que os demais.

* **FALSO** Clientes que são solteiros tem um taxa de 89% de ser ADIMPLENTES.

![alt text](https://github.com/douglasaturnino/ml-shap/assets/95532957/77cefe78-0edd-4a9b-adcd-0e38e47c4837)

## 5. Resultados para o negócio

De acordo com os critérios definidos, foi feita uma previsão de adimplencia. Como resultado para o negócio foram criados:

* Uma API onde será feita a previsão.

* Um dashboard feito no streamlit para preve a adimplencia do cliente.

* Uma documentação para o entendimento do projeto.

## 6. Conclusão

O objetivo foi alcançado, dado que o produto de dados foram gerados com sucesso. O funcionario pode utilizar a ferramenta criado para fazer a previsão para a tomada de dicisão de emprestimo.

O dashboard pode ser acessado por [aqui](https://ml-shap.streamlit.app/)

* Foi construida uma documentação com o problema a ser resolvido e detalhes das classes [aqui](https://douglasaturnino.github.io/ml-shap/).

## 7. Próximos passos

* Aprimorar a apresentação do painel, pois sua disposição atual pode não ser intuitiva.
* Exibir os resultados acima do painel, em vez de destacá-los dentro do menu em verde.
