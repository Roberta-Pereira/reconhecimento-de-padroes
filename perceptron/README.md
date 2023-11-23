<h1 align='center'> Perceptron Simples </h1>

O perceptron é um algoritmo básico de aprendizado de máquina que permite classificar os dados em duas categorias.

### Funcionamento do algoritmo
O perceptron é um modelo de neurônio artificial que recebe um conjunto de entradas, realiza um cálculo ponderado dessas entradas e aplica uma função de ativação para determinar a saída. O objetivo é aprender a fazer previsões corretas, ajustando os pesos das entradas para minimizar o erro.

### Passos para treinamento do algortimo
1. Inicializar pesos com valores aleatórios.
2. Calcula a soma ponderada para cada variavel de entrada de uma amostra
3. Aplica a função de ativação
4. Compara a saída do algortimo com a saída desejada (classificação real) e calcula o erro
5. Ajusta os pesos com base na taxa de apendizado a fim de diminuir o erro
6. Repete as etapas 2 a 5 para cada amostra até que a condição de parada seja satisfeita (para esse projeoto: erro maior que tolerancia pre determinada ou número de epocas atingido)

  <img src="https://github.com/Roberta-Pereira/reconhecimento-de-padroes/assets/50178585/f284706c-9d54-4a4f-9fe7-37efd8fb8df8.png" width="380" height="300">
 
  Fonte: Flauzino, Silva, Spatti (2019, p. 65)

### Dados de exemplo
O algoritmo desenvolvido foi aplicado na base de dados Breast Cancer Wisconsin (Original) referente a 699 amostras de indivíduos com câncer de mama. Os dados foram classificados em malingno e benigno. Cada linha da tabela representa uma amostra de exemplo e cada coluna uma  variável de entrada para o problema, sendo a última coluna a classificação real da amostra.
