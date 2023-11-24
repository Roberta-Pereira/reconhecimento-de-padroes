<h1 align='center'> K-MEANS </h1>

O K-Means é um algoritmo de clusterização que agrupa instâncias de dados em K clusters, onde K é um número de agrupamentos desejados. É um algortimo de aprendizagem não supervisionada, pois os dados não estão rotulados.

O objetivo é detectar padrões agrupando os dados de maneira que os pontos que pertencerm ao mesmo cluster estejam o mais próximo possível e pontos em clusters distintos estejam o mais distante possível.

### Funcionamento do algoritmo

- Inicialização: seleciona aleatoriamente K instâncias como centroides iniciais ou de maneira determinística.
- Atribuição: cada amostra é atribuída ao cluster mais próximo, com base na distância euclidiana entre o ponto e o centro do cluster.
- Atualização: os centroides são recalculados como a média das instâncias atribuídas a cada cluster.
- Repetição: os passos de atribuição e atualização são repetidos até que não ocorram mudanças na atribuição dos clusters ou até que seja atingido o número máximo de iterações.

#### Parâmetros
- *K*: o número de clusters desejados.
- Critério de parada: pode ser o número máximo de iterações ou uma tolerância para alterações nas atribuições dos clusters.

### Passos para treinamento do algortimo
1. Definir o número k de clusters;
2. Escolher aleatoriamente k pontos de treinamento para representar os centros dos clusters;
3. Calcular a distância de todos os demais pontos de treinamento ao centro dos clusters e alocar esses pontos ao cluster mais próximo;
4. Após alocar todos os pontos, o novo centro de cada cluster é calculado pela média dos seus pontos;
5. Repetir os postos 3 e 4 até que os novos agrupamentos não difira do anterior.

### Dados de exemplo
Para execução desse algortimo foi criado um conjunto de dados com quatro distribuições
Gaussianas bidimensionais para representar as amostras. 
O algoritmo foi aplicado em conjuntos de dados utilizando diferentes valores para o desvio-padrão das gaussianas e diferetnes valores para k.

Métrica utilizada: distancia euclidiana
