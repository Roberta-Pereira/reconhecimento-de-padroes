import numpy as np
import matplotlib.pyplot as plt
import copy

def kmeans(k, dados):
  N = dados.shape[0] 
  n = dados.shape[1]

  ind = np.random.choice(N, k, replace=False)

  #Escolha dos centros dos clusters
  c = dados[ind]

  dist = []
  agrupamentos = []
  agrupamentos_ant = []

  for i in range(k): 
    agrupamentos.append([])
    agrupamentos_ant.append([-1])
   
  while(agrupamentos != agrupamentos_ant):
    agrupamentos_ant = copy.deepcopy(agrupamentos)
     
    for i in range(k): 
      agrupamentos[i].clear()

    #Calcula a distância dos pontos ao centro dos clusters
    for ponto in range(N):
      for centro in range(k):
          dist.append(np.around(np.linalg.norm(dados[ponto]-c[centro]),2))

      #Aloca o ponto ao cluster mais próximo
      cluster = dist.index(min(dist)) 
      agrupamentos[cluster].append(list(dados[ponto]))
      
      dist.clear()
      
    #Calcula os novos centros
    for i in range(k):
       c[i] = ponto_medio(agrupamentos[i])

  return agrupamentos

def ponto_medio(agrupamentos):
    c = np.around(np.mean(agrupamentos, axis=0),2)
    return c

def gaussianas(n, mean, sd):
    g1 = np.random.normal(mean,sd,size=(n,2)) + np.full((n,2), [2,2])
    g2 = np.random.normal(mean,sd,size=(n,2)) + np.full((n,2), [4,4])
    g3 = np.random.normal(mean,sd,size=(n,2)) + np.full((n,2), [2,4])
    g4 = np.random.normal(mean,sd,size=(n,2)) + np.full((n,2), [4,2])

    dados = np.concatenate((g1, g2, g3, g4), axis=0)
    dados = np.round(dados,2)

    return dados

def grafico_dados(dados, sd):   

    fig, sub = plt.subplots(1, 3, sharex=True, sharey=True)
    fig.suptitle('Conjunto de dados')
    
    for i in range(len(sd)):
      amostras = np.asarray(dados[i])
      x = np.asarray(amostras)[:,0]
      y = np.asarray(amostras)[:,1]

      sub[i].plot(x,y,'o', color='black', alpha=0.3)
      sub[i].title.set_text("sd = {}".format(sd[i]))   

    plt.show()

def grafico_resultado(clusters, k, sd):
        
    fig, sub = plt.subplots(1, 3, sharex=True, sharey=True)
    fig.suptitle('Resultado para K = {}'.format(k))

    cor = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red', 4:'black', 5:'tab:purple', 6:'tab:pink', 7:'tab:olive'}
    
    for i in range(len(sd)):
      dados = clusters[i]

      for j in range(k):
          x = np.asarray(dados[j])[:,0]
          y = np.asarray(dados[j])[:,1]

          sub[i].plot(x,y,'o', color=cor[j])
      sub[i].title.set_text("sd = {}".format(sd[i]))
                            
    plt.show()

if __name__ == "__main__":
  qtdAgrupamentos = (2, 4, 8)
  desvioPadrao = (0.3, 0.5, 0.7)
  clusters = []
  amostras = []
  gerarDados = True

  for k in qtdAgrupamentos:
      clusters.clear()
      for i, sd in enumerate(desvioPadrao):
        if gerarDados:
          dados = gaussianas(100, 0, sd)
          amostras.append(dados)
        else:
          dados = amostras[i]
        resultado = kmeans(k, dados)
        clusters.append(resultado)
      gerarDados = False
      grafico_resultado(clusters, k, desvioPadrao)

  grafico_dados(amostras, desvioPadrao)