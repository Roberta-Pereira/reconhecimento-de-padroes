import numpy as np
import pandas as pd
import random as rd
import math

def le_dados():
    df = pd.read_csv("breast-cancer-wisconsin.csv", 
                    sep=";", 
                    names=['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',  'Marginal Adhesion',
                            'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'])
    
    #Tratamento de dados
    df.drop(df.loc[df["Bare Nuclei"]=='?'].index, inplace=True) #elimina dados faltantes
    df["Class"] = df["Class"].replace([2,4], [1,0]) #rotula as classes  
    df = df.astype(int)

    #Armazenamento de dados
    classe_0 = df[df['Class'] == 0].to_numpy()
    classe_1 = df[df['Class'] == 1].to_numpy()
    return [classe_0, classe_1]
    
def sub_conjuntos(dados, folds):

    dim = dados.shape
    N = dim[0] #Número de amostras
    n = dim[1] #Dimensão de entrada

    if (N % folds) != 0:
        zeros = folds - (N % folds) 
        dados = np.r_[dados,np.zeros([zeros,n])] #Para completar o tamanho do subconjunto, em caso de divisão não exata, é inserido vetores com valores 0
    
    sub_conjuntos = np.vsplit(dados,folds) #Separa as amostras em subconjuntos do tamanho do fold

    return sub_conjuntos

def selec_treino(folds,i):
    treino =  np.delete(folds,i,0)  #Exclui do conjunto de dados as amostras utilizadas para teste
    amostras = treino[:][:,:,0:9]   #Pega os dados referentes as amostras (posição 0 a 8)
    amostras = amostras.reshape(amostras.shape[1]*amostras.shape[0], amostras.shape[2]) #Transforma em uma matriz, em que cada linha é a amostra a ser analisada
    
    rotulo_real = treino[:][:,:,[9]] #Pega os dados referentes as saidas (posição 9)
    rotulo_real = rotulo_real.reshape(rotulo_real.shape[1]*rotulo_real.shape[0], rotulo_real.shape[2])

    return amostras, rotulo_real

def treinar(X, Y, eta_aprendizagem, tolerancia, max_epocas, par):
    funcao_ativacao = lambda x: 1 if x >=0 else 0

    if par==1:
        X = np.c_[-1*np.ones((X.shape[0],1)), X] #Acrescenta a linha do bias

    dim = X.shape
    N = dim[0] #Número de amostras
    n = dim[1] #Dimensão de entrada

    n_epocas = 0
    erro_epoca = tolerancia+1
    vet_erro = np.zeros((1,max_epocas))    

    w = np.random.uniform(-0.5,0.5, size=n)

    while (erro_epoca > tolerancia) and (n_epocas < max_epocas):
        ei2 = 0
        xseq = np.random.permutation(N) #Gera sequencia embaralhada das amostras
        for amostra_i in range(N):
            iseq = xseq[amostra_i]
            soma_ponderada = np.dot(X[iseq],w)
            saida_algoritmo = funcao_ativacao(soma_ponderada)
            erro_amostra = (Y[iseq] - saida_algoritmo)

            w = w + (eta_aprendizagem*erro_amostra*X[iseq,:])

            # Acumula erro por epoca
            ei2 = ei2 + erro_amostra**2

        vet_erro[0][n_epocas] = ei2/N
        erro_epoca = vet_erro[0][n_epocas]
        n_epocas+=1
       
    classe_0 = round(((N-np.count_nonzero(saida_algoritmo))/N)*100,2) #Porcentagem de amostras da classe 0
    classe_1 = round((np.count_nonzero(saida_algoritmo)/N)*100,2) #Porcentagem de amostras da classe 1

    return w

def testar(X, Y, w, par):
    funcao_ativacao = lambda x: 1 if x >=0 else 0

    if par==1:
        X = np.c_[-1*np.ones((X.shape[0],1)), X]

    dim = X.shape
    N = dim[0] #Número de amostras
    n = dim[1] #Dimensão de entrada

    erro = np.zeros_like(Y)
    saida = np.zeros_like(Y)

    acertos = 0

    for i in range(N):
        saida[i] = funcao_ativacao((np.dot(X[i],w)))
        
        erro[i] = (Y[i] - saida[i])

        if erro[i] == 0:
            acertos = acertos + 1

    acuracia = acertos/N
    desvio_padrao = np.std(erro)
    erro_medio = np.mean(erro)
    
    classe_0 = round(((N-np.count_nonzero(saida))/N)*100,2) #Porcentagem de amostras da classe 0
    classe_1 = round((np.count_nonzero(saida)/N)*100,2) #Porcentagem de amostras da classe 1

    return saida, acuracia, desvio_padrao, erro_medio, (classe_0,classe_1)

def estatisticas(dados):   
    df = pd.DataFrame(dados, columns=['Acuracia (%)', 'Desvio Padrão', 'Erro médio', '% Classe 0 (Maligno)', '% Classe 1 (Benigno)'])
    df["Acuracia (%)"] = (df["Acuracia (%)"] * 100)
    df.drop(df.loc[df["Acuracia (%)"] <= df["Acuracia (%)"].min()].index, inplace=True) #elimina dados faltantes
    print(df, "\n")
        
    minimo = [df['Acuracia (%)'].min(), df['Desvio Padrão'].min(), df['Erro médio'].min()]
    print("MÍNIMO: \n")
    print("Acuracia (%):", minimo[0], 
        "\nDesvio Padrão:", minimo[1],
        "\nErro médio:", minimo[2],"\n")
    media = [df['Acuracia (%)'].mean(), df['Desvio Padrão'].mean(), df['Erro médio'].mean()]
    print("\nMÉDIA: \n")
    print("Acuracia (%):", media[0], 
        "\nDesvio Padrão:", media[1],
        "\nErro médio:", media[2],"\n")
    maximo = [df['Acuracia (%)'].max(), df['Desvio Padrão'].max(), df['Erro médio'].max()]
    print("\nMÁXIMO: \n")
    print("Acuracia (%):", maximo[0], 
        "\nDesvio Padrão:", maximo[1],
        "\nErro médio:", maximo[2],"\n")

if __name__ == "__main__":
    classe_0, classe_1 = le_dados()

    #Exclui colunas de ID 
    classe_0 = np.delete(classe_0,0,1) 
    classe_1 = np.delete(classe_1,0,1)

    eta_aprendizagem = 0.5 #Taxa de aprendizado
    experimentos = 20
    tolerancia = 0.0001
    max_epocas = 1000

    fold_0 = sub_conjuntos(classe_0,experimentos)
    fold_1 = sub_conjuntos(classe_1,experimentos)

    i=0
    resultado = []

    estatistica_teste = []

    while i<experimentos:

        x0_treino, y0_treino = selec_treino(fold_0,i)
        x1_treino, y1_treino = selec_treino(fold_1,i)

        x_treino = np.concatenate((x0_treino,x1_treino))
        y_treino = np.concatenate((y0_treino,y1_treino))

        w = treinar(x_treino, y_treino, eta_aprendizagem, tolerancia, max_epocas, 1)

        x_teste = np.concatenate((fold_0[i][:,0:9],fold_1[i][:,0:9]))
        y_teste = np.concatenate((fold_0[i][:,9],fold_1[i][:,9]))

        resultado = testar(x_teste, y_teste, w, 1)
        estatistica_teste.append([resultado[1], resultado[2], resultado[3], resultado[4][0], resultado[4][1]])
    
        i=i+1
    estatisticas(estatistica_teste)