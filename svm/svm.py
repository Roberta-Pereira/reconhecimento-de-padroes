import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV,  StratifiedShuffleSplit
from sklearn.metrics import accuracy_score


def read_data():
    df = pd.read_csv("glass.csv", 
                    sep=",", 
                    names=['Id', 'RI', 'Na', 'Mg',  'Al',
                            'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type'])
    
    #Tratamento de dados
    df.astype(int)

    X = df[['RI', 'Na', 'Mg',  'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']].to_numpy()
    y = df['Type'].to_numpy()

    return X, y

#Separa o conjunto de dados para validacao cruzada
def select_data(X, y):
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    skf.get_n_splits(X, y)
    return skf

def svm_data(X_train, y_train, X_test, c, g):
    model = svm.SVC(kernel='rbf', C=c, gamma=g).fit(X_train,y_train)
    predict = model.predict(X_test)

    return model, predict

#Escolhe os parametros para o modelo
def svm_parameters(X, y):
    C = [0.01, 0.1, 1, 10, 100]
    gamma = [ 0.01, 0.1, 1, 10, 100]

    parameters = {'kernel':['rbf'], 'C': C, 'gamma': gamma }
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(X, y)
    clf.cv_results_

    scores = clf.cv_results_["mean_test_score"].reshape(len(C), len(gamma))

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(
        scores,
        interpolation="nearest",
        cmap=plt.cm.hot,
    )
    plt.xlabel("gamma")
    plt.ylabel("C")
    plt.colorbar()
    plt.xticks(np.arange(len(gamma)), gamma, rotation=45)
    plt.yticks(np.arange(len(C)), C)
    plt.title("Validation accuracy")
    plt.show()

    return clf.best_params_

def statistics(dataframe):
    df = pd.DataFrame(dataframe, columns=['Experimento', 'Acuracia (%)', 'Classe 1', 'Classe 2', 'Classe 3', 'Classe 4', 'Classe 5', 'Classe 6', 'Classe 7'])
    std = df["Acuracia (%)"].std()
    df["Acuracia (%)"] = (df["Acuracia (%)"] * 100)
    mean_acuracy = df['Acuracia (%)'].mean()

    total = df[['Classe 1','Classe 2', 'Classe 3', 'Classe 4', 'Classe 5', 'Classe 6', 'Classe 7']].sum()
    porcent = total.div(214) * 100
    print(df)
    print(total, porcent)

    print("\nA acurácia média encontrada após 10 experimentos foi de {0:.2f}% e o desvio padrão: {1:.2f} \n\n".format(mean_acuracy, std))

    return df

if __name__ == "__main__":

    fold = 0
    acuracy = []
    std = []
    results = []
    dataframe = []
    aux = []
   
    X, y = read_data()
    folds = select_data(X,y)
    parameter = svm_parameters(X, y)
    c = parameter['C']
    gamma = parameter['gamma']

    for train_index, test_index in folds.split(X, y):
        data_class = dict.fromkeys(np.unique(y) , 0)
        data_class[4] = data_class.get(4, 0)

        fold += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model, predict = svm_data(X_train, y_train, X_test, c, gamma)
        ac = accuracy_score(y_test,predict)
        acuracy.append(ac)

        results.append([y_test, predict, acuracy])

        for i in predict:
            data_class[i] = data_class.get(i, 0) + 1
        
        aux = [fold, ac]

        for j in range(1,8):
            aux.append(data_class.get(j))
        
        data_class.clear()
        dataframe.append(aux)

    statistics(dataframe)