import os
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from  sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow import keras
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding

def get_data(din):
    bf = pd.read_csv(din,sep='\t')
    genes = bf['Hugo_Symbol']
    cols = list(bf.columns)
    cols.remove('Entrez_Gene_Id')
    cols.remove('Hugo_Symbol')
    bf = bf[cols]
    bf = bf.transpose()
    bf.columns = genes
    return bf

def get_score_threshold(X, y):
    #Creo que el problema 1 es llamar a los datos, los estaba llamando de una forma pero no era así
    model = Sequential()
    model.add(Embedding(10000, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(Dense(16, activation = "relu"))
    model.add(Dense(1, activation = "sigmoid"))
    model.summary()

    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=["accuracy"])

    x_val = X_train [:10000]
    x2_train = X_train[10000:]

    y_val = y_train [:10000]
    y2_train = y_train[10000:]

    fitmod = model.fit(x2_train, y2_train, steps_per_epoch= 5, epochs = 30, batch_size = 2, validation_data = (x_val, y_val), verbose = 1)
    
    results = model.evaluate(X_test, y_test)
    import pdb; pdb.set_trace()
    return results

opts = [['breast','brca'], ['liver','lihc'], ['lung','luad'],['prostate','prad'],['stomach','stad'],['thyroid','thca']]
dirin = 'data/'

result = []

keep_info = .1
beta = .999
for opt in opts:
    din1 = dirin + opt[0] + '-rsem-fpkm-gtex.txt'
    din2 = dirin + opt[1] + '-rsem-fpkm-tcga.txt'
    din3 = dirin + opt[1] + '-rsem-fpkm-tcga-t.txt'
    if os.path.isfile(din1) and os.path.isfile(din2) and os.path.isfile(din3):
        normal1 = get_data(din1)
        normal2 = get_data(din2)
        abnormal = get_data(din3)      
        X_test = pd.concat( [normal2, abnormal] ).values
        y_test = np.zeros(len(X_test), dtype=int)
        y_test[len(normal2):] = 1
       # import pdb; pdb.set_trace()
        y_pred, X_test, y_test, modelo = get_score_threshold(X_test,y_test)

        Accuracy = metrics.accuracy_score(y_test, y_pred)
        print("El Accuracy es:", Accuracy)
        Precision = metrics.precision_score(y_test, y_pred)
        print("La precision es:", Precision)
        Recall = metrics.recall_score(y_test, y_pred)
        print("El recall es:", Recall)
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
        Specificity = tn / (tn+fp)

        ret = [Accuracy]
        ret = [round(e,3) for e in ret]
        ret = [opt[0], len(normal1), len(normal2), len(abnormal)] + [keep_info,beta] + ret
        print(ret[0],ret[-6:])     
        result += [ ret ]

        #ax=sns.boxplot(x="Pclass", y="Age",hue="Sex", data= df, linewidth=2.5)

        data_idx = np.arange(len(X_test))
        plt.subplots(figsize=(10, 10))
        plt.scatter(data_idx[y_test==1], y_test[y_test==1],s= 10, label='Real Tumor', color='black')
        plt.scatter(data_idx[y_test==1], y_pred[y_test==1], s= 10, color='red',label='Predicción tumor')
        plt.scatter(data_idx[y_test==0], y_test[y_test==0],s= 10, label='Real Normal', color='blue')
        plt.scatter(data_idx[y_test==0], y_pred[y_test==0], s= 10, color='green',label='Predicción normal')
        plt.xlabel('Data point',fontsize=18)
        plt.ylabel('Residual signal',fontsize=18)
        plt.title(opt[0],fontsize=20)
        plt.legend(fontsize=18)
        plt.show()  
        #fig.tight_layout()
        #fig.savefig('figs/' + opt[0] + '.png')
        #plt.close(fig)


with open('results/result.csv','w') as f:
    f.write('dataset,GTEx(N),TCGA(N),TCGA(C),keep_info,beta,F1,Precision,Recall,Specificity,Accuracy,AUC\n')
    for e in result:
        st = ','.join(map(str,e)) + '\n'
        f.write(st)
