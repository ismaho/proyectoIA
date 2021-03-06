import os
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from  sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix



def get_data(din):
    bf = pd.read_csv(din,sep='\t')
    bf = bf[~bf.isin([np.nan, np.inf, -np.inf]).any(1)].copy()
    genes = bf['Hugo_Symbol']
    cols = list(bf.columns)
    cols.remove('Entrez_Gene_Id')
    cols.remove('Hugo_Symbol')
    bf = bf[cols]
    bf = bf.transpose()
    bf.columns = genes
    return bf


def get_score_threshold(X, y):
    
    X_digits= X
    y_digits = y
    pca = PCA()
    # set the tolerance to a large value to make the example faster
    logistic = LogisticRegression(max_iter=100000, tol=0.1)
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {
        'pca__n_components': [32, 64, 128],
        'logistic__C': np.logspace(-4, 4, 4),
    }
    search = GridSearchCV(pipe, param_grid, n_jobs=-1)
    search.fit(X_digits, y_digits)
    prediction = search.predict(X_digits)
    score_in = search.best_score_
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    
    # Plot the PCA spectrum
    pca.fit(X_digits)

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
    ax0.plot(np.arange(1, pca.n_components_ + 1),
            pca.explained_variance_ratio_, '+', linewidth=2)
    ax0.set_ylabel('PCA explained variance ratio')

    ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='n_components chosen')
    ax0.legend(prop=dict(size=12))

    # For each number of components, find the best classifier results
    results = pd.DataFrame(search.cv_results_)
    components_col = 'param_pca__n_components'
    best_clfs = results.groupby(components_col).apply(
        lambda g: g.nlargest(1, 'mean_test_score'))

    best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
                legend=False, ax=ax1)
    ax1.set_ylabel('Classification accuracy (val)')
    ax1.set_xlabel('n_components')

    plt.xlim(-1, 70)

    plt.tight_layout()
    plt.show()
    
    return prediction , search, score_in



opts = [['breast','brca'], ['liver','lihc'], ['lung','luad'],['prostate','prad'],['stomach','stad'],['thyroid','thca']]
dirin = 'data/'

result = []


for opt in opts:
    din2 = dirin + opt[1] + '-rsem-fpkm-tcga.txt'
    din3 = dirin + opt[1] + '-rsem-fpkm-tcga-t.txt'
    if os.path.isfile(din2) and os.path.isfile(din3):
        normal2 = get_data(din2)
        abnormal = get_data(din3)       
        X_test = pd.concat( [normal2, abnormal] ).values
        y_test = np.zeros(len(X_test), dtype=int)
        y_test[len(normal2):] = 1

        #import pdb; pdb.set_trace()
        y_pred, model, score = get_score_threshold(X_test,y_test)

        Accuracy = score
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
        Specificity = tn / (tn+fp)

        title = opt[0] + " Confusion matrix"  
        disp = plot_confusion_matrix(model, X_test, y_test, cmap= plt.cm.Blues, normalize = 'true', display_labels = [("No tumor"),("Tumor")] )
        disp.ax_.set_title(title)
        #disp.savefig('figs/' + opt[0] + 'ConfusionMatrix.png')
        plt.show()

        ret = [Accuracy]
        ret = [round(e,3) for e in ret]
        ret = [opt[0], len(normal2), len(abnormal)]  + ret
        print(ret[0],ret[-6:])     
        result += [ ret ]

        data_idx = np.arange(len(X_test))
        plt.subplots(figsize=(10, 10))
        plt.scatter(data_idx[y_test==1], y_test[y_test == 1],s= 20, label='Real Tumor', color='black')
        plt.scatter(data_idx[y_test==1], y_pred[y_test >= 1], s= 10, color='red',label='Predicción tumor')
        plt.scatter(data_idx[y_test==0], y_test[y_test == 0],s= 20, label='Real Normal', color='blue')
        plt.scatter(data_idx[y_test==0], y_pred[y_test <= 0], s= 10, color='green',label='Predicción normal')
        plt.xlabel('Data point',fontsize=18)
        plt.ylabel('Residual signal',fontsize=18)
        plt.title(opt[0],fontsize=20)
        plt.legend(fontsize=18)
        plt.savefig('figs/' + opt[0] + '.png')
        plt.show()  
        #plt.close(fig)
        


with open('results/result.csv','w') as f:
    f.write('dataset,GTEx(N),TCGA(N),TCGA(C),keep_info,beta,F1,Precision,Recall,Specificity,Accuracy,AUC\n')
    for e in result:
        st = ','.join(map(str,e)) + '\n'
        f.write(st)
