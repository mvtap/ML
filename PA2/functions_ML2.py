import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import tensorflow as tf
from tensorflow import keras


#Classification models
from sklearn.linear_model import LogisticRegression as LR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RFc
from sklearn.tree import DecisionTreeClassifier as DTc
from sklearn.svm import SVC as SVMc
from sklearn.neural_network import MLPClassifier as MLPc

#GridSearch
from sklearn.model_selection import GridSearchCV

#Model metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, auc

import warnings
warnings.filterwarnings('ignore')



def plot_classifier_boundary(model,x,h = .05): #kindly provided in class
    cmap_light = colors.ListedColormap(['lightsteelblue', 'peachpuff'])
    x_min, x_max = x[:, 0].min()-.2, x[:, 0].max()+.2
    y_min, y_max = x[:, 1].min()-.2, x[:, 1].max()+.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.xlim((x_min,x_max))
    plt.ylim((y_min,y_max))





def Test_models(x, y, clusters=2):
    models = [LR(), LDA(), QDA(), RFc(), DTc(), SVMc(kernel='rbf'), 
              SVMc(kernel='poly'), MLPc(activation='relu'), MLPc(activation='tanh'), SVMc(kernel='linear')]
    modls = ['LR','LDA','QDA','RFc','DTc','SVMc_rbf','SVMc_poly','MLPc_relu','MLPc_tanh','SVMc_lin']
    
    grid_params_DTc = {'criterion': ['gini', 'entropy'],
        'max_depth': range(1, 10),
        'min_samples_split': range(1, 10),
        'min_samples_leaf': range(1, 5) }    
    grid_params_RFc = {"n_estimators": [10, 50, 100, 200],
             "criterion": ['gini', 'entropy'],
             "min_samples_split": [2, 5, 10, 20],
             "min_samples_leaf": [1, 2 , 5 , 10],
             "min_impurity_decrease": [0.0, 1.0, 2.0]}
    grid_params_SVMc = {'C':[0.001,0.005,0.01,0.05, 0.1,0.5, 1, 5, 10]}
    grid_params_MLPc = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive'],
            'batch_size':[100, 50, 25]}
    if clusters > 2:
        models = models[1:]
        modls = modls[1:]
    fig = plt.figure(figsize=(30,20))
    Acc = []
    F1 = []
    Auc =  []
    T_time = []
    for i in range(len(models)):
        mod = models[i]
        if i == 4:
            grid_mod = GridSearchCV(DTc(), grid_params_DTc, verbose=1, cv=5, n_jobs=-1, scoring = 'f1')
            grid_mod.fit(x, y)
            mod = grid_mod.best_estimator_
        elif i == 3:
            grid_mod = GridSearchCV(RFc(random_state=42), grid_params_RFc, verbose=1, cv=5, n_jobs=-1, scoring = 'f1')
            grid_mod.fit(x, y)
            mod = grid_mod.best_estimator_
        elif i == 5:
            grid_mod = GridSearchCV(SVMc(kernel='rbf'), grid_params_SVMc, verbose=1, cv=5, n_jobs=-1, scoring = 'f1')
            grid_mod.fit(x, y)
            mod = grid_mod.best_estimator_
        elif i == 6:
            grid_mod = GridSearchCV(SVMc(kernel='poly'), grid_params_SVMc, verbose=1, cv=5, n_jobs=-1, scoring = 'f1')
            grid_mod.fit(x, y)
            mod = grid_mod.best_estimator_
        elif i == 7:
            grid_mod = GridSearchCV(MLPc(activation='relu'), grid_params_MLPc, verbose=1, cv=5, n_jobs=-1, scoring = 'f1')
            grid_mod.fit(x, y)
            mod = grid_mod.best_estimator_
        elif i == 8:
            grid_mod = GridSearchCV(MLPc(activation='tanh'), grid_params_MLPc, verbose=1, cv=5, n_jobs=-1, scoring = 'f1')    
            grid_mod.fit(x, y)
            mod = grid_mod.best_estimator_
        elif i == 9:
            grid_mod = GridSearchCV(SVMc(kernel='linear'), grid_params_SVMc, verbose=1, cv=5, n_jobs=-1, scoring = 'f1')
            grid_mod.fit(x, y)
            mod = grid_mod.best_estimator_
        else:
            mod = models[i]
        start = time.time()
        mod.fit(x, y)
        stop = time.time()

        print(mod)
        print('Accuracy:', cross_val_score(mod, x, y, cv=5, scoring=('accuracy')))
        Acc.append(np.mean(cross_val_score(mod, x, y, cv=5, scoring=('accuracy'))))
        print('F1:', cross_val_score(mod, x, y, cv=5, scoring=('f1')))
        F1.append(np.mean(cross_val_score(mod, x, y, cv=5, scoring=('f1'))))
        print('AUC:', cross_val_score(mod, x, y, cv=5, scoring=('roc_auc')))
        Auc.append(np.mean(cross_val_score(mod, x, y, cv=5, scoring=('roc_auc'))))

        print(f"Training time: {stop - start}s")
        T_time.append(stop-start)
        print('\n')

        ax = fig.add_subplot(4, 4, i+1)
        plot_classifier_boundary(mod,x)
        ax.scatter(x[:,0],x[:,1],color=cmap(y))
        ax.set_title(models[i], fontsize = 13)
        ax.set_xlabel('$x1$')
        ax.set_ylabel('$x2$')

    Metrics = [Acc, F1, Auc, T_time]
    Labels = ['Accuracy score', 'F1', 'AUC', 'Training time']
    for i in range(4):
        ax = fig.add_subplot(4, 4, i+13) 
        ax.barh(modls, Metrics[i])
        ax.set_xlabel(Labels[i])
    plt.show() 




    def test_tf(x,y):
        model = keras.models.Sequential()

        model.add(keras.layers.Flatten(input_shape=[len(x)]))
        model.add(keras.layers.Dense(100, activation="relu"))
        model.add(keras.layers.Dense(1, activation="relu"))

        model.summary()

        model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

        history = model.fit(x, y, epochs=100)
