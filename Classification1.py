#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 23:53:28 2021

@author: chirag
"""
#fruits = pd.DataFrame(df.drop(['Sample_ID', 'Collector', 'Texture_num'], axis=1))
#feature_names = fruits.columns[1:]
#X = fruits[feature_names]
####Loading required Data####
X = fruits.drop(['Texture'], axis = 1)
X = scaler.fit_transform(X)
y = fruits['Texture']

####Constants and other variables####
seq = ['Sa', 'LoSa', 'SaLo', 'SaClLo', 'SaCl', 'Cl', 'ClLo', 'Lo', 'SiCl', 'SiClLo', 'SiLo', 'Si']
neigh1 = np.array([[0, 1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[1,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	1,	0,	1,	0,	0,	0,	1,	0,	0,	1,	0],
[0,	0,	1,	0,	1,	0,	1,	1,	0,	0,	0,	0],
[0,	0,	0,	1,	0,	1,	1,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	1,	0,	1,	0,	1,	1,	0,	0],
[0,	0,	0,	1,	1,	1,	0,	1,	1,	1,	1,	0],
[0,	0,	1,	1,	0,	0,	1,	0,	0,	0,	1,	0],
[0,	0,	0,	0,	0,	1,	1,	0,	0,	1,	0,	0],
[0,	0,	0,	0,	0,	1,	1,	0,	1,	0,	1,	0],
[0,	0,	1,	0,	0,	0,	1,	1,	0,	1,	0,	1],
[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0]])

neigh2 = np.array([[1, 1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[1, 1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	1,	1,	1,	0,	0,	0,	1,	0,	0,	1,	0],
[0,	0,	1,	1,	1,	0,	1,	1,	0,	0,	0,	0],
[0,	0,	0,	1,	1,	1,	1,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	1,	1,	1,	0,	1,	1,	0,	0],
[0,	0,	0,	1,	1,	1,	1,	1,	1,	1,	1,	0],
[0,	0,	1,	1,	0,	0,	1,	1,	0,	0,	1,	0],
[0,	0,	0,	0,	0,	1,	1,	0,	1,	1,	0,	0],
[0,	0,	0,	0,	0,	1,	1,	0,	1,	1,	1,	0],
[0,	0,	1,	0,	0,	0,	1,	1,	0,	1,	1,	1],
[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	1]])


#Lr_tr, Lr_t, Dt_tr, Dt_t, Knn_tr, Knn_t, Lda_tr, Lda_t, Gnb_tr, Gnb_t, Svm_tr, Svm_t, Lr_con, Dt_con, Knn_con, Lda_con, Gnb_con, Svm_con, Lr_f, Dt_f, Knn_f, Lda_f, Gnb_f, Svm_f = [[] for _ in range(24)]
Lr_tr, Lr_t, Lda_tr, Lda_t, Svm_tr, Svm_t, Lr_con, Lda_con, Svm_con, Lr_tr_k, Lr_t_k, Lda_tr_k, Lda_t_k, Svm_tr_k, Svm_t_k, Lr_tr_tda, Lr_t_tda, Lda_tr_tda, Lda_t_tda, Svm_tr_tda, Svm_t_tda, Lr_tr_tda1, Lr_t_tda1, Lda_tr_tda1, Lda_t_tda1, Svm_tr_tda1, Svm_t_tda1  = [[] for _ in range(27)]

####Bootstrapping with 100 iterations####
for _ in range(100):
    #Data Split
    #X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    #LR
    logreg.fit(X_train, y_train)
    y_train_pred = logreg.predict(X_train)
    y_test_pred = logreg.predict(X_test)
    Lr_tr.append(logreg.score(X_train, y_train))
    Lr_t.append(logreg.score(X_test, y_test))
    Lr_tr_k.append(cohen_kappa_score(y_train, y_train_pred))
    Lr_t_k.append(cohen_kappa_score(y_test, y_test_pred))
    conf_tr = confusion_matrix(y_train, y_train_pred, labels = seq)
    Lr_tr_tda.append((np.sum(conf_tr * neigh1))/np.sum(conf_tr)) 
    Lr_tr_tda1.append((np.sum(conf_tr * neigh2))/np.sum(conf_tr)) 
    conf_test = confusion_matrix(y_test, y_test_pred, labels = seq)
    Lr_t_tda.append((np.sum(conf_test * neigh1))/np.sum(conf_test)) 
    Lr_t_tda1.append((np.sum(conf_test * neigh2))/np.sum(conf_test))
    Lr_con.append(conf_test)
    #LDA
    lda.fit(X_train, y_train)
    y_train_pred = lda.predict(X_train)
    y_test_pred = lda.predict(X_test)
    Lda_tr.append(lda.score(X_train, y_train))
    Lda_t.append(lda.score(X_test, y_test))
    Lda_tr_k.append(cohen_kappa_score(y_train, y_train_pred))
    Lda_t_k.append(cohen_kappa_score(y_test, y_test_pred))
    conf_tr = confusion_matrix(y_train, y_train_pred, labels = seq)
    Lda_tr_tda.append((np.sum(conf_tr * neigh1))/np.sum(conf_tr)) 
    Lda_tr_tda1.append((np.sum(conf_tr * neigh2))/np.sum(conf_tr)) 
    conf_test = confusion_matrix(y_test, y_test_pred, labels = seq)
    Lda_t_tda.append((np.sum(conf_test * neigh1))/np.sum(conf_test)) 
    Lda_t_tda1.append((np.sum(conf_test * neigh2))/np.sum(conf_test))
    Lda_con.append(conf_test)
    #SVM
    svm.fit(X_train, y_train)
    y_train_pred = svm.predict(X_train)
    y_test_pred = svm.predict(X_test)
    Svm_tr.append(svm.score(X_train, y_train))
    Svm_t.append(svm.score(X_test, y_test))
    Svm_tr_k.append(cohen_kappa_score(y_train, y_train_pred))
    Svm_t_k.append(cohen_kappa_score(y_test, y_test_pred))
    conf_tr = confusion_matrix(y_train, y_train_pred, labels = seq)
    Svm_tr_tda.append((np.sum(conf_tr * neigh1))/np.sum(conf_tr)) 
    Svm_tr_tda1.append((np.sum(conf_tr * neigh2))/np.sum(conf_tr)) 
    conf_test = confusion_matrix(y_test, y_test_pred, labels = seq)
    Svm_t_tda.append((np.sum(conf_test * neigh1))/np.sum(conf_test)) 
    Svm_t_tda1.append((np.sum(conf_test * neigh2))/np.sum(conf_test))
    Svm_con.append(conf_test)
    print(np.sum(conf_test))

####Compiling the results####
Metrics = [Lr_tr, Lr_t, Lda_tr, Lda_t, Svm_tr, Svm_t]
Kappa = [Lr_tr_k, Lr_t_k, Lda_tr_k, Lda_t_k, Svm_tr_k, Svm_t_k]
ANA = [Lr_tr_tda1, Lr_t_tda1, Lda_tr_tda1, Lda_t_tda1, Svm_tr_tda1, Svm_t_tda1]
Only_NA = [Lr_tr_tda, Lr_t_tda, Lda_tr_tda, Lda_t_tda, Svm_tr_tda, Svm_t_tda]
Conf = [Lr_con, Lda_con, Svm_con]

####Writing the results in output file####
stdout_fileinfo = sys.stdout
sys.stdout = open(name,'a')
print('Sequence-LR,LDA,SVM')
print(group)
print('Overall Accuracy')
for i in Metrics:
    i = np.array(i)
    print('{:.4f},{:.4f}'.format(i.mean(), i.std()))
print('Kappa')
for i in Kappa:
    i = np.array(i)
    print('{:.2f},{:.4f}'.format(i.mean(), i.std()))
print('Added Neighbourhood Accuracy')
for i in ANA:
    i = np.array(i)
    print('{:.4f},{:.4f}'.format(i.mean(), i.std()))
print('Only Neighbourhood Accuracy')
for i in Only_NA:
    i = np.array(i)
    print('{:.4f},{:.4f}'.format(i.mean(), i.std()))
print('Confusion Matrix')
for i in Conf:
    i = np.array(i)
    Lr_con_mean = np.around(np.mean(i, axis=0))
    Lr_con_std = np.around(np.std(i, axis=0))
    print(Lr_con_mean)
    print(Lr_con_std)
sys.stdout.close()
sys.stdout = stdout_fileinfo