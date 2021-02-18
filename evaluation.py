import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np

from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.layers import GlobalAveragePooling2D, Dense
from keras import backend as K
from keras.engine.network import Network

from keras.applications import MobileNetV2
from oneclass_data_loader import DataLoader

import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import brentq
from scipy.interpolate import interp1d


#  evaluation function
def eval_bc_model(model_dir, loader, batch_size=32):
    tp, tn, fp, fn = 0, 0, 0, 0
    model = load_model(model_dir, custom_objects={"recall": recall, 'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN})
    num_data = len(loader.data)
    num_batch = int(np.floor(num_data / batch_size))
    for i in range(num_batch):
        x, y = loader.next_batch(batch_size)
        pred = model.predict(x)
        pred = pred.argmax(axis = 1)
        y = y.argmax(axis = 1) 

        for idx in range(len(pred)):
            if pred[idx]==1 and y[idx]==1:
                tp += 1
            elif pred[idx]==0 and y[idx]==0:
                tn += 1
            elif pred[idx]==1 and y[idx]==0:
                fp += 1
            elif pred[idx]==0 and y[idx]==1:
                fn += 1
    
    num_processed = int(batch_size * num_batch)
    x, y = loader.next_batch(num_data - num_processed)
    pred = model.predict(x)
    pred = pred.argmax(axis = 1)
    y = y.argmax(axis = 1) 

    for idx in range(len(pred)):
        if pred[idx]==1 and y[idx]==1:
            tp += 1
        elif pred[idx]==0 and y[idx]==0:
            tn += 1
        elif pred[idx]==1 and y[idx]==0:
            fp += 1
        elif pred[idx]==0 and y[idx]==1:
            fn += 1

    return tp, tn, fp, fn


def eval_dloc_model(model_dir, sample_loader, test_loader, n_neighbors=5):
    model = load_model(model_dir, custom_objects={"original_loss": original_loss})

    # load data
    sample_x, sample_y = sample_loader.next_batch(len(sample_loader.data))
    test_x, test_y = test_loader.next_batch(len(test_loader.data))

    # extract embedding
    sample_x = model.predict(sample_x)
    test_x = model.predict(test_x)

    # norm 
    ms = MinMaxScaler()
    sample_x = ms.fit_transform(sample_x)
    test_x = ms.transform(test_x)

    # generate score
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    _ = clf.fit(sample_x)
    z = -clf._decision_function(test_x)
    z = np.array(z).reshape(-1, 1)
    g_true = [np.where(r==1)[0][0] for r in test_y]
    

    fpr, tpr, thresholds = metrics.roc_curve(g_true, z)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1., full_output=False)
    eer_thresh = interp1d(fpr, thresholds)(eer) 
    auc = metrics.auc(fpr, tpr)

    pred = np.zeros(z.shape)
    pred[np.where(z>eer_thresh)] = 1
    tn, fp, fn, tp = metrics.confusion_matrix(g_true, pred).ravel()
    print(('-' * 5) + 'result' + ('-' * 10))
    print('EER:{}\nAUC:{}\nTN:{}\nFP:{}\nFN:{}\nTP:{}\n'.format(eer, auc, tn, fp, fn, tp))
    return tp, tn, fp, fn, eer, eer_thresh, auc, fpr, tpr

