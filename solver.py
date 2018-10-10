import numpy as np
from sklearn.utils import shuffle
from deepnet.utils import accuracy
import copy
from deepnet.loss import SoftmaxLoss
import matplotlib.pyplot as plt

def get_minibatches(X, y, minibatch_size,shuffle_bool=True):
    m = X.shape[0]
    minibatches = []
    if shuffle_bool:
        X, y = shuffle(X, y)
        print(X.shape,y.shape)
    for i in range(0, m, minibatch_size):
        X_batch = X[i:i + minibatch_size, :, :, :]
        y_batch = y[i:i + minibatch_size, ]
        minibatches.append((X_batch, y_batch))
    return minibatches

def momentum_update(velocity, params, grads, learning_rate=0.01, mu=0.9):
    for v, param, grad, in zip(velocity, params, reversed(grads)):
        for i in range(len(grad)):
            v[i] = mu * v[i] + learning_rate * grad[i]
            param[i] -= v[i]

def plot_graph(loss,epoch,train_acc,test_acc,val_loss):
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def sgd_momentum(nnet, X_train, y_train, minibatch_size, epoch, learning_rate, mu=1e-6,
                 verbose=True, X_test=None, y_test=None, nesterov=True):
    Loss = []
    ValLoss = []
    Epoch_list = []
    Train_Acc = []
    Test_Acc = []
    minibatches = get_minibatches(X_train, y_train, minibatch_size)
    for i in range(epoch):
        loss = 0
        velocity = []
        for param_layer in nnet.params:
            p = [np.zeros_like(param) for param in list(param_layer)]
            velocity.append(p)

        if verbose:
            print("Epoch {0}".format(i + 1))

        for X_mini, y_mini in minibatches:

            if nesterov:
                for param, ve in zip(nnet.params, velocity):
                    for i in range(len(param)):
                        param[i] += mu * ve[i]

            loss, grads = nnet.train_step(X_mini, y_mini)
            momentum_update(velocity, nnet.params, grads,
                            learning_rate=learning_rate, mu=mu)

        if verbose:
            m_train = X_train.shape[0]
            m_test = X_test.shape[0]
            y_train_pred = np.array([], dtype="int64")
            y_test_pred = np.array([], dtype="int64")
            for i in range(0, m_train, minibatch_size):
                X_tr = X_train[i:i + minibatch_size, :, :, :]
                y_tr = y_train[i:i + minibatch_size, ]
                y_train_pred = np.append(y_train_pred, nnet.predict(X_tr))
            for i in range(0, m_test, minibatch_size):
                X_te = X_test[i:i + minibatch_size, :, :, :]
                y_te = y_test[i:i + minibatch_size, ]
                y_test_pred = np.append(y_test_pred, nnet.predict(X_te))
            _,val_loss = nnet.evaluate(X_test,y_test)
            train_acc = accuracy(y_train, y_train_pred)
            test_acc = accuracy(y_test, y_test_pred)
            print("Loss = {0} | Training Accuracy = {1} | Test Accuracy = {2} | Test loss = {3}".format(
                loss, train_acc, test_acc,val_loss))
            Loss.append(loss)
            Epoch_list.append(epoch)
            Train_Acc.append(train_acc)
            Test_Acc.append(test_acc)
            ValLoss.append(val_loss)
    plot_graph(Loss,Epoch_list,Train_Acc,Test_Acc,ValLoss)
    return nnet
