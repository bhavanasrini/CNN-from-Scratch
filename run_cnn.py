import numpy as np
from deepnet.layers import *
from deepnet.solver import sgd_momentum
from deepnet.nnet import CNN
import sys
from LoadDataModule import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

def print_summary(name,layer_dim,kernel_size):
    if (name == "Conv2D"):
        print(name,"               ",(None,layer_dim),"              ",(layer_dim[0]*(kernel_size*(kernel_size*kernel_size)+1)))
    elif (name == "Maxpooling"):
        print(name,"           ",(None,layer_dim),"              ","0")
    elif (name == "Flatten"):
        print(name,"                 ",(None,np.prod(layer_dim)),"                   ","0")
    elif(name == "Dense"):
        print(name,"                   ",(None,kernel_size),"                   ",(kernel_size*(np.prod(layer_dim)+1)))

def make_mnist_cnn(X_dim, num_class):
    print("Model Summary")
    print("--------------------------------------------------------------")
    print("Layer (type)                 Output Shape              Param #")
    conv1 = Conv(X_dim, n_filter=2, h_filter=3,
                w_filter=3, stride=1, padding=0)
    print_summary("Conv2D",conv1.out_dim,3)
    sigmoid_conv = sigmoid()
    maxpool1 = Maxpool(conv1.out_dim, size=2, stride=1)
    print_summary("Maxpooling",maxpool1.out_dim,0)
    conv2 = Conv(maxpool1.out_dim, n_filter=2, h_filter=3,
                w_filter=3, stride=1, padding=0)
    print_summary("Conv2D",conv2.out_dim,3)
    relu_conv = ReLU()
    maxpool2 = Maxpool(conv2.out_dim, size=2, stride=1)
    print_summary("Maxpooling",maxpool2.out_dim,0)
    flat = Flatten()
    print_summary("Flatten",maxpool2.out_dim,0)
    fc1 = FullyConnected(np.prod(maxpool2.out_dim), 50)
    #print(fc1)
    print_summary("Dense",maxpool2.out_dim,50)
    tanh_conv = tanh()
    fc2 = FullyConnected(50, num_class)
    print_summary("Dense",50,num_class)
    return [conv1,sigmoid_conv,maxpool1,conv2,relu_conv, maxpool2, flat, fc1,tanh_conv,fc2]

if __name__ == "__main__":
    np.random.seed(107629491)
    ld = LoadDataModule()
    images,labels = ld.load('train')
    training_x = images[:48000]
    validation_x = images[48000:]
    training_y = labels[:48000]
    validation_y = labels[48000:]
    training_x = training_x.astype('float32')
    validation_x = validation_x.astype('float32')
    training_x = training_x / 255
    validation_x = validation_x / 255
    shape = (-1, 1, 28, 28)
    training_x = training_x.reshape(shape)
    validation_x = validation_x.reshape(shape)
    mnist_dims = (1, 28, 28)
    cnn = CNN(make_mnist_cnn(mnist_dims, num_class=10))
    cnn = sgd_momentum(cnn, training_x, training_y, minibatch_size=200, epoch=20,
                       learning_rate=0.001, X_test=validation_x, y_test=validation_y)
    images,labels = ld.load('test')
    test_x = images
    test_y = labels
    test_x = test_x / 255
    test_x = test_x.reshape(shape)
    test_pred_y = cnn.predict(test_x)
    print(classification_report(test_y, test_pred_y))
    cm = confusion_matrix(test_y, test_pred_y)
    y_classes = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
    sns.heatmap(cm, annot=True, fmt='d',xticklabels=y_classes, yticklabels=y_classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()