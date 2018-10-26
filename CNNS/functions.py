import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from scipy.special import expit
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report, confusion_matrix
import time

seed = 42
np.random.seed(seed)


def reshape_x(x_array):
    new_x = []
    for row in x_array:
        img = row.reshape(28, 28)
        new_x.append(img)
    x = np.array(new_x)
    return x


def soft_max(x):
    """
    Stable softmax function as to not over run the float max of python 10^308
    :param x:
    :return:
    """
    x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return x/np.sum(x, axis=1, keepdims=True)


def crossEntropyLoss(x, y):
    # X is the output from fully connected layer (num_examples x num_classes)
    # y is labels (num_examples x 1)
    m = y.shape[0]
    p = soft_max(x)
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m

    # and its derivative
    dx = p.copy()
    dx[range(m), y] = dx[range(m), y] - 1
    dx = dx / m

    return loss, dx


def get_minibatches(x, y, minibatch_size, shuf=True):
    m = x.shape[0]
    minibatches = []
    if shuf:
        x, y = shuffle(x, y)
    for i in range(0, m, minibatch_size):
        x_batch = x[i:i + minibatch_size, :, :, :]
        y_batch = y[i:i + minibatch_size, ]
        minibatches.append((x_batch, y_batch))
    return minibatches


def get_minibatch(x, y, minibatch_size, shuf=True):
    if shuf:
        x, y = shuffle(x, y)
    for i in range(0, minibatch_size):
        x_batch = x[i:i + minibatch_size, :, :, :]
        y_batch = y[i:i + minibatch_size, ]
    return x_batch, y_batch


def batchdata(x, minibatch_size):
    m = x.shape[0]
    minibatches = []
    for i in range(0, m, minibatch_size):
        x_batch = x[i:i + minibatch_size, :, :, :]
        minibatches.append(x_batch)
    return minibatches


def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)  # both are not one hot encoded


def vanilla_update(params, grads, learning_rate=0.01):
    for param, grad in zip(params, reversed(grads)):
        for i in range(len(grad)):
            param[i] += - learning_rate * grad[i]


# Stochastic gradient descent
def sgd(nnet, x_train, y_train, f, minibatch_size, epoch, learning_rate, verbose=True, x_test=None, y_test=None):
    minibatches = get_minibatches(x_train, y_train, minibatch_size)
    val_x, val_y = get_minibatch(x_train, y_train, 1000)
    mb = batchdata(x_train, 1000)
    e_nnet = []
    e_accuracy = []
    e_validate = []
    e_loss = []
    e_loss_val = []
    for i in range(epoch):
        t1 = time.time()
        loss = 0
        pred = []
        if verbose:
            print("Epoch {0}".format(i + 1))
            f.write('Epoch {0}\n'.format(i + 1))
        for x_mini, y_mini in minibatches:
            loss, grads = nnet.train_step(x_mini, y_mini)
            vanilla_update(nnet.params, grads, learning_rate=learning_rate)
        if verbose:
            # todo break up x_train
            for j in range(len(mb)):
                pred.append(nnet.predict(mb[j]))
            pv = np.concatenate(pred, axis=0)
            train_acc = accuracy(y_train, pv)
            #train_acc = accuracy(y_train, nnet.predict(x_train))
            test_acc = accuracy(val_y, nnet.predict(val_x))
            v_loss, grads = nnet.train_step(val_x, val_y)
            e_nnet.append(nnet)
            e_accuracy.append(train_acc)
            e_validate.append(test_acc)
            e_loss.append(loss)
            e_loss_val.append(v_loss)
            t2 = time.time()
            print('Epoch run time {}'.format(t2-t1))
            print("Loss = {:.3f} | Train Accuracy = {:.3f} | Valid. Accuracy = {:.3f}".format
                  (loss, train_acc, test_acc))
            f.write('Epoch run time {}\n'.format(t2-t1))
            f.write("Loss = {:.3f} | Train Accuracy = {:.3f} | Valid. Accuracy = {:.3f}\n".format
                    (loss, train_acc, test_acc))

    return e_nnet, e_accuracy, e_validate, e_loss, e_loss_val

def model_summary(cnn, model_type, f):
    """
    Print the model summary
    :param model: nn model
    :param model_type: name of file for summary to be saved as :type: string 'nn.png'
    """
    print('-'*73)
    print('Layer (Type) Output Shape     Params')
    print('-'*73)
    print('Conv2D_1     {}      Filters: {} Shape: {} x {} Stride:{} Padding:{}'.format
          (cnn.layers[0].out_dim, cnn.layers[0].n_filter, cnn.layers[0].h_filter, cnn.layers[0].w_filter,
           cnn.layers[0].stride, cnn.layers[0].padding))
    print('-'*73)
    print('Activation: ', cnn.layers[1].name)
    print('-'*73)
    print('MaxPool_1    {}      Shape: {} x {} Stride:{}'.format
          (cnn.layers[2].out_dim, cnn.layers[2].size, cnn.layers[2].size, cnn.layers[2].stride))
    print('-'*73)
    print('Conv2D_2     {}      Filters: {} Shape: {} x {} Stride:{} Padding:{}'.format
          (cnn.layers[3].out_dim, cnn.layers[3].n_filter, cnn.layers[3].h_filter, cnn.layers[3].w_filter,
           cnn.layers[3].stride, cnn.layers[3].padding))
    print('-'*73)
    print('Activation: ', cnn.layers[4].name)
    print('-'*73)
    print('MaxPool_2    {}      Shape: {} x {} Stride:{}'.format
          (cnn.layers[5].out_dim, cnn.layers[5].size, cnn.layers[5].size, cnn.layers[5].stride))
    print('-'*73)
    print('Flatten')
    print('-'*73)
    print('Fully Connected Layer         Neurons: {}'.format(cnn.layers[7].out_dim))
    print('-'*73)
    print('Activation: ', cnn.layers[8].name)
    print('-'*73)
    print('Output Layer                  Output: {}'.format(cnn.layers[9].out_dim))
    print('-'*73)
    print('Activation: Softmax')

    f.write('='*73+'\n')
    f.write('Layer (Type) Output Shape     Params\n')
    f.write('='*73+'\n')
    f.write('Conv2D_1     {}      Filters: {} Shape: {} x {} Stride:{} Padding:{}\n'.format
          (cnn.layers[0].out_dim, cnn.layers[0].n_filter, cnn.layers[0].h_filter, cnn.layers[0].w_filter,
           cnn.layers[0].stride, cnn.layers[0].padding))
    f.write('='*73+'\n')
    f.write('Activation: ' + cnn.layers[1].name+'\n')
    f.write('='*73+'\n')
    f.write('MaxPool_1    {}      Shape: {} x {} Stride:{}\n'.format
          (cnn.layers[2].out_dim, cnn.layers[2].size, cnn.layers[2].size, cnn.layers[2].stride))
    f.write('='*73+'\n')
    f.write('Conv2D_2     {}      Filters: {} Shape: {} x {} Stride:{} Padding:{}\n'.format
          (cnn.layers[3].out_dim, cnn.layers[3].n_filter, cnn.layers[3].h_filter, cnn.layers[3].w_filter,
           cnn.layers[3].stride, cnn.layers[3].padding))
    f.write('='*73+'\n')
    f.write('Activation: ' + cnn.layers[4].name+'\n')
    f.write('='*73+'\n')
    f.write('MaxPool_2    {}      Shape: {} x {} Stride:{}\n'.format
          (cnn.layers[5].out_dim, cnn.layers[5].size, cnn.layers[5].size, cnn.layers[5].stride))
    f.write('='*73+'\n')
    f.write('Flatten\n')
    f.write('='*73+'\n')
    f.write('Fully Connected Layer         Neurons: {}\n'.format(cnn.layers[7].out_dim))
    f.write('='*73+'\n')
    f.write('Activation: ' + cnn.layers[8].name+'\n')
    f.write('='*73+'\n')
    f.write('Output Layer                  Output: {}\n'.format(cnn.layers[9].out_dim))
    f.write('='*73+'\n')
    f.write('Activation: Softmax\n\n')

def plot_history(loss, accuracy, validate, loss_validate):
    # As loss always exists
    epochs = range(1, len(loss) + 1)

    # Loss
    plt.figure(1)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, loss_validate, 'g', label='Validate loss')
    title = 'cnn_loss'
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    t = 'cnn_loss.png'
    plt.savefig(t)

    # Accuracy
    plt.figure(2)
    plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
    plt.plot(epochs, validate, 'g', label='Validate Accuracy')
    title = 'cnn_Accuracy'
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    t = 'cnn_accuracy.png'
    plt.savefig(t)
    plt.show()

    temp = max(enumerate(accuracy), key=(lambda x: x[1]))

    return temp


def plot_confusion_matrix(cm, model_type, f, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param cm: confusion matrix
    :param classes: class_names 0-9
    :param normalize: normalize true/false
    :param title: Title of plot :type: String
    :param cmap: color map
    :param model_type: nn model type
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix\n============================")
        t = model_type + '_norm_cfm.png'
    else:
        print('Confusion matrix, without normalization\n============================')
        t = model_type + '_cfm.png'

    print(cm)
    print("\n")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(t)


class TanH:
    def __init__(self):
        self.params = []
        self.name = 'tanH'

    def forward(self, x):
        """
        Passes the Matrix X into the tanh function for activation.
        :param x: nxm Matrix
        :return:  nxm matrix
        """
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self, dout):
        delta_x = dout * (1 - self.out ** 2)
        return delta_x, []


class ReLU:
    def __init__(self):
        self.params = []
        self.name = 'ReLU'

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        dx = dout.copy()
        dx[self.x <= 0] = 0
        return dx, []

class sigmoid():
    def __init__(self):
        self.params = []
        self.name = 'Sigmoid'

    def forward(self, x):
        out = expit(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx, []
