from image_to_cols import *
import data_process
import functions as mf
import layers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


f = open('report.doc', 'w')
ld = data_process.LoadDataModule()
# Load Data into training/testing sets
x_train, y_train = ld.load('train')
x_test, y_test = ld.load('test')
# Reshape the data back into images from 784 to 28x28 images
x_train = mf.reshape_x(x_train)
x_test = mf.reshape_x(x_test)
# train shape of (60000, 1, 28, 28) test shape (10000, 1, 28, 28)
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')
# min max scale from 0-255 to 0-1 scale
x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))
x_dims = (1, 28, 28)
num_classes = 10
class_names = np.unique(y_train)

# Conv layers with x dims 2 filters with kernel 3x3 stride of 1 and no padding
conv1 = layers.Conv(x_dims, n_filter=2, h_filter=3, w_filter=3, stride=1, padding=0)
# activation for layer 1 'sigmoid'
sig = mf.sigmoid()
# MaxPool layer 2x2 stride of 1
pool1 = layers.Maxpool(conv1.out_dim, size=2, stride=2)
# Conv layer with 2 filters kernel size of 3x3 stride of 1 and no padding
conv2 = layers.Conv(pool1.out_dim, n_filter=2, h_filter=3, w_filter=3, stride=1, padding=0)
# activation for layer 2 rectified linear
relu = mf.ReLU()
# MaxPool layer 2x2 stride 1
pool2 = layers.Maxpool(conv2.out_dim, size=2, stride=1)
# Flatten the matrix
flat = layers.Flatten()
# Fully connected layer with 50 neurons
fc1 = layers.FullyConnected(np.prod(pool2.out_dim), 50)
# Activation for fully connected layer of 50 neurons is tanh
tanh = mf.TanH()

# Fully connected layer with 10 neurons 'output layer'
out = layers.FullyConnected(50, num_classes)

cnn = layers.CNN([conv1, sig, pool1, conv2, relu, pool2, flat, fc1, tanh, out])

mf.model_summary(cnn, 'cnn_model_plot.png', f)

e_nnet, e_accuracy, e_validate, e_loss, e_loss_val = mf.sgd(cnn, x_train, y_train, f, minibatch_size=200, epoch=20,
                                                            learning_rate=0.01)


best_net = mf.plot_history(e_loss, e_accuracy, e_validate, e_loss_val)
mb = mf.batchdata(x_test, 1000)
pred = []
for j in range(len(mb)):
    pred.append(e_nnet[best_net[0]].predict(mb[j]))
pv = np.concatenate(pred, axis=0)

# y_pred = e_nnet[best_net[0]].predict(x_test)

print('Test Set Accuracy with best model parameters: {}'.format(mf.accuracy(y_test, pv)))
f.write('\n\n')
f.write('Test Set Accuracy with best model parameters: {}\n'.format(mf.accuracy(y_test, pv)))

# Print classification report
print("Classification report \n=======================")
print(classification_report(y_true=y_test, y_pred=pv))
print("Confusion matrix \n=======================")
print(confusion_matrix(y_true=y_test, y_pred=pv))

f.write("Classification report \n=======================\n")
f.write(classification_report(y_true=y_test, y_pred=pv)+'\n')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true=y_test, y_pred=pv)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
mf.plot_confusion_matrix(cnf_matrix, 'cnn', f, classes=class_names, title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
mf.plot_confusion_matrix(cnf_matrix, 'cnn', f, classes=class_names, normalize=True, title='Normalized confusion matrix')

plt.show()
