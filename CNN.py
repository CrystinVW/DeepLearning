################################
### IMPORT LIBRARIES
##################################
from sklearn.model_selection import train_test_split
from sklearn import metrics
from skimage.color import rgb2lab
from skimage.transform import resize
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob


##################################
### COLLECTS THE DATA FROM THE FOLDERS
#### ITERATES THROUGH EACH ONE
##### CONVERTS THE IMAGE
###### RESIZES
####### BUILDS X AND y
######################################
def get_data(start_path, num_different_ys, long):
    ############################
    #### CREATE X AND Y DATASET
    ############################
    x_dataset = []
    y_dataset = []

    ##############
    ### HELPS ADD ALPHABETICALLY TO GO THROUGH DICTIONARIES
    #############
    num = 0

    #############
    ### ITERATE THROUGH EACH FOLDER, A-Y, -J
    for z in range(num_different_ys):

        letter = chr(ord('a') + num)
        ##############################
        #### NOT PREDICTING J
        ###############################
        if long and letter == 'j':
            letter = chr(ord('b') + num)
        #################################
        #### GRABS PATH OF IMAGE
        ##################################
        directory = start_path + letter + '/'
        for item in glob.glob(directory + "*.png"):
            #################################
            #### READS IN AS FLOAT, THEN CONVERTS
            #### TO NUMBER BETWEEN 0-1
            ##################################
            letter_image = plt.imread(item).astype(np.float32)
            try:

                letter_image = rgb2lab(letter_image / 255.0)[:, :, 0]
            except:
                #######
                # IN CASE THE IMAGE SIZE IS NOT CORRECT SIZE
                b = letter_image[:, -1:]  # ADDS LAST COLUMN TO THE NP ARRAY
                #### EXTENDS HORIZONTALLY
                letter_image = np.hstack((letter_image, b))

            ##############################
            ### AFTER EACH ITERATION
            ### MARK WITH A 1 THE LETTER
            ### OR Y OUTCOME WE ARE GOING
            ### THROUGH
            ###############################
            label = np.zeros((num_different_ys,), dtype=np.float32)
            label[num] = 1.0

            ## APPEND label
            y_dataset.append(label)
            ##################
            ## RESIZE IMAGE
            letter_image = resize(letter_image, fix_image_size, mode='reflect')
            letter_image = letter_image.astype(np.float32)
            #################################
            #### APPEND NP IMAGE ARRAY
            #### TO MASTER LIST/DATASET
            ##################################
            x_dataset.append(letter_image)

        ### INCREMENT LETTER BY ONE TO HELP WITH ALPHABET
        ### AND TO HELP WIH INDEX OF LABEL
        num += 1
        # print(y_dataset)
    ##### TEMP X = STACK THE IMAGES ON TOP OF EACH OTHER
    temp_X = np.stack([image[:, :, np.newaxis] for image in x_dataset], axis=0).astype(np.float32)
    return ImageData(X_data=temp_X.astype(np.float32),
                     y_data=np.matrix(y_dataset).astype(np.float32))


#####################################
#### DIFFERENT LAYERS THAT ARE TO BE APPLIED TO
#### THE CONVOLUTIONAL NEURAL NETWORK
###### FULLY CONNECTED LAYER
###### CONVOLUTIONAL LAYER
###### OUTPUT LAYER


##### FULLY CONNECTED LAYER
def fully_connected_layer(layer_name, num_nodes):
    return tf.nn.relu(output_layer(layer_name, num_nodes))


####### CONVOLUTIONAL LAYER
def convolutional_layer(layer_name, kernel_size, num_nodes):
    return tf.nn.relu(
        tf.nn.conv2d(layer_name, tf.get_variable('CW', [kernel_size, kernel_size, layer_name.get_shape()[3],
                                                        num_nodes], tf.float32,
                                                 tf.contrib.layers.variance_scaling_initializer()), [1, 1, 1, 1],
                     'SAME') + tf.get_variable('CB', [num_nodes, ], tf.float32, tf.constant_initializer(0.0)))


#######################
##### OUTPUT LAYER
#########################
def output_layer(layer_name, num_nodes):
    return tf.matmul(layer_name, tf.get_variable('W', [layer_name.get_shape()[1], num_nodes], tf.float32,
                                                 tf.contrib.layers.variance_scaling_initializer())) + tf.get_variable(
        'B', [num_nodes, ], tf.float32, tf.constant_initializer(0.0))


#######################
### DROPOUT
#######################
def drop_out_nn(layer_name, percentage, x_y):
    return tf.cond(x_y, lambda: tf.nn.dropout(layer_name, percentage), lambda: layer_name)


#############################
#####  MAX POOLING - GRABS MAX OUT OF EACH SAMPLE MATRIX
###############################
def max_pooling(layer_name, sampling):
    return tf.nn.max_pool(layer_name, [1, sampling, sampling, 1], [1, sampling, sampling, 1], 'SAME')


##########################
### MAKES ITS WAY THROUGH THE NETWORK
###########################
### FIX THIS METHOD
def traverse_network(layer_name, x_y, long):
    # FIRST LAYER
    with tf.variable_scope('layer1'):
        layer1_out = drop_out_nn(max_pooling(convolutional_layer(layer_name, 3, 24), 2), 0.7,
                                 x_y)  # tweak these numbers

    # SECOND LAYER
    with tf.variable_scope('layer2'):
        layer2_out = drop_out_nn(max_pooling(convolutional_layer(layer1_out, 3, 60), 2), 0.7, x_y)  # play these numbers



    # FLATTEN LAYER
    with tf.variable_scope('flatten'):
        layer2_out_flattened = tf.contrib.layers.flatten(layer2_out)

    # FULLY CONNECT
    with tf.variable_scope('connect'):
        layer3_out = drop_out_nn(fully_connected_layer(layer2_out_flattened, 720), 0.7, x_y)  # play with these numbers

    # OUTPUT LAYER
    with tf.variable_scope('OUT'):
        if long:
            return output_layer(layer3_out, 24)
        else:
            return output_layer(layer3_out, 9)


            ###########################################
            ### SOME OF THIS PART IS BASED OFF OF THE CODE
            ### PROVIDED THROUGH CLASS TO BUILD ON
            #########################################


####### RUNS THE PROGRAM
def conv_neural_net(X_train, y_train, X_test, y_test, learning_rate, num_epochs, size_of_small_batches, long):
    x_batch = tf.placeholder(tf.float32, shape=(None, fix_image_size[0], fix_image_size[1], 1))
    if long:
        y_batch = tf.placeholder(tf.float32, shape=(None, 24))
    else:
        y_batch = tf.placeholder(tf.float32, shape=(None, 9))

    train_x_y = tf.placeholder(tf.bool)

    ####### RUN THE MODEL
    if long:
        run_model = traverse_network(x_batch, train_x_y, True)
    else:
        run_model = traverse_network(x_batch, train_x_y, False)

    loss_amount = tf.nn.softmax_cross_entropy_with_logits(logits=run_model, labels=y_batch)

    ###### OPTIMIZE THE LOSS
    loss = tf.reduce_mean(loss_amount)
    find_best_solution = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    predicted_y = tf.nn.softmax(run_model)

    with tf.Session() as round:
        round.run(tf.global_variables_initializer())

        ##### PRINT REMINDER
        print("REMEMBER: We want the lowest loss score! IF it goes up the algorithm is bad.")

        #### RUN FOR X AMOUNT OF TIMES
        for epoch in range(num_epochs):

            total_score = []
            for item in grab_portion(X_train, y_train, size_of_small_batches):
                #############
                ### Output = OPTIMIZED LOSS FROM NETWORK
                #############
                output = round.run([find_best_solution, loss],
                                   feed_dict={x_batch: item[0], y_batch: item[1], train_x_y: True})

                total_score.append(output[1])

            ###################################
            ### Grabs the average score to report as loss
            #################################
            average_score = np.mean(total_score)
            print("Epoch # ", epoch, "loss -> ", average_score)

        ####################################
        #### CHECK RESULTS
        #####################################

        y_test_pred, report_loss_each_epoch = round.run([predicted_y, loss],
                                                        feed_dict={x_batch: X_test, y_batch: y_test, train_x_y: False})

        ####################################
        #### GET ARGMAX OF THE TESTING DATA
        ####################################
        y_pred_max = np.argmax(y_test_pred, axis=1).astype(np.int32)
        y_actual_max = np.argmax(y_test, axis=1).astype(np.int32)

        ##########################
        ## PRINTS UP ACCURACY, PRECISON, F1-SCORE OF EACH LETTER
        ###########################
        print("loss =", report_loss_each_epoch)
        print("REPORT")
        print(metrics.classification_report(y_actual_max, y_pred_max))
        print("ACCURACY")
        print(metrics.accuracy_score(y_actual_max, y_pred_max))


####################################
##### Batches that are grabbed
##################################
def grab_portion(X_data, y_data, size_of_small_batches):
    idx = np.random.permutation(X_data.shape[0])

    for l in range(int(np.ceil(X_data.shape[0] / size_of_small_batches))):
        start = size_of_small_batches * l
        end = (l + 1) * size_of_small_batches
        yield X_data[idx[start:end], :, :, :], y_data[idx[start:end], :]


if __name__ == "__main__":
    #######################################
    ### READ IN DATA FOR THE FIRST NINE
    ########################################
    ImageData = namedtuple('ImageData', ['X_data', 'y_data'])
    fix_image_size = (32, 32)
    data = get_data('dataset5/A/', 9, False)

    #########################################
    ### SPLIT DATA SETS INTO TRAIN AND TEST SETS
    #########################################
    idx_train, idx_test = train_test_split(range(data.X_data.shape[0]), test_size=0.2, random_state=0)
    X_train = data.X_data[idx_train, :, :, :]
    X_test = data.X_data[idx_test, :, :, :]
    y_train = data.y_data[idx_train, :]
    y_test = data.y_data[idx_test, :]

    ##################################
    #### PRINT SHAPES FOR DATASETS
    ##################################
    print("Processing")
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    conv_neural_net(X_train, y_train, X_test, y_test, 0.001, 10, 25, False)

    ### RESET FOR 24 CLASS CLASSIFICATION
    tf.reset_default_graph()

    #######################################
    ### READ IN DATA FOR 24
    ########################################
    ImageData = namedtuple('ImageData', ['X_data', 'y_data'])
    fix_image_size = (32, 32)
    data = get_data('dataset5/A/', 24, True)

    #########################################
    ### SPLIT DATA SETS INTO TRAIN AND TEST SETS
    #########################################
    idx_train, idx_test = train_test_split(range(data.X_data.shape[0]), test_size=0.2, random_state=0)
    X_train = data.X_data[idx_train, :, :, :]
    X_test = data.X_data[idx_test, :, :, :]
    y_train = data.y_data[idx_train, :]
    y_test = data.y_data[idx_test, :]

    ##################################
    #### PRINT SHAPES FOR DATASETS
    ##################################
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    print("Processing")
    conv_neural_net(X_train, y_train, X_test, y_test, 0.001, 10, 25, True)
