import os

import cv2
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('Cargando datos para entrenamiento...')
e0 = cv2.getTickCount()

# Carga de thread para imagenes

# De este directorio se debe quitar el bit nro 20 + nombre hasta el valor del label
train_img_dir = tf.train.match_filenames_once(".\\training_images\\*.jpg")
test_img_dir = tf.train.match_filenames_once(".\\test_images\\*.jpg")

# Hacer una fila de los archivos a abrir
filename_queue = tf.train.string_input_producer(train_img_dir, shuffle=False, capacity=10000)
test_filename_queue = tf.train.string_input_producer(test_img_dir, shuffle=False, capacity=1000)

name_queue = tf.train.string_input_producer(train_img_dir, shuffle=False, capacity=10000)
test_name_queue = tf.train.string_input_producer(test_img_dir, shuffle=False, capacity=1000)

# imagereader es un lector que lee un archivo completo a la vez
image_reader = tf.WholeFileReader()
# leyendo un archivo se obtienen los datos de nombre y datos de la img
_, image_file = image_reader.read(filename_queue)
__, test_image_file = image_reader.read(test_filename_queue)
name_file, ___ = image_reader.read(name_queue)
test_name_file, ____ = image_reader.read(test_name_queue)

# Decodificar las imagenes a tensores
image = tf.image.decode_jpeg(image_file, channels=1)
test_image = tf.image.decode_jpeg(test_image_file, channels=1)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [1, 120, 320, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [1, 3])
# variable learning rate
lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (the last layer has 3 softmax neurons)
K = 6  # first convolutional layer output depth
L = 12  # second convolutional layer output depth
M = 24  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

W4 = tf.Variable(tf.truncated_normal([6 * 16 * M, N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, 3], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [3]))

# The model
stride = 2  # output is 120x320
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 60x160
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 5  # output is 15x40
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 6 * 16 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
YY4 = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(YY4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of one image
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
t0 = (cv2.getTickCount() - e0) / cv2.getTickFrequency()
print('Carga de datos correcta en tiempo ', t0)
# iniciar
print('Iniciando la sesion ')
# tiempo de inicio
e1 = cv2.getTickCount()

# Inicializar tf
init = (tf.global_variables_initializer(), tf.local_variables_initializer())

# Comenzar una nueva sesion.
with tf.Session() as sess:
    sess.run(init)
    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0

    # coordinador para iniciar un threading de todos los jpg
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    trainacc = 0.0
    canttrain = 0.0
    trainlos = 0
    testacc = 0
    canttest = 0
    testlos = 0
    promtestacc = 0
    promtestlos = 0
    promacc = 0
    promlos = 0
    for i in range(sess.run(filename_queue.size())):
        name_tensor = sess.run([name_file])[0].decode('utf-8')[29]
        full_name = "ninguno"
        if int(name_tensor) == 0:
            full_name = "Izquierda"
        elif int(name_tensor) == 1:
            full_name = "Derecha"
        elif int(name_tensor) == 2:
            full_name = "Adelante"
        imageugh2 = image.eval()
        imageugh = np.asarray(imageugh2, dtype=np.uint8)
        imageugh = cv2.putText(imageugh, full_name, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imshow('Lol', imageugh)
        cv2.waitKey()

    # al terminar se pide unir los threads y finalizar
    coord.request_stop()
    coord.join(threads)

    saver = tf.train.Saver()
    save_path = saver.save(sess, "./trained_model/model.ckpt")
    print("Model saved in file: %s" % save_path)
    t0 = (cv2.getTickCount() - e1) / cv2.getTickFrequency()
    print("Tiempo de entrenamiento: ", t0)
    os.system('pause')

"""

        # la totalidad de imagenes corre 4 veces y aqui se hace el training
        name_tensor = sess.run([name_file])[0].decode('utf-8')[29]
        y_ = np.zeros([1, 3])
        y_[0, int(name_tensor)] = 1
        image_tensor = sess.run([image])
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)
        if i % 10 == 0 and i != 0:

            if i % 100 == 0:
                trainacc = 0.0
                canttrain = 0.0
                trainlos = 0

            a, c = sess.run([accuracy, cross_entropy], {X: image_tensor, Y_: y_, pkeep: 1.0})
            os.system('cls')
            print(str(i) + ": TRAIN: accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
            canttrain = canttrain + 1
            trainacc = trainacc + a
            promacc = trainacc / canttrain
            trainlos = trainlos + c
            promlos = trainlos / canttrain
            print("Last 100 accuracy: {:f}\nLast 100 loss: {:f}".format(promacc, promlos))
            print("Total Test accuracy: {:f}\nTotal Test loss: {:f}".format(promtestacc, promtestlos))

        if i % 50 == 0 and i != 0:
            test_name_tensor = sess.run([test_name_file])[0].decode('utf-8')[25]
            print(test_name_tensor)
            test_y_ = np.zeros([1, 3])
            test_y_[0, int(test_name_tensor)] = 1
            test_image_tensor = sess.run([test_image])
            a, c = sess.run([accuracy, cross_entropy], {X: test_image_tensor, Y_: test_y_, pkeep: 1.0})
            os.system('cls')
            print(str(i) + ": TEST : accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
            canttest = canttest + 1
            testacc += a
            promtestacc = testacc / canttest
            testlos += c
            promtestlos = testlos / canttest

            print("Last 100 accuracy: {:f}\nLast 100 loss: {:f}".format(promacc, promlos))
            print("Total Test accuracy: {:f}\nTotal Test loss: {:f}".format(promtestacc, promtestlos))

        # the backpropagation training step
        sess.run(train_step, {X: image_tensor, Y_: y_, lr: learning_rate, pkeep: 0.75})

"""
