import cv2
import math
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# para no confundir a pycharm y usar las librerias se debe agregar asi si no sale el autocomplete
# TODO: ELIMINAR ESTA PARTE Y TESTEAR DESDE CMD
try:
    # noinspection PyUnresolvedReferences
    from cv2 import cv2
except ImportError:
    pass

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

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
L = 400
M = 100
N = 50
O = 20
# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([38400, L], stddev=0.1))  # 120*320 =
B1 = tf.Variable(tf.ones([L])/10)
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.ones([M])/10)
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.ones([N])/10)
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B4 = tf.Variable(tf.ones([O])/10)
W5 = tf.Variable(tf.truncated_normal([O, 3], stddev=0.1))
B5 = tf.Variable(tf.zeros([3]))

# The model, with dropout at each layer
XX = tf.reshape(X, [-1, 120*320])

Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)

Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)

Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)

Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)

Ylogits = tf.matmul(Y4d, W5) + B5
Y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
t0 = (cv2.getTickCount() - e0)/cv2.getTickFrequency()
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
        # learning rate decay
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

        # al terminar se pide unir los threads y finalizar
    coord.request_stop()
    coord.join(threads)

    saver = tf.train.Saver()
    save_path = saver.save(sess, "./trained_model/model.ckpt")
    print("Model saved in file: %s" % save_path)
    t0 = (cv2.getTickCount() - e1) / cv2.getTickFrequency()
    print("Tiempo de entrenamiento: ", t0)
    os.system('pause')
