import cv2
import math
import tensorflow as tf

# para no confundir a pycharm y usar las librerias se debe agregar asi si no sale el autocomplete
# TODO: ELIMINAR ESTA PARTE Y TESTEAR DESDE CMD
try:
    # noinspection PyUnresolvedReferences
    from cv2 import cv2
except ImportError:
    pass


class ImageHandler(object):
    def __init__(self):
        # hacer una fila con los nombres de imagenes incluyendo todas las imagenes de un directorio
        self.filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once("./training_images/*.jpg"), shuffle=True)

        # Leer toda la imagen jpg en este caso de la dimension establecida en el agent
        self.image_reader = tf.WholeFileReader()

    def getafile(self):
        return self.image_reader.read(self.filename_queue)

print('Cargando datos para entrenamiento...')
e0 = cv2.getTickCount()

# input X: 320x120 imagenes en escala a grises, ke = the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 320, 120, 1])
# ground truth se pone aca
Y_ = tf.placeholder(tf.float32, [None, 3])
# tasa de aprendizaje
lr = tf.placeholder(tf.float32)
# Probabilidad de dejar un nodo funcionando = 1.0 para los examenes (no dropout) y 0.75 durante aprendizaje
pkeep = tf.placeholder(tf.float32)

# tres capas convolucionales con sus canales totales, y una capa totalmente conectada
K = 6  # profundidad de la primera capa
L = 12  # profundidad de la segunda capa
M = 24  # profundidad de la tercera capa
N = 200  # dimension de la capa totalmente conectada

# WX = variable hecha de una matriz de 6,6,1,XX dimesion y cada valor iniciado en random de 0 a 1 (truncated)
# BX = variable hecha de una matriz de dimension K con valor constante de 0.1
W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 canal del anterior, K canales de salida
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

# WX = variable de matriz X Y truncated
W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, 3], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [3]))

# El modelo
stride = 1  # salida de la misma dimension
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # salida de la dimension/2
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # salida de la dimension anterior/2
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reformateo de la salida de la 3ra convolucion para la ultima capa
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

# ultima capa con activacion ReLU, y aplicando dropouts para evitar overfittings
Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
YY4 = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(YY4, W5) + B5
# salida ultima
Y = tf.nn.softmax(Ylogits)

# funcion de perdida cross-entropy (= -sum(Y_i * log(Yi)) ), normalizado
# TensorFlow provee el softmax_cross_entropy_with_logits para evitar problemas numericos de estabilidad con log(0)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)  # una sola imagen a la vez sino multiplicar por el batch al final *100

# exactitud del modelo entrado, entre 0 (malo) y 1 (lo mejor)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# paso de entrenamiento, la tasa de aprendizaje es un estado p completar
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
t0 = (cv2.getTickCount() - e0)/cv2.getTickFrequency()
print('Carga de datos correcta en tiempo ', t0)
# iniciar
print('Iniciando la sesion ', t0)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


def training_step(i, update_test_data, update_train_data):

    # Leer un archivo completo de la fila
    label_name, image_file = imghd.getafile()

    # Decodificar una imagen JPG a una imagen que se pueda usar en tensorflow
    image = tf.image.decode_jpeg(image_file)
    label = tf.zeros(3, tf.float32)  # todo: elegir el nro de posicion de label real
    print(label_name)
    label[label_name[5]] = 1

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 5000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    # compute training values for visualisation
    if update_train_data:
        a, c, im, w, b = sess.run(train_step, {X: image, Y_: label, pkeep: 1.0})
        print("Accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")

    # compute test values for visualisation
    if update_test_data:
        print("no data for testing")
        # a, c, im = sess.run([accuracy, cross_entropy, It], {X: mnist.test.images, Y_: mnist.test.labels, pkeep:
        # 1.0})
        # print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test
        # accuracy:" + str(a) + " test loss: " + str(c))
        # the backpropagation training step"""
    sess.run(train_step, {X: image, Y_: label, lr: learning_rate, pkeep: 0.75})


# tiempo de inicio
e1 = cv2.getTickCount()
imghd = ImageHandler()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

try:
    for j in range(10000 + 1):
        training_step(j, j % 100 == 0, j % 20 == 0)
finally:
    coord.request_stop()
    coord.join(threads)
    saver = tf.train.Saver()
    save_path = saver.save(sess, "/model.ckpt")
    print("Model saved in file: %s" % save_path)
    t0 = (cv2.getTickCount() - e1) / cv2.getTickFrequency()
    print("Tiempo de entrenamiento: ", t0)
    sess.close()
