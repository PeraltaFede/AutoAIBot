import io
import struct
import threading
# noinspection PyCompatibility
import socketserver
import os
import subprocess

import cv2
import numpy as np
import pygame
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# para no confundir a pycharm y usar las librerias se debe agregar asi si no sale el autocomplete
# TODO: ELIMINAR ESTA PARTE Y TESTEAR DESDE CMD.
try:
    # noinspection PyUnresolvedReferences
    from cv2 import cv2
except ImportError:
    pass


class NeuralNetwork(object):

    def __init__(self):
        self.X = tf.placeholder(tf.float32, [1, 120, 320, 1])

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

        W4 = tf.Variable(tf.truncated_normal([15 * 40 * M, N], stddev=0.1))
        B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
        W5 = tf.Variable(tf.truncated_normal([N, 3], stddev=0.1))
        B5 = tf.Variable(tf.constant(0.1, tf.float32, [3]))

        # The model
        stride = 1  # output is 120x320
        Y1 = tf.nn.relu(tf.nn.conv2d(self.X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
        stride = 2  # output is 60x160
        Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
        stride = 4  # output is 15x40
        Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

        # reshape the output from the third convolution for the fully connected layer
        YY = tf.reshape(Y3, shape=[-1, 15 * 40 * M])

        Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
        Ylogits = tf.matmul(Y4, W5) + B5
        self.Y = tf.nn.softmax(Ylogits)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        self.sess = tf.Session()
        # Restore variables from disk.
        saver.restore(self.sess, "./trained_model/model.ckpt")
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init)
        print("Modelo de red neuronal restaurado e iniciado.")

    def predict(self, image):
        global next_direction
        # Decodificar las imagenes a tensores
        # Expand dimensions since the model expects images to have shape: [1, None, None, 1]
        image_np_expanded = np.expand_dims(image, axis=0)
        image_np_expanded = np.expand_dims(image_np_expanded, axis=3)
        y_pred = self.sess.run(self.Y, feed_dict={self.X: image_np_expanded})
        next_direction = np.argmax(y_pred, 1)


class AutobotThread(socketserver.StreamRequestHandler):

    def handle(self):
        pygame.init()
        myfont = pygame.font.SysFont("monospace", 15)
        screen = pygame.display.set_mode((200, 200), 0, 24)
        label = myfont.render("Detenido", 1, (255, 255, 0))
        screen.blit(label, (0, 0))
        pygame.display.flip()

        print("Conexion establecida en Autobot: ", self.client_address)
        print('Autobot driving iniciado.\nUtiliza las teclas '
              'q, x para finalizar el programa.')

        try:
            global running, newimg, next_direction, roi
            while running:
                if newimg:
                    NeuralNetwork.predict(neuralnet, image=roi)
                    newimg = False
                    cv2.imshow('Computer vision', realimg)
                    key_input = pygame.key.get_pressed()
                    # ordenes de dos teclas
                    if next_direction == 1:
                        self.connection.send(b"DOR")
                        label = myfont.render("Derecha", 1, (255, 255, 0))
                        next_direction = -1

                    elif next_direction == 0:
                        self.connection.send(b"DOL")
                        label = myfont.render("Izquierda", 1, (255, 255, 0))
                        next_direction = -1

                        # ordenes una tecla
                    elif next_direction == 2:
                        self.connection.send(b"DOF")
                        label = myfont.render("Delante", 1, (255, 255, 0))
                        next_direction = -1

                    elif next_direction == -1:
                        print('detenido')
                        label = myfont.render("Detenido", 1, (255, 255, 0))
                        self.connection.send(b"DOS")

                    if key_input[pygame.K_x] or key_input[pygame.K_q]:
                        print("Detener el programa")
                        self.connection.send(b"DOE")
                        running = False
                        break

                    screen.fill((0, 0, 0))
                    screen.blit(label, (0, 0))
                    pygame.display.flip()


                else:
                    for _ in pygame.event.get():
                        _ = pygame.key.get_pressed()

            pygame.quit()
            cv2.destroyAllWindows()
        finally:
            print('Server finalizado en AutobotDriver')


class VideoThread(socketserver.StreamRequestHandler):

    name = "Video-Thread"

    def handle(self):
        global running, roi, realimg, newimg, neuralnet
        print("Conexion establecida video: ", self.client_address)
        running = True
        roi = 0
        # obtener las imagenes del stream una por una
        try:
            while running:
                # Read the length of the image as a 32-bit unsigned int. If the
                # length is zero, quit the loop
                image_len = struct.unpack('<L', self.rfile.read(struct.calcsize('<L')))[0]
                if not image_len:
                    print('Finalizado por Cliente')
                    break
                # Construct a stream to hold the image data and read the image
                # data from the connection

                image_stream = io.BytesIO()
                image_stream.write(self.rfile.read(image_len))

                image_stream.seek(0)
                jpg = image_stream.read()
                realimg = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                # region es Y, X
                roi = image[120:240, :]
                realimg = cv2.rectangle(realimg, (0, 120), (318, 238), (30, 230, 30), 1)
                newimg = True
        finally:
            print('Server finalizado en VideoStreaming')


class ThreadServer(object):

    def server_thread(host, port):
        server = socketserver.TCPServer((host, port), AutobotThread)
        server.serve_forever()

    def server_thread2(host, port):
        server = socketserver.TCPServer((host, port), VideoThread)
        server.serve_forever()

    server_ip = '192.168.0.13'
    if b"Fede Android" in subprocess.check_output("netsh wlan show interfaces"):
        server_ip = '192.168.43.59'
    print(server_ip)
    print("Iniciando Threads")
    video_thread = threading.Thread(target=server_thread2, args=(server_ip, 8000))
    video_thread.start()
    print("Video thread iniciado")
    autobot_thread = threading.Thread(target=server_thread, args=(server_ip, 8001))
    autobot_thread.start()
    print("Autobot thread iniciado")


if __name__ == '__main__':

    neuralnet = NeuralNetwork()
    running = True
    roi = None
    realimg = None
    newimg = False
    next_direction = -1
    # Start new Threads
    e1 = cv2.getTickCount()
    while running:
        pass
    e2 = cv2.getTickCount()
    # calcular el total de streaming
    time0 = (e2 - e1) / cv2.getTickFrequency()
    print("Duracion del streaming:", time0)
    os.system('pause')
