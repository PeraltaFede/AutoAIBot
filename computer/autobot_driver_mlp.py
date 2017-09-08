import io
import os
# noinspection PyCompatibility
import socketserver
import struct
import subprocess
import threading

import cv2
import numpy as np
import pygame
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NeuralNetwork(object):
    def __init__(self):
        # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
        self.X = tf.placeholder(tf.float32, [1, 120, 320, 1])

        # five layers and their number of neurons (tha last layer has 10 softmax neurons)
        L = 400
        M = 100
        N = 50
        O = 20
        # Weights initialised with small random values between -0.2 and +0.2
        # When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones
        # ([K])/10
        W1 = tf.Variable(tf.truncated_normal([38400, L], stddev=0.1))  # 120*320 =
        B1 = tf.Variable(tf.ones([L]) / 10)
        W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
        B2 = tf.Variable(tf.ones([M]) / 10)
        W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
        B3 = tf.Variable(tf.ones([N]) / 10)
        W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
        B4 = tf.Variable(tf.ones([O]) / 10)
        W5 = tf.Variable(tf.truncated_normal([O, 3], stddev=0.1))
        B5 = tf.Variable(tf.zeros([3]))

        # The model, with dropout at each layer
        XX = tf.reshape(self.X, [-1, 120 * 320])

        Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)

        Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)

        Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)

        Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)

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
        next_direction = np.argmax(y_pred, 1)[0]


class AutobotThread(socketserver.StreamRequestHandler):
    def handle(self):
        neuralnet = NeuralNetwork()
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
            current_direction = -1
            global running, newimg, next_direction, roi
            while running:
                if newimg:
                    neuralnet.predict(image=roi)
                    newimg = False
                    cv2.imshow('Computer vision', realimg)
                    # ordenes de dos teclas
                    if next_direction == 1 and current_direction != 1:
                        self.connection.send(b"DOR")
                        label = myfont.render("Derecha", 1, (255, 255, 0))
                        next_direction = -1
                        current_direction = 1

                    elif next_direction == 0 and current_direction != 0:
                        self.connection.send(b"DOL")
                        label = myfont.render("Izquierda", 1, (255, 255, 0))
                        next_direction = -1
                        current_direction = 0

                        # ordenes una tecla
                    elif next_direction == 2 and current_direction != 2:
                        self.connection.send(b"DOF")
                        label = myfont.render("Delante", 1, (255, 255, 0))
                        next_direction = -1
                        current_direction = 2

                    elif next_direction == -1 and current_direction != -1:
                        print('detenido')
                        label = myfont.render("Detenido", 1, (255, 255, 0))
                        self.connection.send(b"DOS")
                        current_direction = -1

                    key_input = pygame.key.get_pressed()
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
    autobot_thread.join()
    video_thread.join()


if __name__ == '__main__':
    running = True
    roi = None
    realimg = None
    newimg = False
    next_direction = -1
    # Start new Threads
    e1 = cv2.getTickCount()
