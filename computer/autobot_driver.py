import io
import struct
import threading
# noinspection PyCompatibility
import socketserver
import os

import cv2
import numpy as np
import pygame
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# para no confundir a pycharm y usar las librerias se debe agregar asi si no sale el autocomplete
# TODO: ELIMINAR ESTA PARTE Y TESTEAR DESDE CMD. debe funcionar SOLO recibiendo imagenes y enviando la direccion
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
        y_pred = self.sess.run(self.Y, feed_dict={self.X: image})
        print(y_pred)


class AutobotThread(socketserver.StreamRequestHandler):

    def handle(self):
        myfont = pygame.font.SysFont("monospace", 15)
        pygame.init()
        screen = pygame.display.set_mode((200, 200), 0, 24)
        screen.set_caption("Teclado")
        label = myfont.render("Detenido", 1, (255, 255, 0))
        screen.blit(label, (100, 100))

        print("Conexion establecida en Autobot: ", self.client_address)
        print('Empieza a coleccionar datos manejando.\nUtiliza las flechas '
              'para manejar. Solo se guardan los datos Arriba, Izq., Der.')

        try:
            global running, saved_frame, roi, newimg
            saved_frame = 0
            currentstate = 4  # 0 = izquierda ; 1 = derecha; 2 = delante ; 3 = reversa; 4 = stop
            while running:
                if newimg:
                    newimg = False
                    cv2.imshow('Computer vision', realimg)
                    key_input = pygame.key.get_pressed()
                    # ordenes de dos teclas
                    if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
                        print("Delante Derecha")
                        cv2.imwrite('training_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 1), roi)
                        if not currentstate == 1:
                            self.connection.send(b"DOR")
                            currentstate = 1
                            label = myfont.render("Delante Derecha", 1, (255, 255, 0))
                            screen.blit(label, (100, 100))
                        saved_frame += 1

                    elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
                        print("Delante Izquierda")
                        cv2.imwrite('training_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 0), roi)
                        if not currentstate == 0:
                            self.connection.send(b"DOL")
                            currentstate = 0
                            label = myfont.render("Delante Izquierda", 1, (255, 255, 0))
                            screen.blit(label, (100, 100))
                        saved_frame += 1

                        # ordenes una tecla
                    elif key_input[pygame.K_UP]:
                        print("Delante")
                        cv2.imwrite('training_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 2), roi)
                        if not currentstate == 2:
                            self.connection.send(b"DOF")
                            currentstate = 2
                            label = myfont.render("Delante", 1, (255, 255, 0))
                            screen.blit(label, (100, 100))
                        saved_frame += 1

                    elif key_input[pygame.K_RIGHT]:
                        print("Derecha")
                        cv2.imwrite('training_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 1), roi)
                        if not currentstate == 1:
                            self.connection.send(b"DOR")
                            currentstate = 1
                            label = myfont.render("Derecha", 1, (255, 255, 0))
                            screen.blit(label, (100, 100))
                        saved_frame += 1

                    elif key_input[pygame.K_LEFT]:
                        print("Izquierda")
                        cv2.imwrite('training_images/frame{:>05}-{:>01}.jpg'.format(total_frame, 0), roi)
                        if not currentstate == 0:
                            self.connection.send(b"DOL")
                            currentstate = 0
                            label = myfont.render("Izquierda", 1, (255, 255, 0))
                            screen.blit(label, (100, 100))
                        saved_frame += 1

                    elif key_input[pygame.K_DOWN]:
                        if not currentstate == 3:
                            self.connection.send(b"DOB")
                            currentstate = 3
                            label = myfont.render("Reversa", 1, (255, 255, 0))
                            screen.blit(label, (100, 100))
                        print("Reversa")

                    elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                        print("Detener el programa")
                        label = myfont.render("Finalizar programa", 1, (255, 255, 0))
                        screen.blit(label, (100, 100))
                        self.connection.send(b"DOE")
                        running = False
                        break

                    else:
                        if not currentstate == 4:
                            print('Esperando ordenes')
                            label = myfont.render("Detenido", 1, (255, 255, 0))
                            screen.blit(label, (100, 100))
                            currentstate = 4
                            self.connection.send(b"DOS")

            pygame.quit()
            cv2.destroyAllWindows()
        finally:
            print('Server finalizado en AutobotDriver')


class VideoThread(socketserver.StreamRequestHandler):

    name = "Video-Thread"

    def handle(self):
        global running, roi, total_frame, realimg, newimg, neuralnet
        total_frame = 0
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
                NeuralNetwork.predict(neuralnet, image=roi)
                total_frame += 1
        finally:
            print('Server finalizado en VideoStreaming')


class ThreadServer(object):

    def server_thread(host, port):
        server = socketserver.TCPServer((host, port), AutobotThread)
        server.serve_forever()

    def server_thread2(host, port):
        server = socketserver.TCPServer((host, port), VideoThread)
        server.serve_forever()

    print("iniciando esto sin que nadie le haya pedido")
    video_thread = threading.Thread(target=server_thread2, args=('192.168.0.13', 8000))
    video_thread.start()
    print("Video thread started")
    autobot_thread = threading.Thread(target=server_thread, args=('192.168.0.13', 8001))
    autobot_thread.start()
    print("Autobot thread started")


if __name__ == '__main__':

    neuralnet = NeuralNetwork()
    running = True
    saved_frame = 0
    total_frame = 0
    roi = None
    realimg = None
    newimg = False
    # global running, saved_frame, total_frame, roi, realimg, newimg
    # Start new Threads
    e1 = cv2.getTickCount()
    ThreadServer()
    ThreadServer.video_thread.join()
    ThreadServer.autobot_thread.join()
    e2 = cv2.getTickCount()
    # calcular el total de streaming
    time0 = (e2 - e1) / cv2.getTickFrequency()
    print("Duracion del streaming:", time0)
    print('Total cuadros           : ', total_frame)
    print('Total cuadros guardados : ', saved_frame)
    print('Total cuadros desechados: ', total_frame - saved_frame)
    os.system('pause')
