import numpy as np
import cv2
import threading
# noinspection PyCompatibility
import socketserver
import os


# para no confundir a pycharm y usar las librerias se debe agregar asi si no sale el autocomplete
# TODO: ELIMINAR ESTA PARTE Y TESTEAR DESDE CMD. debe funcionar SOLO recibiendo imagenes y enviando la direccion
try:
    # noinspection PyUnresolvedReferences
    from cv2 import cv2
except ImportError:
    pass


class NeuralNetwork(object):

    def __init__(self):
        self.model = None

    def create(self):
        # layer_size = np.int32([38400, 32, 4])
        # self.model.create(layer_size)
        self.model = cv2.ml.ANN_MLP_load('mlp_xml/mlp.xml')

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)


class RCControl(object):
    def __init__(self):
        print('Autobot iniciado')

    def steer(self, prediction):
        if prediction == 2:
            print("Forward")
            return 'DOF'
        elif prediction == 0:
            print("Left")
            return 'DOL'
        elif prediction == 1:
            print("Right")
            return 'DOR'
        else:
            self.stop()

    @staticmethod
    def stop():
        print("Stop")
        return 'DOS'


# class SendDataHandler(socketserver.StreamRequestHandler):

#    def handle(self):

class VideoStreamHandler(socketserver.StreamRequestHandler):

    # create neural network
    model = NeuralNetwork()
    model.create()

    rc_car = RCControl()
    
    def handle(self):
        stream_bytes = ' '
        # stream video frames one by one
        try:
            while True:
                stream_bytes += self.rfile.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    gray = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    # lower half of the image
                    half_gray = gray[120:240, :]

                    cv2.imshow('image', image)

                    # reshape image
                    image_array = half_gray.reshape(1, 38400).astype(np.float32)

                    # neural network makes prediction
                    prediction = self.model.predict(image_array)
                    self.connection.send(self.rc_car.steer(prediction))

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.connection.send(self.rc_car.stop())
                        break

            cv2.destroyAllWindows()

        finally:
            print("Finalizado con exito")
            os.system('pause')


class ThreadServer(object):
    def server_thread(host, port):
        server = socketserver.TCPServer((host, port), VideoStreamHandler)
        server.serve_forever()

#    def server_thread2(host, port):
#        server = socketserver.TCPServer((host, port), SendDataHandler)
#        server.serve_forever()

#    distance_thread = threading.Thread(target=server_thread2, args=('192.168.0.9', 8002))
#    distance_thread.start()
    video_thread = threading.Thread(target=server_thread, args=('192.168.0.9', 8000))
    video_thread.start()
    print("Inicializando stream.\nEsperando datos...")

if __name__ == '__main__':
    ThreadServer()
