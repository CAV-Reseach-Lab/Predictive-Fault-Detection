#Server Side of the sensor probe 
import time
import struct
import numpy as np
from torch import from_numpy
from libs.stream_struct import StructStream, StreamError

class ClientStop(Exception):
    pass
class server():
    def __init__(self, clientIP):
        self.clientUri = 'tcpip://'+clientIP+':18001'
        self.prevCon = False
        self.dataPacket = struct.Struct("7f")
        self.sampleRate = 100.0
        self.sampleTime = 1/self.sampleRate

        self.gyroscope = [0.0,0.0,0.0]
        self.accelerometer = [0.0,0.0,0.0]
        self.encoder = 0

        self.bufferSize = self.dataPacket.size
        self.buffer = self.dataPacket.pack(self.gyroscope[0], self.gyroscope[1], self.gyroscope[2], self.accelerometer[0], self.accelerometer[1], self.accelerometer[2], self.encoder)
        self.server = StructStream(self.clientUri, 's', self.bufferSize, self.bufferSize)
        self.connected = self.server.connected

    @staticmethod
    def elapsedTime(startTime):
        return time.time() - startTime

    def connect(self, timeout=60):
        startTime = time.time()
        while (not self.server.connected) and (self.elapsedTime(startTime) < timeout):
            self.server.checkConnection()
        
        if not self.server.connected:
            print("No clients connected in ", timeout, " seconds.")
        else:
            print("A client connected!")
        self.connected = self.server.connected
        return self.server.connected
    
    def recieve(self):
        '''Recieve data from the client'''
        try:
            self.buffer, bytesRecieved = self.server.receive(self.buffer)
            if bytesRecieved < len(self.buffer):
                print("Client has stopped sending data")

                self.gyroscope = [0.0,0.0,0.0]
                self.accelerometer = [0.0,0.0,0.0]
                self.encoder = 0
                raise ClientStop
            self.gyroscope[0], self.gyroscope[1], self.gyroscope[2], self.accelerometer[0], self.accelerometer[1], self.accelerometer[2], self.encoder = self.dataPacket.unpack(self.buffer)
            return self.gyroscope, self.accelerometer, self.encoder
        except StreamError as e:
            #print(e.get_error_message())
            print("Client lost connection with server. Disconnecting.")
            self.server.terminate()
            raise e
        except ClientStop as e:
            raise e
        return self.gyroscope, self.accelerometer, self.encoder
    def dataSample(self):
        '''Samples the incoming data and returns a 240 samples of data worth about 2.4 seconds of sensor data"'''
        data = np.array([[0.0,0.0,0.0], [0.0,0.0,0.0], 0],dtype=object)
        dataFrame = np.array([[0,0,0]], dtype = np.float32)
        try:
            for i in range(239):
                data[0], data[1], data[2] = self.recieve()
                avgGyro = (data[0][0] + data[0][1] + data[0][2])/3.0
                avgAcc = (data[1][0] + data[1][1] + data[1][2])/3.0
                tmpArr = [avgGyro, avgAcc, data[2]]
                dataFrame = np.vstack((dataFrame, tmpArr))
        except ClientStop as e:
            raise e
        dataFrame = np.reshape(dataFrame,(1, 3, 240)) 
        tensor = from_numpy(dataFrame)
        return tensor, data
    def terminate(self):
        self.server.terminate()
        
        