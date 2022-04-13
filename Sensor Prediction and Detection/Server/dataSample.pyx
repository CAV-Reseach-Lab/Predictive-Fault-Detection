import numpy as np
cimport numpy as np
import sensor_probe_server as server
import time

ctypedef np.float32_t DTYPE_t

def dataSample(myServer):
    """Samples the incoming data and returns a 240 samples of data worth about 2.4 seconds of sensor data"""
    cdef int start, end, diff
    start = time.time()
    cdef np.ndarray[np.float64_t, ndim=2] dataFrame = np.zeros((1,3), dtype=np.float64)
    cdef np.ndarray[np.float64_t] gyro = np.zeros((3), dtype=np.float64) 
    cdef np.ndarray[np.float64_t] acc = np.zeros((3), dtype=np.float64)
    cdef float enc = 0.0
    #cdef fieldsName = ['Gyroscope', 'Accelerometer', 'Encoder']
    cdef int i = 0
    cdef float avgGyro, avgAcc, tmpEnc 
    cdef np.ndarray[np.float64_t] tmpArr = np.zeros((3), dtype=np.float64)
    try:
        for i in range(239):
            myServer.recieve(gyro, acc, enc)
            avgGyro = (gyro[0] + gyro[1] + gyro[2])/3.0
            avgAcc = (acc[0] + acc[1] + acc[2])/3.0
            tmpArr[0] = avgGyro
            tmpArr[1] = avgAcc
            tmpArr[2] = enc
            #print(tmpArr)
            #with open("./sampleData.csv", 'a') as csvFile:
            #    writer = csv.DictWriter(csvFile, fieldsName)
            #    writer.writerow({'Gyroscope': tmpArr[0], 'Accelerometer': tmpArr[1], 'Encoder': tmpArr[2]})
            dataFrame = np.vstack((dataFrame, tmpArr))
    #print("Processing")
    #dataFrame = np.reshape(dataFrame,(1, 3, 240)) 
    #tensor = from_numpy(dataFrame)
    except server.ClientStop as e:
        raise e
    end = time.time()
    diff = end - start
    print("Data Sample Time: ", diff)
    return dataFrame