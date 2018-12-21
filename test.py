import numpy as np
from multiprocessing import Process, Queue


if __name__ == '__main__':
    def prepareDataThread( dataQueue, numpyImages, numpyGT, params):
        print ('55')

        dataQueue.put(5)
    numpyImages = np.zeros([5,5,5,5])
    numpyGT = np.zeros([5,5,5,5])
    dataQueue = Queue(50)  # max 50 images in queue
    dataPreparation = [None] *5
    params = []
    prepareDataThread(dataQueue, numpyImages, numpyGT, params)
    # thread creation
    # for proc in range(0,5):
    #     dataPreparation[proc] = Process(target=prepareDataThread, args=(dataQueue, numpyImages, numpyGT, params))
    #     dataPreparation[proc].daemon = True
    #     dataPreparation[proc].start()
