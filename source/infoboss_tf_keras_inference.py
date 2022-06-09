import numpy as np
from tensorflow.keras.applications.resnet50 import *
import cv2
import os
import time
import tensorflow as tf

from tensorflow.keras.models import load_model

recent_model = '/home/systartup/data/' + '322-3.6308.hdf5'
model = load_model(recent_model)

input_name = 'PD_xylem_y0001.npz'
fname = '/home/systartup/data/' + input_name

npzfile = np.load(fname, allow_pickle=True)
Xh_train = npzfile['Xh_train']
Xh_val = npzfile['Xh_val']
Xh_test = npzfile['Xh_test']
Xv_train = npzfile['Xv_train']
Xv_val = npzfile['Xv_val']
Xv_test = npzfile['Xv_test']
Y_train = npzfile['y_train']
Y_val = npzfile['y_val']
Y_test = npzfile['y_test']
lmbda = npzfile['lmbda']
shift = npzfile['shift']

X_train = list()
X_train.append(Xh_train)
X_train.append(Xv_train)

X_test = list()
X_test.append(Xh_test)
X_test.append(Xv_test)

X_val = list()
X_val.append(Xh_val)
X_val.append(Xv_val)

Y_train = Y_train.astype(np.float32).reshape((-1,1))
Y_test = Y_test.astype(np.float32).reshape((-1,1))


print(len(Xh_train), len(Xh_val), len(Xh_test))

start_time = time.time()

for i in range(len(Xh_train)):
    X_train = list()
    X_train.append(Xh_train[i:i+1])
    X_train.append(Xv_train[i:i+1])
    p_time = time.time()
    Y_pred = model.predict(X_train)
    c_time = time.time()
    sec = c_time - p_time
    gps = "GPS : %0.1f" % (1 / sec)
    print(i, gps)
# Y_pred = model.predict(X_train)

end_time = time.time()
sec = end_time - start_time
gps = "GPS : %0.1f" % (len(Xh_train) / sec)
print('# of data = ', len(Xh_train))
print('spent time = ', sec, '(second)')
print('gene per second = ', gps)

# p_time = 0
# while cv2.waitKey(1) < 0:
#     for img_id in os.listdir('/home/systartup/data/img'):
#
#         try:
#             image = cv2.imread(os.path.join('/home/systartup/data/img', img_id))
#             src = image.copy()
#             image = cv2.resize(image, (224, 224))
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         except:
#             continue
#
#         image = np.expand_dims(image, axis=0)
#         image = preprocess_input(image)
#
#         c_time = time.time()
#         sec = c_time - p_time
#         p_time = c_time
#         fps = 1 / (sec)
#         str = "FPS : %0.1f" % fps
#         outputs = model.predict(image)
#         predicted = classes[outputs.argmax()]
#         print(str)
#
#         # cv2.putText(src, str + " " + predicted, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=3)
#         # cv2.imshow('sample', src)
#         key = cv2.waitKey(1)
#         if key % 256 == 27:  # esc stop
#             break
#
#     cv2.destroyAllWindows()