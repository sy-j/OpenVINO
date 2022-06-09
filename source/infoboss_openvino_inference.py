import argparse
import pdb
import sys
import os
import time
import cv2
import numpy as np
from openvino.inference_engine import IECore


from tensorflow.keras.models import load_model


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-m', '--model', required=True, type=str,
                      help='Required. Path to an .xml or .onnx file with a trained model.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, MYRIAD, HDDL or HETERO: '
                           'is acceptable. The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.')
    return parser.parse_args()


def main():
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


    input_shapes = [sl.shape[1:] for sl in X_train]
    print(input_shapes)



    args = parse_args()
    # ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
    ie = IECore()

    # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
    # (.xml and .bin files) or (.onnx file)
    net = ie.read_network(args.model)
    # ---------------------------Step 3. Configure input & output----------------------------------------------------------
    # Get names of input and output blobs
    input_blob = next(iter(net.input_info))
    print(net.input_info)
    print(net.input_info)
    print(iter(net.input_info))
    print(input_blob)
    out_blob = next(iter(net.outputs))
    # Set input and output precision manually
    net.input_info[input_blob].precision = 'U8'
    net.outputs[out_blob].precision = 'FP32'

    # Get a number of classes recognized by a model
    # num_of_classes = max(net.outputs[out_blob].shape)

    # ---------------------------Step 4. Loading model to the device-------------------------------------------------------
    exec_net = ie.load_network(network=net, device_name=args.device)

    # ---------------------------Step 5. Create infer request--------------------------------------------------------------
    # load_network() method of the IECore class with a specified number of requests (default 1) returns an ExecutableNetwork
    # instance which stores infer requests. So you already created Infer requests in the previous step.

    # ---------------------------Step 6. Prepare input---------------------------------------------------------------------
    # import pdb
    # pdb.set_trace()
    p_time = 0




    # image = original_image.copy()
    # _, _, h, w = net.input_info[input_blob].input_data.shape

    # if image.shape[:-1] != (h, w):
    #     image = cv2.resize(image, (w, h))

    # # Change data layout from HWC to CHW
    # image = image.transpose((2, 0, 1))
    # # Add N dimension to transform to NCHW
    # image = np.expand_dims(image, axis=0)


    start_time = time.time()
    # ---------------------------Step 7. Do inference----------------------------------------------------------------------
    # res = exec_net.infer(inputs={input_blob: Xh_train})  # dictinary

    for i in range(len(Xh_train)):
        p_time = time.time()
        res = exec_net.infer(inputs={'x': Xh_train[i], 'x_1': Xv_train[i]})  # dictinary
        # print(res)
        c_time = time.time()
        sec = c_time - p_time
        gps = "GPS : %0.1f" % (1 / sec)
        print(i, gps)

    # ---------------------------Step 8. Process output--------------------------------------------------------------------


    end_time = time.time()
    sec = end_time - start_time
    gps = "GPS : %0.1f" % (len(Xh_train) / sec)
    print('# of data = ', len(Xh_train))
    print('spent time = ', sec, '(second)')
    print('gene per second = ', gps)


    # Generate a label list
    res = res[out_blob]  # (1, 1000)
    # Change a shape of a numpy.ndarray with results to get another one with one dimension
    # probs = res.reshape(num_of_classes)  # (1000,)
    # Get an array of args.number_top class IDs in descending order of probability


    # top = probs.argmax()


if __name__ == '__main__':
    sys.exit(main())

