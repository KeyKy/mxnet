import os
import sys
sys.path.insert(0, '/data2/obj_detect/mxnet/python')
import mxnet as mx
import numpy as np
import argparse
import time


if __name__ == '__main__':
    net_path = '/data2/obj_detect/person-exist/symbols/mobilenet-symbol.json'
    sym = mx.sym.load(net_path)
    mod = mx.mod.Module(
            context = mx.cpu(),
            symbol = sym,
            data_names = ['data',],
            label_names = ['softmax_label',]
    )
    mod.bind(data_shapes=[('data', (1, 3, 224, 224))],
             label_shapes=[('softmax_label', (1,))],
             for_training=False)
    mod.init_params(initializer=mx.init.Normal(0.001))

    data = np.random.uniform(-1, 1, (1000, 3, 224, 224))
    data = mx.nd.array(data)
    data_iter = mx.io.NDArrayIter(data=data, batch_size=1)

    t_start = time.time()
    t_list = []
    for i in range(1000):
        data_iter.reset()
        for batch in data_iter:
            t1 = time.time()
            mod.forward(batch, is_train=False)
            for output in mod.get_outputs():
                output.wait_to_read()
            t2 = time.time()
            print('batch time {}'.format(t2-t1))
    t_end = time.time()

    print t_list
    print t_end - t_start
