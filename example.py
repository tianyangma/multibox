#!/usr/bin/python

from det_coder import Coder
import numpy as np
import caffe
import utils
import matplotlib.pyplot as plt


def prepare_inputs(image, gt):

    probs, targets = coder.encode(gt)
    assert probs.shape[0] == 2
    assert targets.shape[0] == 4

    # only put non zero weights on positive samples.
    pred_weights = labels = probs[0, :] > 0.5
    pred_weights = pred_weights[np.newaxis, np.newaxis, :, np.newaxis]

    # put equal weights on all samples.
    label_weights = np.ones(pred_weights.shape)

    labels = pred_weights
    targets = targets[np.newaxis, :, :, np.newaxis]

    inputs = {
        'data': image,
        'labels': labels,
        'targets': targets,
        'pred_weights': pred_weights,
        'label_weights': label_weights,
    }

    return inputs


def unpack_outputs(blobs):
    blobs = np.transpose(blobs)
    blobs = blobs.reshape(blobs.shape[1], blobs.shape[2])
    return blobs

solver = caffe.SGDSolver('solver.prototxt')

coder = Coder()

image = np.random.rand(224, 224, 3)
data = np.reshape(image, (1, 3, 224, 224))
gt = coder._generate_boxes(1)
inputs = prepare_inputs(data, gt)

caffe.set_mode_gpu()
net = solver.net
utils.set_inputs(net, **inputs)
for step in range(100):
    solver.step(1)
    delta = unpack_outputs(net.blobs['preds_reshape'].data)
    probs = unpack_outputs(net.blobs['final_probs'].data)

    bboxes = np.zeros((100, 5))
    bboxes[:, 0:4] = coder.decode(delta)
    bboxes[:, 4] = probs[:, 1]
    dets = utils.nms(bboxes)

    if step % 10 == 0: 
        ax = utils.draw_image(image)
        utils.vis_bboxes(ax, dets * 224, 'red')
        utils.vis_bboxes(ax, gt * 224, 'green')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('%04d.png'%step)
        plt.close()

