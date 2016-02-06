import numpy as np
import caffe
import matplotlib.pyplot as plt


def set_inputs(net, **kwargs):
    for key, value in kwargs.iteritems():
        net.blobs[key].reshape(*value.shape)
        net.blobs[key].data[:] = value


def vis_bboxes(ax, bboxes, edgecolor='red'):
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, :]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False,
                          edgecolor=edgecolor,
                          linewidth=3.5)
        )


def draw_image(im):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    return ax

def nms(bboxes, threshold=0.5):
  indices = np.argsort(bboxes[:, 4])
  indices[:] = indices[::-1]
  return bboxes[indices[0:2], :]
