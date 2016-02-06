"""A encoder/decoder class to convert object detection labels to regression
targets for CNN.
"""

import numpy as np


class Coder(object):

    def __init__(self):
        self.anchors = self._generate_boxes(100)
        self.num_anchors = self.anchors.shape[0]

    def encode(self, xyxy, fg_thresh=0.7, bg_thresh=0.3):
        # fg->1; bg->0; others->0.5
        probs = np.ones((1, self.num_anchors)) * 0.5

        overlaps = Coder._overlaps(xyxy, self.anchors)
        probs[0, np.sum(overlaps < bg_thresh, 0) > 0] = 0
        probs[0, np.sum(overlaps > fg_thresh, 0) > 0] = 1
        probs[0, np.argmax(overlaps, 1)] = 1
        labels = np.concatenate((probs, 1 - probs))

        matches = np.argmax(overlaps, 0)
        flags = labels[0, :] > 0.5

        locs = np.zeros((self.anchors.shape))
        locs[flags, :] = Coder._to_delta(xyxy[matches[flags], :],
                                         self.anchors[flags, :])
        locs = np.transpose(locs)
        return labels, locs

    def decode(self, locs):
        return Coder._from_delta(locs, self.anchors)

    @staticmethod
    def _to_delta(xyxy_a, xyxy_b):
        delta = np.zeros(xyxy_a.shape)
        a = Coder._xyxy_to_xywh(xyxy_a)
        b = Coder._xyxy_to_xywh(xyxy_b)

        eps = np.finfo(np.float32).eps
        delta[:, 0] = np.divide(a[:, 0] - b[:, 0], b[:, 2] + eps)
        delta[:, 1] = np.divide(a[:, 1] - b[:, 1], b[:, 3] + eps)
        delta[:, 2] = np.log(np.divide(a[:, 2], b[:, 2]))
        delta[:, 3] = np.log(np.divide(a[:, 3], b[:, 3]))
        return delta

    @staticmethod
    def _from_delta(delta, xyxy_b):
        a = np.zeros(delta.shape)
        b = Coder._xyxy_to_xywh(xyxy_b)
        a[:, 0] = np.multiply(delta[:, 0], b[:, 2]) + b[:, 0]
        a[:, 1] = np.multiply(delta[:, 1], b[:, 3]) + b[:, 1]
        a[:, 2] = np.multiply(np.exp(delta[:, 2]), b[:, 2])
        a[:, 3] = np.multiply(np.exp(delta[:, 3]), b[:, 3])
        return Coder._xywh_to_xyxy(a)

    @staticmethod
    def _generate_boxes(num_boxes=1):
        xyxy = np.random.rand(num_boxes, 4)
        boxes = np.zeros(xyxy.shape)
        boxes[:, 0] = np.minimum(xyxy[:, 0], xyxy[:, 2])
        boxes[:, 1] = np.minimum(xyxy[:, 1], xyxy[:, 3])
        boxes[:, 2] = np.maximum(xyxy[:, 0], xyxy[:, 2])
        boxes[:, 3] = np.maximum(xyxy[:, 1], xyxy[:, 3])
        return boxes

    @staticmethod
    def _xyxy_to_xywh(xyxy):
        xywh = np.zeros(xyxy.shape)
        xywh[:, 0] = 0.5 * (xyxy[:, 0] + xyxy[:, 2])
        xywh[:, 1] = 0.5 * (xyxy[:, 1] + xyxy[:, 3])
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        return xywh

    @staticmethod
    def _xywh_to_xyxy(xywh):
        xyxy = np.zeros(xywh.shape)
        xyxy[:, 0] = xywh[:, 0] - 0.5 * xywh[:, 2]
        xyxy[:, 1] = xywh[:, 1] - 0.5 * xywh[:, 3]
        xyxy[:, 2] = xywh[:, 0] + 0.5 * xywh[:, 2]
        xyxy[:, 3] = xywh[:, 1] + 0.5 * xywh[:, 3]
        return xyxy

    @staticmethod
    def _overlaps(xyxy_a, xyxy_b):
        num_a = xyxy_a.shape[0]
        num_b = xyxy_b.shape[0]
        overlaps = np.zeros((num_a, num_b))

        xywh_a = Coder._xyxy_to_xywh(xyxy_a)
        xywh_b = Coder._xyxy_to_xywh(xyxy_b)

        eps = np.finfo(np.float32).eps
        area_a = np.multiply(xywh_a[:, 2], xywh_a[:, 3]) + eps
        area_b = np.multiply(xywh_b[:, 2], xywh_b[:, 3]) + eps

        for i in range(num_a):
            a_i = xyxy_a[i, :]
            xmin = np.maximum(a_i[0], xyxy_b[:, 0])
            ymin = np.maximum(a_i[1], xyxy_b[:, 1])
            xmax = np.minimum(a_i[2], xyxy_b[:, 2])
            ymax = np.minimum(a_i[3], xyxy_b[:, 3])
            inter_area = np.multiply(xmax - xmin, ymax - ymin)
            overlaps[i, :] = (inter_area) / (area_a[i] + area_b - inter_area)
        return overlaps
