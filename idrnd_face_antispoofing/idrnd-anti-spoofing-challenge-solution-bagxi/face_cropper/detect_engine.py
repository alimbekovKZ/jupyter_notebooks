import os

import torch
import torch.nn.functional as F
import numpy as np

from .net_s3fd import s3fd
from .bbox import decode, nms
from .image import Image


class FaceDetectionEngine:
    def __init__(self, **kwargs):
        weights_path = kwargs.get(
            'weights_path',
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                's3fd_convert.pth'
            ),
        )
        self._net = s3fd()
        self._net.load_state_dict(torch.load(weights_path))

        if torch.cuda.is_available():
            self._net.cuda()
        self._net.eval()

    @staticmethod
    def dilate_bbox(bbox, dilate_pixels):
        #0, 1 upper left
        #2, 3 lower right

        bbox[0] = bbox[0] - dilate_pixels
        bbox[1] = bbox[1] - dilate_pixels
        bbox[2] = bbox[2] + dilate_pixels
        bbox[3] = bbox[3] + dilate_pixels
        bbox = np.clip(bbox, 0, 4000)
        return bbox

    def predict(self, img, dilate_bbox=True, dilate_pixels=10):
        img = Image(img)
        bboxlist = self._detect(img)
        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x[:-1] for x in bboxlist if x[-1] > 0.5]

        if dilate_bbox:
            return [self.dilate_bbox(box, dilate_pixels) for box in bboxlist]

        return bboxlist

    @staticmethod
    def crop(img, bbox):
        bbox = [int(i) for i in bbox]
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

    def _detect(self, img):
        net = self._net
        img = img - np.array([104, 117, 123])
        img = img.transpose(2, 0, 1)
        img = img.reshape((1,) + img.shape)
        img = torch.from_numpy(img).float()

        if torch.cuda.is_available():
            img = img.cuda()

        # BB, CC, HH, WW = img.size()
        with torch.no_grad():
            olist = net(img)

        bboxlist = []
        for i in range(len(olist) // 2):
            olist[i * 2] = F.softmax(olist[i * 2], dim=1)
        olist = [oelem.data.cpu() for oelem in olist]
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            # FB, FC, FH, FW = ocls.size()  # feature map size
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            # anchor = stride * 4
            poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
            for _, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[0, 1, hindex, windex]
                loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
                priors = torch.Tensor(
                    [[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                variances = [0.1, 0.2]
                box = decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0] * 1.0
                bboxlist.append([x1, y1, x2, y2, score])
        bboxlist = np.array(bboxlist)
        if len(bboxlist) == 0:
            bboxlist = np.zeros((1, 5))
        return bboxlist
