import cv2
from albumentations.augmentations import functional as F

from .detect_engine import FaceDetectionEngine


def crop_faces(img, face_detector):
    """Returns: Tuple(Tuple(face_crop, bbox), img_without_faces)"""
    preds = face_detector.predict(img, dilate_bbox=True)
    cutout_faces = F.cutout(img, holes=[bbox.astype(int) for bbox in preds])
    face_crops = [(face_detector.crop(img, bbox), bbox) for bbox in preds]

    return face_crops, cutout_faces
