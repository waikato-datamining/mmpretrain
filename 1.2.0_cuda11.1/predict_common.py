from typing import Dict
import numpy as np
from mmpretrain import ImageClassificationInferencer


def init_model(config: str, checkpoint: str, device: str = "cuda") -> ImageClassificationInferencer:
    """
    Initializes the model.

    :param config: the configuration file
    :type config: str
    :param checkpoint: the checkpoint file to sue
    :type checkpoint: str
    :param device: the device to perform the inference on, eg "cuda" or "cpu"
    :type device: str
    :return: the inferencer instance
    """
    inferencer = ImageClassificationInferencer(model=config, pretrained=checkpoint, device=device)
    return inferencer


def inference_model(inferencer: ImageClassificationInferencer, img, top_k: int = None) -> Dict[str, float]:
    """
    Inference image(s) with the classifier.

    Based on mmcls.apis.inference module.

    :param inferencer: The loaded classifier.
    :type inferencer: ImageClassificationInferencer
    :param img: the image filename or loaded image
    :type img: str or np.ndarray
    :param top_k: whether to return just the top K predictions or all (when None)
    :type top_k: int
    :return: the dictionary with class labels and their associated scores
    :rtype: dict
    """
    preds = inferencer(img)[0]
    result = dict()
    scores = preds["pred_scores"]

    if top_k is not None:
        sorted_scores = np.flip(np.argsort(scores))
        for k in range(top_k):
            i = sorted_scores[k]
            result[inferencer.classes[i]] = float(scores[i])
    else:
        for i in range(len(scores)):
            result[inferencer.classes[i]] = float(scores[i])

    return result
