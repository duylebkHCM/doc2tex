import Levenshtein
from nltk.metrics.distance import edit_distance


def get_single_ED(gt, pred):
    # ICDAR2019 Normalized Edit Distance
    if len(gt) == 0 or len(pred) == 0:
        norm_ED = 0
    elif len(gt) > len(pred):
        norm_ED = 1 - edit_distance(pred, gt) / len(gt)
    else:
        norm_ED = 1 - edit_distance(pred, gt) / len(pred)
    return norm_ED


def get_word_NED(list_preds, list_gts):
    word_NED = 0
    word_len = 0

    if isinstance(list_preds, str):
        list_preds = [list_preds]
    if isinstance(list_gts, str):
        list_gts = [list_gts]

    for gt, pred in zip(list_gts, list_preds):
        word_gt = gt.split()
        word_pred = pred.split()

        cur_max_len = max(len(word_gt), len(word_pred))

        if len(gt) == 0 or len(pred) == 0:
            word_NED += 0
        else:
            word_NED += 1 - Levenshtein.distance(word_gt, word_pred) / cur_max_len

        word_len += cur_max_len

    word_NED = word_NED / float(len(list_gts))

    return word_NED
