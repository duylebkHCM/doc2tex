import os
import lmdb
import cv2
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.data_const import LMDB_CONST, LABEL_KEY
import time


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    isValid = True
    imgH = None
    imgW = None
    try:
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
        if imgH * imgW == 0:
            isValid = False
    except Exception as e:
        isValid = False
    return isValid, (imgH, imgW)


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    with open(gtFile, "r", encoding="utf-8") as data:
        datalist = data.readlines()
        datalist = datalist[1:]  # exclude column's header

    nSamples = len(datalist)
    for i in range(nSamples):
        image_name, label = (
            datalist[i].strip("\n").split(LABEL_KEY.DELIMITER.value)
        )  # column is seperated by tab char
        imagePath = os.path.join(inputPath, image_name)
        if not os.path.exists(imagePath):
            print("%s does not exist" % imagePath)
            continue

        with open(imagePath, "rb") as f:
            imageBin = f.read()

        try:
            isValid, (imgH, imgW) = checkImageIsValid(imageBin)
            if not isValid:
                print("%s is not a valid image" % imagePath)
                continue
            else:
                imageKey = f"{LMDB_CONST.IMAGE.value}-{cnt:09d}".encode()
                labelKey = f"{LMDB_CONST.LABEL.value}-{cnt:09d}".encode()
                nameKey = f"{LMDB_CONST.PATH.value}-{cnt:09d}".encode()
                heightKey = f"{LMDB_CONST.HEIGHT.value}-{cnt:09d}".encode()
                widthKey = f"{LMDB_CONST.WIDTH.value}-{cnt:09d}".encode()

                cache[imageKey] = imageBin
                cache[labelKey] = label.encode()
                cache[nameKey] = image_name.encode()
                cache[heightKey] = np.array([imgH], dtype=np.int32).tobytes()
                cache[widthKey] = np.array([imgW], dtype=np.int32).tobytes()
        except Exception as e:
            print("error occured", e)
            with open(outputPath + "/error_image_log.txt", "a") as log:
                log.write("%s-th image data occured error\n" % str(i))
            continue

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print("Written %d / %d" % (cnt, nSamples))
        cnt += 1

    nSamples = cnt - 1
    cache[f"{LMDB_CONST.N_SAMPLES.value}".encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print("Created dataset with %d samples" % nSamples)


if __name__ == "__main__":
    start_time = time.time()
    inputPath, gtFile, outputPath = sys.argv[1:]
    createDataset(inputPath, gtFile, outputPath)
    print("Elpased time", time.time() - start_time)
