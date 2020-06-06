import cv2
import operator
import numpy as np
from test import character_predict


def image_preprocess(image):
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 199, 32)

    blur = cv2.GaussianBlur(thresh, (5, 5), cv2.BORDER_DEFAULT)
    kernel = np.ones((5, 5), np.uint8)
    blur = cv2.dilate(blur, kernel, iterations=1)

    thresh = cv2.bitwise_not(blur)
    return thresh


def seg_predict(image, cnt):
    # TODO: morphx_ex
    character_image = []

    for i, ctr in enumerate(cnt):

        x, y, w, h = cv2.boundingRect(ctr)
        roi = image[y-15:y+h+15, x-15:x+w+15]
        roi_f = cv2.resize(
            roi, (128, 128), interpolation=cv2.INTER_AREA)

        roi_f = cv2.cvtColor(roi_f, cv2.COLOR_BGR2RGB)
        character_image.append(roi_f)
        # cv2.imshow('character'+str(i), roi_f)

        cv2.waitKey(0)

    return list(map(character_predict, character_image))


def calculate(character):
    op1 = 0
    op2 = 0
    flag = False

    for i in range(len(character)):
        if character[i].isdigit() and not flag:
            op1 = op1*10 + int(character[i])

        if character[i].isdigit() == False:
            flag = True

        if character[i].isdigit() and flag:
            op2 = op2*10 + int(character[i])

    ops = {"+": operator.add, "-": operator.sub,
           "*": operator.mul, "/": operator.truediv}
    return ops[character[1]](op1, op2)


def main(image_path):
    image = cv2.imread(image_path, 0)
    # scale = 40
    # width = int(image.shape[1] * scale / 100)
    # height = int(image.shape[0] * scale / 100)
    # dim = (width, height)
    # image = cv2.resize(
    #     image, dim, interpolation=cv2.INTER_AREA)
    prepped = image_preprocess(image)
    ctrs, hier = cv2.findContours(prepped, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    characters = seg_predict(image, sorted_ctrs)
    output = calculate(characters)
    print(output)


if __name__ == "__main__":
    model_path = "model/model-bst.h5"
    image_path = "test-image.jpg"
    main(image_path)
