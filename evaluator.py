import cv2
import operator
import numpy as np
from test import character_predict


def image_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 199, 80)

    inv = cv2.bitwise_not(thresh)
    blur = cv2.GaussianBlur(thresh, (5, 5), cv2.BORDER_DEFAULT)
    kernel = np.ones((5, 5), np.uint8)
    dil = cv2.dilate(inv, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    er = cv2.erode(dil, kernel, iterations=2)
    return er


def seg_predict(prep, cnt):

    character_image = []
    image = cv2.bitwise_not(prep)
    for i, ctr in enumerate(cnt):

        x, y, w, h = cv2.boundingRect(ctr)
        roi = image[y:y+h, x:x+w]
        cv2.imshow('character'+str(i), roi)

        cv2.waitKey(0)
        roi_f = cv2.resize(
            roi, (128, 128), interpolation=cv2.INTER_AREA)
        cv2.imwrite('Details/character'+str(i) + ".jpg", roi_f)
        roi_f = cv2.cvtColor(roi_f, cv2.COLOR_BGR2RGB)

        print(character_predict(roi_f))
        character_image.append(roi_f)

    return list(map(character_predict, character_image))


def calculate(character):
    op1 = 0
    op2 = 0
    op = '+'
    flag = False

    for i in range(len(character)):
        if character[i].isdigit() and not flag:
            op1 = op1*10 + int(character[i])

        if character[i].isdigit() == False:
            op = character[i]
            flag = True

        if character[i].isdigit() and flag:
            op2 = op2*10 + int(character[i])

    ops = {"+": operator.add, "-": operator.sub,
           "*": operator.mul, "/": operator.truediv}
    print(str(op1) + " " + op + " " + str(op2) +
          " = " + str(ops[op](op1, op2)))


def main(image_path):

    orig = cv2.imread(image_path)
    (H, W, _) = orig.shape
    if H > W:
        orig = cv2.rotate(orig, cv2.ROTATE_90_CLOCKWISE)

    processed = image_preprocess(orig)
    cv2.imshow('image', processed)
    cv2.waitKey(0)

    kernel = np.ones((5, 5), np.uint8)
    intermediate = cv2.dilate(processed, kernel, iterations=2)

    ctrs, hier = cv2.findContours(intermediate, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    characters = seg_predict(processed, sorted_ctrs)
    calculate(characters)


if __name__ == "__main__":
    model_path = "model/model-bst.h5"
    image_path = "test-image.jpg"
    main(image_path)
