import cv2
import numpy as np

image = cv2.imread('./test-image.jpg', 0)

scale = 40
width = int(image.shape[1] * scale / 100)
height = int(image.shape[0] * scale / 100)
dim = (width, height)
image = cv2.resize(
    image, dim, interpolation=cv2.INTER_AREA)
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 199, 32)

cv2.imshow('image', thresh)
cv2.waitKey(0)
blur = cv2.GaussianBlur(thresh, (5, 5), cv2.BORDER_DEFAULT)
kernel = np.ones((5, 5), np.uint8)
blur = cv2.dilate(blur, kernel, iterations=1)

thresh = cv2.bitwise_not(blur)
ctrs, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])


# TODO: morphx_ex

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)
    # Getting ROI
    roi = image[y-15:y+h+15, x-15:x+w+15]

    # show ROI
    #cv2.imwrite('roi_imgs.png', roi)
    cv2.imshow('charachter'+str(i), roi)
    # cv2.rectangle(image, (x-50, y-50),
    #               (x + w + 50, y + h + 50), (90, 0, 255), 2)
    # cv2.imshow("rec", image)
    cv2.waitKey(0)
