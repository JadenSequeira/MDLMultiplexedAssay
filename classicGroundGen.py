import cv2
import numpy as np


for p in range(14545):
    print(p)
    image = cv2.imread("C:\\Users\\Jaden\\Desktop\\croppedBrIm\\" + str(p) + ".jpg")
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ### image preprocessing-------------------------------------------------------------------
    #ret, threshold = cv2.threshold(FF_img.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # for i in [5,9,11,13,21,25]:
    #    for j in range(3,9):
    #         #threshold = cv2.adaptiveThreshold(FF_img.copy(),255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,i,j)
    #         threshold = cv2.adaptiveThreshold(FF_img.copy(),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,i,j)
    #         kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #         opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel2)
    #         cv2.imwrite("C:\\Users\\Jaden\\Desktop\\b.png", threshold)#settings\\gaub" + str(i) + "c" + str(j) + "a.png", threshold)

    #threshold = cv2.adaptiveThreshold(img.copy(), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    ret, threshold = cv2.threshold(img.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, threshold2 = cv2.threshold(img.copy(), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    swa = 0
    sha = 0
    sx = 0
    sy = 0
    ind = 0
    for i in range(len(contours)):
        cnt = contours[i]
        ar = cv2.contourArea(cnt)
        area.append(ar)

        (xa, ya, wa, ha) = cv2.boundingRect(cnt)
        # if (wa > 500 and ha > 500 and xa+wa < 680 and ya+ha < 680):
        if (5000 < wa*ha ):
            swa = wa -20
            sha = ha-10
            sx = xa+10
            sy  = ya
            ind = i

    image1 = cv2.drawContours(img.copy(), contours, -1, (255,255,255), 3)
    image2 = cv2.fillConvexPoly(img.copy(), contours[ind], (255, 255, 255), lineType=100, shift=0)
    #image3 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    ret, threshy = cv2.threshold(image2, 100, 255, cv2.THRESH_BINARY)

    cv2.rectangle(threshold,(sx,sy),(sx + swa, sy + sha),(255,255,255),1)
    # final = np.zeros((threshold.shape[0], threshold.shape[1]), "uint8")
    # for k in range(sx, sx+swa):
    #     for j in range(sy, sy + sha):
    #         final[k][j] = threshold2[k][j]

    finaling = np.multiply(threshy/255, threshold2)
    kernel22 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    final = cv2.morphologyEx(finaling, cv2.MORPH_OPEN, kernel22)
    #kernel23 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #final = cv2.dilate(final,kernel23,iterations = 1)
    FF = np.expand_dims(final.copy(), axis=-1)
    BF = np.expand_dims(img.copy(), axis=-1)
    bst = np.expand_dims(np.zeros((FF.shape[0], FF.shape[0])), axis=-1)
    HH_img = np.concatenate([FF, FF, bst], axis=2)
    HH_img = HH_img.astype("uint8")
    JJ_img = np.concatenate([BF, BF, BF], axis=2)
    HH = cv2.addWeighted(HH_img, 0.15, JJ_img, 0.85, 0)
    cv2.imwrite("C:\\Users\\Jaden\\Desktop\\croppedCVQual\\" + str(p) + ".jpg", HH)  # settings\\gaub" + str(i) + "c" + str(j) + "a.png", threshold)
    cv2.imwrite("C:\\Users\\Jaden\\Desktop\\croppedCVFLIM\\" + str(p) + ".jpg", final)
# cv2.imshow("img", final)
# cv2.waitKey(0)
# cv2.destroyAllWindows()