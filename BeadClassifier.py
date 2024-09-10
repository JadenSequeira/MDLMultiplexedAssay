import time

import cv2
import matplotlib
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QMovie, QPixmap, QCursor, QDesktopServices, QPainter, QImage
from PyQt5.QtWidgets import QLabel, QApplication, QMessageBox, QFrame, QFileDialog
#~155
image = cv2.imread("C:\\Users\\Jaden\\Desktop\\imager\\Cropped Image0.jpg", cv2.IMREAD_COLOR)
image = cv2.resize(image, (700, 700))
Copy = image.copy()
Copy1 = image.copy()
image = cv2.GaussianBlur(image, (5, 5), 2)
frame1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fourcc2 = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#fourcc3 = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#out3 = cv2.VideoWriter('C:\\Users\\Jaden\\Desktop\\videod.mp4', fourcc3, 2, (320, 240))
out = cv2.VideoWriter('C:\\Users\\Jaden\\Desktop\\videop.mp4', fourcc2, 2, (700, 700))
out2 = cv2.VideoWriter('C:\\Users\\Jaden\\Desktop\\videost.mp4', fourcc2, 2, (700, 700))

positions = []
positions2 = []
circles = []
swa = 700
sha = 700
sx = 350
sy = 350
s = 7
start = 5
h = 17
framecounter = 0
max = False
weight = 1
# for j in range(5, 10, 1):
#     for k in range(15, 25, 1):
while (start < s):
    print("j ", start, s)
    start = start + 1
    twostart = 10
    while twostart < h:
        ret, final = cv2.threshold(frame1, twostart, 255, cv2.THRESH_BINARY)
        ret, final1 = cv2.threshold(frame1, start, 255, cv2.THRESH_BINARY)
        final1 = cv2.subtract(final1, final)
        copy = Copy.copy()
        copy3 = Copy.copy()
        print("sum ", np.sum(final1))
        twostart = twostart + 1

        if (cv2.sumElems(final1)[0] / 255 > 2):
             contours, _ = cv2.findContours(final1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
             area = {}
             perimeter = {}
             convexity = []
             hull = []

             radi = []
             radii2 = []
             for i in range(len(contours)):
                 cnt = contours[i]
                 ar = cv2.contourArea(cnt)
                 area[i] = ar
                 per = cv2.arcLength(cnt, True)
                 perimeter[i] = per
                 convexity.append(cv2.isContourConvex(cnt))
                 (x, y), radius = cv2.minEnclosingCircle(cnt)

                 (xa, ya, wa, ha) = cv2.boundingRect(cnt)
                 if (wa > 500 and ha > 500 and xa+wa < 680 and ya+ha < 680):
                     if (swa*sha > wa*ha ):
                         swa = wa
                         sha = ha
                         sx = xa
                         sy  = ya
                         print("w and h: ", swa, sha)



                 point = []
                 point2 = []
                 center = (float(x),float(y))#int(x), int(y))
                 dcenter = (int(x), int(y))
                 frad = float(radius)
                 radius = int(radius)
                 radi.append(radius)
                 if (per != 0 and (radius < 28 and radius >25) or (radius <12 and radius > 7)):
                     skip = False
                     skip2 = False
                     posy = np.array(positions)
                     if (len(posy) > 1 and len(posy[:,0]) > 1 and len(posy[:,0]) > 1):
                         for p in range(len(posy)):
                             dist = np.sqrt(((posy[p][0]-x)*(posy[p][0]-x))+((posy[p][1]-y)*(posy[p][1]-y)))
                             if (dist < (radius+posy[p][2])):
                                 skip = True
                                 if (radius > positions[p][2]):
                                     positions[p][2] = radius
                     posy2 = np.array(positions2)
                     if (len(posy2) > 1 and len(posy2[:,0]) > 1 and len(posy2[:,0]) > 1):
                         for p in range(len(posy2)):
                             dist = np.sqrt(((posy2[p][0]-x)*(posy2[p][0]-x))+((posy2[p][1]-y)*(posy2[p][1]-y)))
                             if (dist < 0.5*(radius+posy2[p][2])):
                                 skip2 = True
                                 #if (radius > positions[p][2]):
                                 #    positions[p][2] = radius
                     if (not(skip)):
                         radii2.append(2 * ar / per)
                         cv2.circle(Copy, dcenter, radius, (0, 255, 0), 2)
                         point.append(x)
                         point.append(y)
                         point.append(radius)
                         positions.append(point)
                     if (not (skip2)):
                         radii2.append(2 * ar / per)
                         cv2.circle(copy3, dcenter, radius, (0, 255, 0), 2)
                         point2.append(x)
                         point2.append(y)
                         point2.append(radius)
                         positions2.append(point2)
                 else:
                     radii2.append(0)

             circles.append([start,twostart,len(radii2)])
             a = np.expand_dims(final1, axis = -1)
             c = np.concatenate([a, a, a], axis = 2)
             out.write(Copy)
             out2.write(copy3)

        #     framecounter = framecounter + 1
        #     if (framecounter == 3):
        #         framecounter = 0
        #
        #         if (start >= twostart-2):
        #             twostart
        if (swa > 650 or sha > 650 or sx < 20 or sy < 20):#580
             print("h: ", h)
             if (h < 30):
                h = h + 1
             if (h == 30 and (swa > 650 or sha > 650)):
                 max = True
             if (start-twostart > - 2):
                 start = start - np.abs(start - twostart)
             print(s)



             time.sleep(0.2)

if (h == 30 and (swa < 580 or sha < 580)):
    max = False

cv2.rectangle(Copy, (sx, sy), (sx + swa, sy + sha), (0, 255, 0), 2)
out.write(Copy)
time.sleep(0.2)

thick = 40
cornerthick = 35
uppery = sy + sha - thick
lowery = sy + thick
upperx = sx + swa - thick
lowerx = sx + thick

count = 0
positio = []
for i in positions:
    k = 0
    temp = []
    for j in positions2:
        dist = np.sqrt(((j[0] - i[0]) * (j[0] - i[0])) + ((j[1] - i[1]) * (j[1] - i[1])))
        if (dist < np.max([i[2], j[2]]) and (i[2] - j[2]) > 10):
            k = k + 1
            temp.append(j)

    if (k > 2 and k <4):
        for p in temp:
            positio.append(p)
    else:
        positio.append(i)

            # if (radius > positions[p][2]):
            #    positions[p][2] = radius





indicesRem = []
finalPositions = []
if (max == False):
    for i in range(len(positio)):
        tcoord = positio[i]
        if (tcoord[0] > upperx or tcoord[0] < lowerx or tcoord[1] > uppery or tcoord[1] < lowery or (tcoord[0] > upperx - cornerthick and tcoord[1] > uppery - cornerthick)
        or (tcoord[0] > upperx - cornerthick and tcoord[1] < lowery + cornerthick) or (tcoord[0] < lowery + cornerthick and tcoord[1] > uppery - cornerthick)
        or (tcoord[0] < lowerx + cornerthick and tcoord[1] < lowery + cornerthick)):
            indicesRem.append(i)
        else:
            finalPositions.append(positio[i])
else:
    finalPositions = positio

for i in finalPositions:
    print("pos ", i)
    cv2.circle(Copy1, (int(i[0]), int(i[1])), i[2], (0, 255, 0), 2)

if (max == False):
    cv2.rectangle(Copy1, (sx, sy), (sx + swa, sy + sha), (0, 255, 0), 2)

print("max ", max)
out.write(Copy1)
out.write(Copy1)
out.write(Copy1)
out.write(Copy1)

time.sleep(0.2)

print("positio ", positio)
print(positions)
print(positions2)
print(len(finalPositions))
print(sx, sy, sha, swa)
out.release()
out2.release()
cv2.imshow("image", Copy1)
#cv2.imshow("image", image)


cv2.waitKey(0)

cv2.destroyAllWindows()
