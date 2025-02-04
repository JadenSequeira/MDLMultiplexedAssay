import os
from os import listdir
from os.path import isfile, join
import cv2

base_dir = "C:\\Users\\Jaden\\Desktop\\scurve\\"

onlyfiles = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]
print(onlyfiles)

# B C 1000, D E 500, H I 250, J K 125, L M 62.5, N 0
num = 0
numbers1 = []
for i in onlyfiles:
    word = i
    splitup = [*word]
    if (splitup[4] == 'B' or splitup[4] == 'C'):
        name = str(num) + "_1000.tif"

    elif (splitup[4] == 'D' or splitup[4] == 'E'):
        name = str(num) + "_500.tif"
    elif (splitup[4] == 'H' or splitup[4] == 'I'):
        name = str(num) + "_250.tif"
    elif (splitup[4] == 'J' or splitup[4] == 'K'):
        name = str(num) + "_125.tif"
    elif (splitup[4] == 'L' or splitup[4] == 'M'):
        name = str(num) + "_62.5.tif"
    else:
        name = str(num) + "_0.tif"

    os.rename(base_dir+"\\"+i, base_dir + "\\" + name)

    num = num + 1
