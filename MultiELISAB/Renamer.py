import os
from os import listdir
from os.path import isfile, join
import cv2

base_dir = "C:\\Users\\Jaden\\Desktop\\newimgs\\"
onlyfiles = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]
print(onlyfiles)

# B C 1000, D E 500, H I 250, J K 125, L M 62.5, N 0
ins = [1000, 1000, 500, 500, 250, 250, 125, 125, 62.5, 62.5, 0]
glu = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
num = 0
numbers1 = []
for i in onlyfiles:
    word = i
    splitup = [*word]
    # if (splitup[4] == 'B'):
    #     name = str(num) + "_" + str(glu[0]) + "_" + str(ins[0]) + ".tif"
    # elif (splitup[4] == 'C'):
    #     name = str(num) + "_" + str(glu[1]) + "_" + str(ins[1]) + ".tif"
    # elif (splitup[4] == 'D'):
    #     name = str(num) + "_" + str(glu[2]) + "_" + str(ins[2]) + ".tif"
    # elif (splitup[4] == 'E'):
    #     name = str(num) + "_" + str(glu[3]) + "_" + str(ins[3]) + ".tif"
    # elif (splitup[4] == 'H'):
    #     name = str(num) + "_" + str(glu[4]) + "_" + str(ins[4]) + ".tif"
    # elif (splitup[4] == 'I'):
    #     name = str(num) + "_" + str(glu[5]) + "_" + str(ins[5]) + ".tif"
    # elif (splitup[4] == 'J'):
    #     name = str(num) + "_" + str(glu[6]) + "_" + str(ins[6]) + ".tif"
    # elif (splitup[4] == 'K'):
    #     name = str(num) + "_" + str(glu[7]) + "_" + str(ins[7]) + ".tif"
    # elif (splitup[4] == 'L'):
    #     name = str(num) + "_" + str(glu[8]) + "_" + str(ins[8]) + ".tif"
    # elif (splitup[4] == 'M'):
    #     name = str(num) + "_" + str(glu[9]) + "_" + str(ins[9]) + ".tif"
    # else:
    name = str(num) + "_" + str(num) + "_" + str(num) + ".tif"


    os.rename(base_dir+"\\"+i, base_dir + "\\" + name)

    num = num + 1