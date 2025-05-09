## Main GUI for Multiplexed Bead ELISA
#  Author: Jaden Sequeira
#  Email: jaden.sequeira609@gmail.com

import csv
import os
import shutil
import time
import pandas as pd
import Model
import keyboard
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, Qt, QEvent, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QKeyEvent
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QMainWindow, QWidget
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

#Initialize all default settings
minA = 3500
maxA = 30000
Xmin = 100
Xmax = 2000
Ymin = 100
Ymax = 2000
k1 = 3
oiter = 10
k2 = 40
Br = -1
T28 = -1
T45 = -1
imgNum = 0
executed = False
autoMax = 9000
startPos = 1
offseta = 0
numberF = 0
trainS = 0
moveOn = False

#Creates GUI window for the U-Net training and fluorescence measurement
class Ui_MainWindow(QMainWindow):
    #Initialize all file storage location variables
    fname = ""
    sname = ""
    fname1 = ""
    sname1 = ""
    fname2 = ""
    sname2 = ""
    modelName = ""
    fname3 = ""
    sname3 = ""
    fname4 = ""
    sname4 = ""
    fname5 = ""
    sname5 = ""

    #Setup the GUI with GUI components
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1289, 858)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(-6, 0, 1521, 821))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.lineEdit = QtWidgets.QLineEdit(self.tab)
        self.lineEdit.setGeometry(QtCore.QRect(30, 100, 231, 22))
        self.lineEdit.setObjectName("lineEdit")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(30, 70, 201, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setGeometry(QtCore.QRect(410, 10, 850, 750))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setGeometry(QtCore.QRect(30, 320, 81, 16))
        self.label_3.setObjectName("label_3")
        self.line = QtWidgets.QFrame(self.tab)
        self.line.setGeometry(QtCore.QRect(30, 330, 351, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_4 = QtWidgets.QLabel(self.tab)
        self.label_4.setGeometry(QtCore.QRect(70, 360, 81, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.tab)
        self.label_5.setGeometry(QtCore.QRect(230, 360, 81, 16))
        self.label_5.setObjectName("label_5")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_2.setGeometry(QtCore.QRect(30, 380, 131, 22))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_3.setGeometry(QtCore.QRect(200, 380, 131, 22))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_4.setGeometry(QtCore.QRect(30, 440, 131, 22))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.label_6 = QtWidgets.QLabel(self.tab)
        self.label_6.setGeometry(QtCore.QRect(70, 420, 81, 16))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.tab)
        self.label_7.setGeometry(QtCore.QRect(240, 420, 81, 16))
        self.label_7.setObjectName("label_7")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_5.setGeometry(QtCore.QRect(200, 440, 131, 22))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.label_8 = QtWidgets.QLabel(self.tab)
        self.label_8.setGeometry(QtCore.QRect(240, 480, 81, 16))
        self.label_8.setObjectName("label_8")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_6.setGeometry(QtCore.QRect(200, 500, 131, 22))
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.label_9 = QtWidgets.QLabel(self.tab)
        self.label_9.setGeometry(QtCore.QRect(70, 480, 81, 16))
        self.label_9.setObjectName("label_9")
        self.lineEdit_7 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_7.setGeometry(QtCore.QRect(30, 500, 131, 22))
        self.lineEdit_7.setObjectName("lineEdit_7")



        self.label_100 = QtWidgets.QLabel(self.tab)
        self.label_100.setGeometry(QtCore.QRect(250, 540, 81, 16))
        self.label_100.setObjectName("label_100")
        self.lineEdit_100 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_100.setGeometry(QtCore.QRect(240, 560, 90, 22))
        self.lineEdit_100.setObjectName("lineEdit_100")
        self.label_101 = QtWidgets.QLabel(self.tab)
        self.label_101.setGeometry(QtCore.QRect(40, 540, 81, 16))
        self.label_101.setObjectName("label_101")
        self.lineEdit_101 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_101.setGeometry(QtCore.QRect(30, 560, 90, 22))
        self.lineEdit_101.setObjectName("lineEdit_101")
        self.label_102 = QtWidgets.QLabel(self.tab)
        self.label_102.setGeometry(QtCore.QRect(145, 540, 81, 16))
        self.label_102.setObjectName("label_102")
        self.lineEdit_102 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_102.setGeometry(QtCore.QRect(135, 560, 90, 22))
        self.lineEdit_102.setObjectName("lineEdit_102")
        self.label_103 = QtWidgets.QLabel(self.tab)
        self.label_103.setGeometry(QtCore.QRect(30, 600, 100, 16))
        self.label_103.setObjectName("label_103")
        self.lineEdit_103 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_103.setGeometry(QtCore.QRect(30, 620, 90, 22))
        self.lineEdit_103.setObjectName("lineEdit_103")
        self.label_104 = QtWidgets.QLabel(self.tab)
        self.label_104.setGeometry(QtCore.QRect(40, 750, 90, 16))
        self.label_104.setObjectName("label_104")

        self.label_300 = QtWidgets.QLabel(self.tab)
        self.label_300.setGeometry(QtCore.QRect(150, 600, 45, 16))
        self.label_300.setObjectName("label_300")
        self.lineEdit_300 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_300.setGeometry(QtCore.QRect(150, 620, 45, 22))
        self.lineEdit_300.setObjectName("lineEdit_300")

        self.label_301 = QtWidgets.QLabel(self.tab)
        self.label_301.setGeometry(QtCore.QRect(210, 600, 45, 16))
        self.label_301.setObjectName("label_301")
        self.lineEdit_301 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_301.setGeometry(QtCore.QRect(210, 620, 45, 22))
        self.lineEdit_301.setObjectName("lineEdit_301")

        self.label_302 = QtWidgets.QLabel(self.tab)
        self.label_302.setGeometry(QtCore.QRect(270, 600, 45, 16))
        self.label_302.setObjectName("label_302")
        self.lineEdit_302 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_302.setGeometry(QtCore.QRect(270, 620, 45, 22))
        self.lineEdit_302.setObjectName("lineEdit_302")



        self.pushButton = QtWidgets.QPushButton(self.tab)
        self.pushButton.setGeometry(QtCore.QRect(290, 100, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.tab)
        self.pushButton_2.setGeometry(QtCore.QRect(290, 170, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_10 = QtWidgets.QLabel(self.tab)
        self.label_10.setGeometry(QtCore.QRect(30, 140, 201, 16))
        self.label_10.setObjectName("label_10")
        self.lineEdit_8 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_8.setGeometry(QtCore.QRect(30, 170, 231, 22))
        self.lineEdit_8.setObjectName("lineEdit_8")



        self.pushButton_3 = QtWidgets.QPushButton(self.tab)
        self.pushButton_3.setGeometry(QtCore.QRect(30, 660, 301, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.tab)
        self.pushButton_4.setGeometry(QtCore.QRect(30, 710, 301, 28))
        self.pushButton_4.setObjectName("pushButton_4")
        self.label_69 = QtWidgets.QLabel(self.tab)
        self.label_69.setGeometry(QtCore.QRect(30, 270, 61, 16))
        self.label_69.setObjectName("label_69")
        self.radioButton_9 = QtWidgets.QRadioButton(self.tab)
        self.radioButton_9.setGeometry(QtCore.QRect(90, 270, 200, 20))
        self.radioButton_9.setObjectName("radioButton_9")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label_11 = QtWidgets.QLabel(self.tab_2)
        self.label_11.setGeometry(QtCore.QRect(142, 340, 120, 16))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.tab_2)
        self.label_12.setGeometry(QtCore.QRect(30, 370, 100, 16))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.tab_2)
        self.label_13.setGeometry(QtCore.QRect(30, 340, 120, 16))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.tab_2)
        self.label_14.setGeometry(QtCore.QRect(130, 370, 120, 16))
        self.label_14.setObjectName("label_14")
        self.lineEdit_9 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_9.setGeometry(QtCore.QRect(30, 130, 141, 22))
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.pushButton_5 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_5.setGeometry(QtCore.QRect(180, 60, 93, 28))
        self.pushButton_5.setObjectName("pushButton_5")
        self.lineEdit_10 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_10.setGeometry(QtCore.QRect(30, 60, 141, 22))
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.label_15 = QtWidgets.QLabel(self.tab_2)
        self.label_15.setGeometry(QtCore.QRect(30, 40, 201, 16))
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.tab_2)
        self.label_16.setGeometry(QtCore.QRect(30, 110, 201, 16))
        self.label_16.setObjectName("label_16")
        self.pushButton_6 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_6.setGeometry(QtCore.QRect(180, 130, 93, 28))
        self.pushButton_6.setObjectName("pushButton_6")
        self.label_17 = QtWidgets.QLabel(self.tab_2)
        self.label_17.setGeometry(QtCore.QRect(30, 205, 201, 16))
        self.label_17.setObjectName("label_17")
        self.lineEdit_11 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_11.setGeometry(QtCore.QRect(30, 225, 91, 22))
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.pushButton_7 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_7.setGeometry(QtCore.QRect(130, 225, 93, 28))
        self.pushButton_7.setObjectName("pushButton_7")
        self.label_18 = QtWidgets.QLabel(self.tab_2)
        self.label_18.setGeometry(QtCore.QRect(300, 10, 970, 760))
        self.label_18.setText("")
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.tab_2)
        self.label_19.setGeometry(QtCore.QRect(30, 245, 231, 71))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.tab_2)
        self.label_20.setGeometry(QtCore.QRect(30, 270, 231, 71))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.label_21 = QtWidgets.QLabel(self.tab_2)
        self.label_21.setGeometry(QtCore.QRect(30, 410, 81, 16))
        self.label_21.setObjectName("label_21")
        self.label_22 = QtWidgets.QLabel(self.tab_2)
        self.label_22.setGeometry(QtCore.QRect(30, 440, 201, 16))
        self.label_22.setObjectName("label_22")
        self.lineEdit_12 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_12.setGeometry(QtCore.QRect(30, 460, 141, 22))
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.pushButton_8 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_8.setGeometry(QtCore.QRect(180, 460, 93, 28))
        self.pushButton_8.setObjectName("pushButton_8")
        self.line_2 = QtWidgets.QFrame(self.tab_2)
        self.line_2.setGeometry(QtCore.QRect(30, 420, 241, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_23 = QtWidgets.QLabel(self.tab_2)
        self.label_23.setGeometry(QtCore.QRect(30, 660, 201, 16))
        self.label_23.setObjectName("label_23")
        self.lineEdit_13 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_13.setGeometry(QtCore.QRect(30, 620, 131, 22))
        self.lineEdit_13.setText("")
        self.lineEdit_13.setPlaceholderText("")
        self.lineEdit_13.setObjectName("lineEdit_13")
        self.label_25 = QtWidgets.QLabel(self.tab_2)
        self.label_25.setGeometry(QtCore.QRect(30, 600, 81, 16))
        self.label_25.setObjectName("label_25")
        self.label_26 = QtWidgets.QLabel(self.tab_2)
        self.label_26.setGeometry(QtCore.QRect(30, 580, 161, 16))
        self.label_26.setObjectName("label_26")
        self.pushButton_9 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_9.setGeometry(QtCore.QRect(180, 620, 93, 28))
        self.pushButton_9.setObjectName("pushButton_9")
        self.label_27 = QtWidgets.QLabel(self.tab_2)
        self.label_27.setGeometry(QtCore.QRect(190, 580, 161, 16))
        self.label_27.setObjectName("label_27")
        self.check_100 = QtWidgets.QCheckBox(self.tab_2)
        self.check_100.setGeometry(QtCore.QRect(120, 680, 95, 20))
        self.check_100.setObjectName("check_100")
        self.check_101 = QtWidgets.QCheckBox(self.tab_2)
        self.check_101.setGeometry(QtCore.QRect(170, 680, 95, 20))
        self.check_101.setObjectName("check_101")
        self.check_102 = QtWidgets.QCheckBox(self.tab_2)
        self.check_102.setGeometry(QtCore.QRect(230, 680, 95, 20))
        self.check_102.setObjectName("check_102")
        self.label_24 = QtWidgets.QLabel(self.tab_2)
        self.label_24.setGeometry(QtCore.QRect(60, 680, 61, 16))
        self.label_24.setObjectName("label_24")
        self.label_55 = QtWidgets.QLabel(self.tab_2)
        self.label_55.setGeometry(QtCore.QRect(60, 710, 61, 16))
        self.label_55.setObjectName("label_55")
        self.check_103 = QtWidgets.QCheckBox(self.tab_2)
        self.check_103.setGeometry(QtCore.QRect(120, 710, 95, 20))
        self.check_103.setObjectName("check_103")
        self.check_104 = QtWidgets.QCheckBox(self.tab_2)
        self.check_104.setGeometry(QtCore.QRect(220, 710, 95, 20))
        self.check_104.setObjectName("check_104")
        self.pushButton_19 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_19.setGeometry(QtCore.QRect(30, 750, 251, 28))
        self.pushButton_19.setObjectName("pushButton_19")
        self.pushButton_20 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_20.setGeometry(QtCore.QRect(180, 510, 93, 28))
        self.pushButton_20.setObjectName("pushButton_20")
        self.lineEdit_27 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_27.setGeometry(QtCore.QRect(30, 510, 141, 22))
        self.lineEdit_27.setObjectName("lineEdit_27")
        self.label_56 = QtWidgets.QLabel(self.tab_2)
        self.label_56.setGeometry(QtCore.QRect(30, 490, 201, 16))
        self.label_56.setObjectName("label_56")
        self.label_61 = QtWidgets.QLabel(self.tab_2)
        self.label_61.setGeometry(QtCore.QRect(30, 10, 81, 16))
        self.label_61.setObjectName("label_61")
        self.line_6 = QtWidgets.QFrame(self.tab_2)
        self.line_6.setGeometry(QtCore.QRect(30, 20, 241, 16))
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.pushButton_21 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_21.setGeometry(QtCore.QRect(180, 80, 93, 28))
        self.pushButton_21.setObjectName("pushButton_21")
        self.lineEdit_28 = QtWidgets.QLineEdit(self.tab_5)
        self.lineEdit_28.setGeometry(QtCore.QRect(30, 80, 141, 22))
        self.lineEdit_28.setObjectName("lineEdit_28")
        self.pushButton_22 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_22.setGeometry(QtCore.QRect(30, 170, 251, 28))
        self.pushButton_22.setObjectName("pushButton_22")
        self.label_57 = QtWidgets.QLabel(self.tab_5)
        self.label_57.setGeometry(QtCore.QRect(30, 60, 201, 16))
        self.label_57.setObjectName("label_57")
        self.label_58 = QtWidgets.QLabel(self.tab_5)
        self.label_58.setGeometry(QtCore.QRect(30, 110, 201, 16))
        self.label_58.setObjectName("label_58")
        self.lineEdit_29 = QtWidgets.QLineEdit(self.tab_5)
        self.lineEdit_29.setGeometry(QtCore.QRect(30, 130, 141, 22))
        self.lineEdit_29.setObjectName("lineEdit_29")
        self.pushButton_23 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_23.setGeometry(QtCore.QRect(180, 130, 93, 28))
        self.pushButton_23.setObjectName("pushButton_23")
        self.label_59 = QtWidgets.QLabel(self.tab_5)
        self.label_59.setGeometry(QtCore.QRect(30, 230, 250, 16))
        self.label_59.setObjectName("label_59")
        self.label_60 = QtWidgets.QLabel(self.tab_5)
        self.label_60.setGeometry(QtCore.QRect(30, 260, 250, 16))
        self.label_60.setObjectName("label_60")
        self.label_62 = QtWidgets.QLabel(self.tab_5)
        self.label_62.setGeometry(QtCore.QRect(30, 20, 81, 16))
        self.label_62.setObjectName("label_62")
        self.line_5 = QtWidgets.QFrame(self.tab_5)
        self.line_5.setGeometry(QtCore.QRect(30, 30, 241, 16))
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.line_7 = QtWidgets.QFrame(self.tab_5)
        self.line_7.setGeometry(QtCore.QRect(30, 310, 241, 16))
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.label_63 = QtWidgets.QLabel(self.tab_5)
        self.label_63.setGeometry(QtCore.QRect(30, 300, 270, 16))
        self.label_63.setObjectName("label_63")
        self.lineEdit_30 = QtWidgets.QLineEdit(self.tab_5)
        self.lineEdit_30.setGeometry(QtCore.QRect(30, 400, 141, 22))
        self.lineEdit_30.setObjectName("lineEdit_30")
        self.pushButton_24 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_24.setGeometry(QtCore.QRect(180, 400, 93, 28))
        self.pushButton_24.setObjectName("pushButton_24")
        self.label_64 = QtWidgets.QLabel(self.tab_5)
        self.label_64.setGeometry(QtCore.QRect(30, 380, 201, 16))
        self.label_64.setObjectName("label_64")
        self.label_65 = QtWidgets.QLabel(self.tab_5)
        self.label_65.setGeometry(QtCore.QRect(30, 330, 201, 16))
        self.label_65.setObjectName("label_65")
        self.pushButton_25 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_25.setGeometry(QtCore.QRect(180, 350, 93, 28))
        self.pushButton_25.setObjectName("pushButton_25")
        self.lineEdit_31 = QtWidgets.QLineEdit(self.tab_5)
        self.lineEdit_31.setGeometry(QtCore.QRect(30, 350, 141, 22))
        self.lineEdit_31.setObjectName("lineEdit_31")

        self.lineEdit_200 = QtWidgets.QLineEdit(self.tab_5)
        self.lineEdit_200.setGeometry(QtCore.QRect(30, 435, 25, 22))
        self.lineEdit_200.setObjectName("lineEdit_200")
        self.label_200 = QtWidgets.QLabel(self.tab_5)
        self.label_200.setGeometry(QtCore.QRect(60, 435, 50, 16))
        self.label_200.setObjectName("label_200")
        self.pushButton_200 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_200.setGeometry(QtCore.QRect(30, 520, 251, 28))
        self.pushButton_200.setObjectName("pushButton_200")
        self.radioButton_200 = QtWidgets.QRadioButton(self.tab_5)
        self.radioButton_200.setGeometry(QtCore.QRect(120, 435, 95, 20))
        self.radioButton_200.setObjectName("radioButton_200")
        self.radioButton_201 = QtWidgets.QRadioButton(self.tab_5)
        self.radioButton_201.setGeometry(QtCore.QRect(170, 435, 95, 20))
        self.radioButton_201.setObjectName("radioButton_200")
        self.radioButton_202 = QtWidgets.QRadioButton(self.tab_5)
        self.radioButton_202.setGeometry(QtCore.QRect(225, 435, 95, 20))
        self.radioButton_202.setObjectName("radioButton_202")


        self.pushButton_26 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_26.setGeometry(QtCore.QRect(30, 480, 251, 28))
        self.pushButton_26.setObjectName("pushButton_26")
        self.pushButton_27 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_27.setGeometry(QtCore.QRect(180, 640, 93, 28))
        self.pushButton_27.setObjectName("pushButton_27")
        self.pushButton_27.clicked.connect(self.openFile11)
        self.label_66 = QtWidgets.QLabel(self.tab_5)
        self.label_66.setGeometry(QtCore.QRect(30, 590, 141, 16))
        self.label_66.setObjectName("label_66")
        self.lineEdit_32 = QtWidgets.QLineEdit(self.tab_5)
        self.lineEdit_32.setGeometry(QtCore.QRect(30, 640, 141, 22))
        self.lineEdit_32.setObjectName("lineEdit_32")
        self.label_67 = QtWidgets.QLabel(self.tab_5)
        self.label_67.setGeometry(QtCore.QRect(30, 620, 201, 16))
        self.label_67.setObjectName("label_67")
        self.line_8 = QtWidgets.QFrame(self.tab_5)
        self.line_8.setGeometry(QtCore.QRect(30, 600, 241, 16))
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.pushButton_28 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_28.setGeometry(QtCore.QRect(30, 680, 251, 28))
        self.pushButton_28.setObjectName("pushButton_28")
        self.pushButton_28.clicked.connect(self.combiner)
        self.label_400 = QtWidgets.QLabel(self.tab_5)
        self.label_400.setGeometry(QtCore.QRect(30, 730, 201, 16))
        self.label_400.setObjectName("label_400")
        self.label_401 = QtWidgets.QLabel(self.tab_5)
        self.label_401.setGeometry(QtCore.QRect(30, 760, 500, 35))
        self.label_401.setObjectName("label_401")
        self.label_68 = QtWidgets.QLabel(self.tab_5)
        self.label_68.setGeometry(QtCore.QRect(310, 10, 970, 760))
        self.label_68.setText("")
        self.label_68.setObjectName("label_68")
        self.tabWidget.addTab(self.tab_5, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1289, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.label_105 = QtWidgets.QLabel(self.tab_2)
        self.label_105.setGeometry(QtCore.QRect(30, 160, 201, 16))
        self.label_105.setObjectName("label_105")
        self.lineEdit_105 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_105.setGeometry(QtCore.QRect(30, 175, 91, 22))
        self.lineEdit_105.setObjectName("lineEdit_105")
        self.check_105 = QtWidgets.QCheckBox(self.tab_2)
        self.check_105.setGeometry(QtCore.QRect(145, 175, 95, 20))
        self.check_105.setObjectName("check_105")
        self.pushButton_106 = QtWidgets.QPushButton(self.tab)
        self.pushButton_106.setGeometry(QtCore.QRect(290, 220, 93, 28))
        self.pushButton_106.setObjectName("pushButton_106")
        self.label_106 = QtWidgets.QLabel(self.tab)
        self.label_106.setGeometry(QtCore.QRect(30, 200, 201, 16))
        self.label_106.setObjectName("label_106")
        self.lineEdit_106 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_106.setGeometry(QtCore.QRect(30, 220, 231, 22))
        self.lineEdit_106.setObjectName("lineEdit_106")
        self.label_107 = QtWidgets.QLabel(self.tab_2)
        self.label_107.setGeometry(QtCore.QRect(30, 545, 161, 16))
        self.label_107.setObjectName("label_107")
        self.lineEdit_107 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_107.setGeometry(QtCore.QRect(90, 545, 180, 22))
        self.lineEdit_107.setObjectName("lineEdit_106")
        self.label_108 = QtWidgets.QLabel(self.tab_5)
        self.label_108.setGeometry(QtCore.QRect(30, 560, 251, 16))
        self.label_108.setObjectName("label_108")

        self.pushButton.clicked.connect(self.openFile1)
        self.pushButton_2.clicked.connect(self.openFile2)
        self.pushButton_3.clicked.connect(self.generateSeg)
        self.pushButton_4.clicked.connect(self.cropNano)
        self.pushButton_5.clicked.connect(self.openFile3)
        self.pushButton_6.clicked.connect(self.openFile4)
        self.pushButton_7.clicked.connect(self.qualInspect)
        self.pushButton_106.clicked.connect(self.upoffset)
        self.pushButton_9.clicked.connect(self.trainSize)
        self.pushButton_8.clicked.connect(self.openFile5)
        self.pushButton_20.clicked.connect(self.openFile6)
        self.pushButton_21.clicked.connect(self.openFile7)
        self.pushButton_22.clicked.connect(self.testManager)
        self.pushButton_23.clicked.connect(self.openFile8)
        self.pushButton_19.clicked.connect(self.trainer)
        self.pushButton_25.clicked.connect(self.openFile9)
        self.pushButton_24.clicked.connect(self.openFile10)
        self.pushButton_26.clicked.connect(self.calibrater)
        self.pushButton_200.clicked.connect(self.predictor)
        print("a")
        # self.setFocusPolicy(Qt.StrongFocus)
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    #Initialize labels and default settings in GUI
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Image Directory (TIFF MultiLayer)"))
        self.label_3.setText(_translate("MainWindow", "Tuning"))
        self.label_4.setText(_translate("MainWindow", "Min Area"))
        self.label_5.setText(_translate("MainWindow", "Max Area"))
        self.lineEdit_2.setText(_translate("MainWindow", "3500"))
        self.lineEdit_2.setPlaceholderText(_translate("MainWindow", "3500"))
        self.lineEdit_3.setText(_translate("MainWindow", "30000"))
        self.lineEdit_4.setText(_translate("MainWindow", "100"))
        self.lineEdit_4.setPlaceholderText(_translate("MainWindow", "100"))
        self.label_6.setText(_translate("MainWindow", "X-Min"))
        self.label_7.setText(_translate("MainWindow", "X-Max"))
        self.lineEdit_5.setText(_translate("MainWindow", "2000"))
        self.lineEdit_5.setPlaceholderText(_translate("MainWindow", "2000"))
        self.label_8.setText(_translate("MainWindow", "Y-Max"))
        self.lineEdit_6.setText(_translate("MainWindow", "2000"))
        self.lineEdit_6.setPlaceholderText(_translate("MainWindow", "2000"))
        self.label_9.setText(_translate("MainWindow", "Y-Min"))
        self.lineEdit_7.setText(_translate("MainWindow", "100"))
        self.lineEdit_7.setPlaceholderText(_translate("MainWindow", "100"))
        self.pushButton.setText(_translate("MainWindow", "Search"))
        self.pushButton_2.setText(_translate("MainWindow", "Save"))
        self.label_10.setText(_translate("MainWindow", "Save Directory"))
        self.pushButton_3.setText(_translate("MainWindow", "Generate"))
        self.pushButton_4.setText(_translate("MainWindow", "Apply"))
        self.label_69.setText(_translate("MainWindow", "Usage:"))
        self.radioButton_9.setText(_translate("MainWindow", "Calibration and Prediction"))
        self.radioButton_200.setText(_translate("MainWindow", "2.8"))
        self.radioButton_201.setText(_translate("MainWindow", "4.5"))
        self.radioButton_202.setText(_translate("MainWindow", "Both"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Nanowell Slicer"))
        self.label_11.setText(_translate("MainWindow", "Accepted:"))
        self.label_12.setText(_translate("MainWindow", "Total:"))
        self.label_13.setText(_translate("MainWindow", "Rejections: "))
        self.label_14.setText(_translate("MainWindow", "Images Left:"))
        self.pushButton_5.setText(_translate("MainWindow", "Search"))
        self.label_15.setText(_translate("MainWindow", "Image Directory (Save Directory)"))
        self.label_16.setText(_translate("MainWindow", "Save Directory"))
        self.pushButton_6.setText(_translate("MainWindow", "Save"))
        self.label_17.setText(_translate("MainWindow", "Start Position"))
        self.lineEdit_11.setText(_translate("MainWindow", "1"))
        self.lineEdit_11.setPlaceholderText(_translate("MainWindow", "1"))
        self.pushButton_7.setText(_translate("MainWindow", "Start"))
        self.label_19.setText(_translate("MainWindow", "Press W to reject"))
        self.label_20.setText(_translate("MainWindow", "Press E to accept"))
        self.label_21.setText(_translate("MainWindow", "Training"))
        self.label_22.setText(_translate("MainWindow", "Data Directory"))
        self.pushButton_8.setText(_translate("MainWindow", "Search"))
        self.label_23.setText(_translate("MainWindow", "Augmentation:"))
        self.label_25.setText(_translate("MainWindow", "Training Size"))
        self.label_26.setText(_translate("MainWindow", "Dataset Size:"))
        self.pushButton_9.setText(_translate("MainWindow", "Apply"))
        self.label_27.setText(_translate("MainWindow", "Testing Size:"))
        self.check_100.setText(_translate("MainWindow", "90"))
        self.check_101.setText(_translate("MainWindow", "180"))
        self.check_102.setText(_translate("MainWindow", "270"))
        self.label_24.setText(_translate("MainWindow", "Rotation"))
        self.label_55.setText(_translate("MainWindow", "Flip"))
        self.check_103.setText(_translate("MainWindow", "Horizontal"))
        self.check_104.setText(_translate("MainWindow", "Vertical"))
        self.pushButton_19.setText(_translate("MainWindow", "Train"))
        self.pushButton_20.setText(_translate("MainWindow", "Search"))
        self.label_56.setText(_translate("MainWindow", "Model Save Directory"))
        self.label_61.setText(_translate("MainWindow", "Inspection"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Inspector and Training"))
        self.pushButton_21.setText(_translate("MainWindow", "Search"))
        self.pushButton_22.setText(_translate("MainWindow", "Test"))
        self.label_57.setText(_translate("MainWindow", "Testing Directory"))
        self.label_58.setText(_translate("MainWindow", "Model File"))
        self.pushButton_23.setText(_translate("MainWindow", "Search"))
        self.label_59.setText(_translate("MainWindow", "Dice 2.8um: "))
        self.label_60.setText(_translate("MainWindow", "Dice 4.5um:"))
        self.label_62.setText(_translate("MainWindow", "Testing"))
        self.label_63.setText(_translate("MainWindow", "Fluorescent Calibration and Prediction"))
        self.pushButton_24.setText(_translate("MainWindow", "Search"))
        self.label_64.setText(_translate("MainWindow", "Model File"))
        self.label_65.setText(_translate("MainWindow", "Data Directory"))
        self.pushButton_25.setText(_translate("MainWindow", "Search"))
        self.pushButton_26.setText(_translate("MainWindow", "Calibrate"))
        self.pushButton_27.setText(_translate("MainWindow", "Search"))
        self.pushButton_200.setText(_translate("MainWindow", "Predict"))
        self.label_66.setText(_translate("MainWindow", "Fluorescent Predictions"))
        self.label_67.setText(_translate("MainWindow", "Data Directory"))
        self.label_101.setText(_translate("MainWindow", "Brightfield"))
        self.label_102.setText(_translate("MainWindow", "2.8 um"))
        self.label_100.setText(_translate("MainWindow", "4.5 um"))
        self.label_103.setText(_translate("MainWindow", "Image Number"))
        self.label_104.setText(_translate("MainWindow", "Progress: "))
        self.label_400.setText(_translate("MainWindow", "Progress: "))
        self.label_401.setText(_translate("MainWindow", "Please ensure file with Tiffs is named \"Tiffs\" and located in Data Directory"))
        self.label_105.setText(_translate("MainWindow", "Auto"))
        self.lineEdit_105.setText(_translate("MainWindow", "9000"))
        self.lineEdit_105.setPlaceholderText(_translate("MainWindow", "9000"))
        self.check_105.setText(_translate("MainWindow", "Test"))
        self.label_106.setText(_translate("MainWindow", "Offset"))
        self.pushButton_28.setText(_translate("MainWindow", "Plot"))
        self.pushButton_106.setText(_translate("MainWindow", "Enter"))
        self.label_107.setText(_translate("MainWindow", "Name:"))
        self.label_107.setText(_translate("MainWindow", "Name:"))
        self.label_108.setText(_translate("MainWindow", "Progress: "))
        self.label_200.setText(_translate("MainWindow", "Thresh"))
        self.label_300.setText(_translate("MainWindow", "C-k1"))
        self.label_301.setText(_translate("MainWindow", "C-Iter"))
        self.label_302.setText(_translate("MainWindow", "O-k2"))
        self.lineEdit_300.setText(_translate("MainWindow", "3"))
        self.lineEdit_301.setText(_translate("MainWindow", "10"))
        self.lineEdit_302.setText(_translate("MainWindow", "40"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "Testing and Prediction"))

    #Check for a key press during qualtiy inspection
    def keyPressEvent(self, event):
        if isinstance(event, QKeyEvent):
            key_text = event.text()
            print(str(key_text))

    #Check if key is released after being pressed during quality inspection
    def keyReleaseEvent(self, event):
        if isinstance(event, QKeyEvent):
            key_text = event.text()
            print(str(key_text))

    #Overide for checking key presses
    def event(self, event):
        if (event.type() == QEvent.KeyPress) and (event.key() == Qt.Key_Space):
            print('Parent handling space')
            return True
        return QWidget.event(self, event)

    #Overide for relaying key presses
    def eventFilter(self, widget, event):
        if (event.type() == QEvent.KeyPress) and (event.key() == Qt.Key_Space):
            print('Sending space event to parent...')
            self.event(event)
            return True
        return super(Ui_MainWindow, self).eventFilter(widget, event)

    #Store the offset value for cropping
    def upoffset(self):
        global offseta
        if (self.lineEdit_106.text() != ''):
            try:
                offseta = int(self.lineEdit_106.text())
            except ValueError:
                print("Please enter float!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

    #Computer vision function to identify nanowells
    def find_nanowell(BF, opening, save, show1 = False):
        print("params", minA, maxA, Xmin, Xmax, Ymin, Ymax)
        # area range of a square/nanowell
        min_area = minA
        max_area = maxA
        # final segmented nanowell size: nanowell_size=height=width
        nanowell_size = 160
        # adjust coordinates of identified nanowells
        offset_x = -10
        offset_y = -10
        # find all nanowells in a restricted region
        xMin, xMax = Xmin, Xmax
        yMin, yMax = Ymin, Ymax

        # get all contours. Only the outer contours will be returned.
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        image_number = 0  # nanowell counts

        imgcopy = np.asarray(BF).copy()
        imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_GRAY2BGR)

        centroidsBF = []
        half_size = nanowell_size // 2
        for c in cnts:
            area = cv2.contourArea(c)
            if area > min_area and area < max_area:
                # compute the center of the contour
                M = cv2.moments(c)
                x = int(M["m10"] / M["m00"]) + offset_x
                y = int(M["m01"] / M["m00"]) + offset_y
                if xMin < x < xMax and yMin < y < yMax:
                    centroidsBF.append((x, y))
                    # visulaize segmented nanowells
                    imgcopy = cv2.rectangle(imgcopy, (x - half_size, y - half_size),(x + half_size, y + half_size), (0, 0, 255), 3)
                    image_number += 1

        #if the image is being displayed on the GUI
        if show1:
            #Resize image, convert to Pixmap, and display on GUI
            imgcopy = (cv2.resize(imgcopy, (850, 750), interpolation = cv2.INTER_LINEAR)*4).astype("uint8")
            cv2.imwrite(save+"pic.jpg", imgcopy)
            pixmap = QPixmap(save+"pic.jpg")
            return centroidsBF, pixmap
        else:
            return centroidsBF,None

    #Crop the nanowells and store as individual images
    def crop_squares(image, centroids, square_size, save_path, extend = False, offset=0):
        for i in range(len(centroids)):
            x, y = centroids[i]

            # Calculate the top-left corner of the square
            top_left_x = int(x - square_size / 2)
            top_left_y = int(y - square_size / 2)

            # Crop the square
            square = image[top_left_y:top_left_y + square_size, top_left_x:top_left_x + square_size]

            square_save = save_path + str(i + offset) + '.jpg'
            square_save2 = save_path + str(i + offset) + '.tif'
            if (extend):
                # image12 = cv2.cvtColor(square, cv2.CV_16U)
                cv2.imwrite(square_save2, square)
            else:
                cv2.imwrite(square_save, square)

    #training setup for the U-Net
    def trainSize(self):
        global trainS, numberF, moveOn

        #User input error checking

        if self.fname2 == "":
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure a Data Directory is provided.")
            dialog.setWindowTitle("Error")
            dialog.exec()
            return

        if self.sname2 == "":
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure a save Directory is provided.")
            dialog.setWindowTitle("Error")
            dialog.exec()
            return

        if (self.lineEdit_107.text() != ''):
            try:
                self.modelName = str(self.lineEdit_107.text())
            except ValueError:
                print("Please enter float!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure a save name is provided.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (self.lineEdit_13.text() != ''):
            try:
                trainS = int(self.lineEdit_13.text())
            except ValueError:
                print("Please enter float!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure a training size (integer) is provided.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (trainS <= 0 or trainS > numberF):
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure a training size between 1 and " + str(numberF))
            dialog.setWindowTitle("Error")
            dialog.exec()
            return

        #Display Testing Size in GUI
        self.label_27.setText("Testing Size:" + str(numberF-trainS))
        moveOn = True

    #Train the U-Net
    def trainer(self):
        global trainS, numberF, moveOn

        #Check if all required setup is complete
        if (not(moveOn)):
            return

        #Initialize the model name
        modName = self.sname2 + "\\" + self.modelName + ".keras"

        #Partition the training and testing image file names
        totalfiles = [f for f in listdir(self.fname2 + "\\goodBF\\") if isfile(join(self.fname2 + "\\goodBF\\", f))]
        afiles = []
        testfiles = []
        for p in range(len(totalfiles)):
            if (p < trainS):
                afiles.append(totalfiles[p])
            else:
                testfiles.append(totalfiles[p])

        #Initialize the data locations
        basedir1 = self.fname2 + "\\goodBF\\"
        basedir2 = self.fname2 + "\\goodGT\\"

        #Initialize storage arrays
        allImagesBF = []
        allImagesGT = []
        allTestImagesBF = []
        allTestImagesGT = []

        for ss in range(len(afiles)):
            #Read images
            img = cv2.imread(basedir1 + afiles[ss], cv2.IMREAD_COLOR)
            img2 = cv2.imread(basedir2 + afiles[ss], cv2.IMREAD_GRAYSCALE)
            allImagesBF.append(img)
            allImagesGT.append(img2)

            #Augment images through rotation and reflection

            if(self.check_100.isChecked()):
                img3 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                img4 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
                allImagesBF.append(img3)
                allImagesGT.append(img4)

            if(self.check_101.isChecked()):
                img5 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                img6 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
                img5 = cv2.rotate(img5, cv2.ROTATE_90_CLOCKWISE)
                img6 = cv2.rotate(img6, cv2.ROTATE_90_CLOCKWISE)
                allImagesBF.append(img5)
                allImagesGT.append(img6)


            if(self.check_102.isChecked()):
                img7 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                img8 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
                img7 = cv2.rotate(img7, cv2.ROTATE_90_CLOCKWISE)
                img8 = cv2.rotate(img8, cv2.ROTATE_90_CLOCKWISE)
                img7 = cv2.rotate(img7, cv2.ROTATE_90_CLOCKWISE)
                img8 = cv2.rotate(img8, cv2.ROTATE_90_CLOCKWISE)
                allImagesBF.append(img7)
                allImagesGT.append(img8)

            if(self.check_103.isChecked()):
                img9 = cv2.flip(img, 0)
                img10 = cv2.flip(img2, 0)
                allImagesBF.append(img9)
                allImagesGT.append(img10)

            if(self.check_104.isChecked()):
                img11 = cv2.flip(img, 1)
                img12 = cv2.flip(img2, 1)
                allImagesBF.append(img11)
                allImagesGT.append(img12)

        #Initialize save paths
        save_patha = self.sname2 + "\\TestBF\\"
        save_pathb = self.sname2 + "\\TestGT\\"
        if not os.path.exists(save_patha):
            os.makedirs(save_patha)
        if not os.path.exists(save_pathb):
            os.makedirs(save_pathb)

        #Store the test images separately
        for kk in range(len(testfiles)):
            img = cv2.imread(basedir1 + testfiles[kk], cv2.IMREAD_COLOR)
            img2 = cv2.imread(basedir2 + testfiles[kk], cv2.IMREAD_GRAYSCALE)
            cv2.imwrite( save_patha + testfiles[kk], img)
            cv2.imwrite( save_pathb + testfiles[kk], img2)
            allTestImagesBF.append(img)
            allTestImagesGT.append(img2)

        #Build U0Net
        model = Model.build_unet((160, 160, 3), 2)
        print(model.summary())

        #Setup Training datasets

        Height = 160
        Width = 160
        NumofCategories = 3

        allImages = []
        maskImages = []

        allTestImages = []
        maskTestImages = []

        #normalize brightfield images and convert to float32
        for img1 in allImagesBF:
            img = (img1 / 255.0)
            img = img.astype(np.float32)
            allImages.append(img)

        #Generate the mask images by ensuring that 4.5um beads are 127 intensity and 2.9um beads are 255 intensity
        #Ensure the image is a uint8 type
        for mask1 in allImagesGT:
            ret, GG = cv2.threshold(mask1.copy(), 10, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
            ret, GG1 = cv2.threshold(mask1.copy(), 200, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
            GGF = cv2.subtract(GG, GG1)
            GGF = GGF / 2
            GGF = GGF.astype("uint8")
            mask = cv2.add(GGF, GG1)
            mask = (mask / 127) + 1
            mask = mask.astype("uint8")
            maskImages.append(mask)

        #Cast to numpy arrays
        allImagesNP = np.array(allImages)
        maskImagesNP = np.array(maskImages)
        maskImagesNP = maskImagesNP.astype(int)

        print(allImagesNP.shape)
        print(allImagesNP.dtype)
        print(maskImagesNP.shape)
        print(maskImagesNP.dtype)
        print(maskImagesNP[0].dtype)

        #Display an image to show an example of the data to ensure data is correctly handled
        x = cv2.resize(maskImagesNP[0], (18, 18), interpolation=cv2.INTER_NEAREST)
        for i in range(len(x)):
            for j in range(len(x[i])):
                v = x[i][j]

                if (v == 1):
                    x[i][j] = 0

                if (v == 2):
                    x[i][j] = 22

                if (v == 3):
                    x[i][j] = 333
        print(x)

        #normalize brightfield images and convert to float32 (test images)
        for img1 in allTestImagesBF:
            img = img1 / 255.0
            img = img.astype(np.float32)
            allTestImages.append(img)

        #Generate the mask images by ensuring that 4.5um beads are 127 intensity and 2.9um beads are 255 intensity
        #Ensure the image is a uint8 type (test images)
        for mask1 in allImagesGT:
            ret, GG = cv2.threshold(mask1.copy(), 10, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
            ret, GG1 = cv2.threshold(mask1.copy(), 200, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
            GGF = cv2.subtract(GG, GG1)
            GGF = GGF / 2
            GGF = GGF.astype("uint8")
            mask = cv2.add(GGF, GG1)
            mask = (mask / 127) + 1
            mask = mask.astype("uint8")
            maskTestImages.append(mask)

        #Cast to numpy arrays
        allTestImagesNP = np.array(allTestImages)
        maskTestImagesNP = np.array(maskTestImages)
        maskTestImagesNP = maskTestImagesNP.astype(int)

        print(allTestImagesNP.shape)
        print(allTestImagesNP.dtype)

        print(maskTestImagesNP.shape)
        print(maskTestImagesNP.dtype)

        #Display an image to show an example of the data to ensure data is correctly handled
        x = cv2.resize(maskTestImagesNP[1], (24, 24), interpolation=cv2.INTER_NEAREST)
        for i in range(len(x)):
            for j in range(len(x[i])):
                v = x[i][j]

                if (v == 1):
                    x[i][j] = 0

                if (v == 2):
                    x[i][j] = 22

                if (v == 3):
                    x[i][j] = 333

        print(x)

        # np.save("C:\\Users\\Jaden\\Desktop\\Unet-train-images2.npy", allImagesNP)
        # np.save("C:\\Users\\Jaden\\Desktop\\Unet-train-mask2.npy", maskImagesNP)


        #Save the testing data
        np.save(self.sname2 + "\\TestBF.npy" , allTestImagesNP)
        np.save(self.sname2 + "\\TestGT.npy", maskTestImagesNP)

        #Initialize training parameters
        Weight = 160
        Width = 160
        numofCategories = 3

        #Convert imags to categorical
        test = maskImagesNP[0]
        test = test - 1
        test2 = to_categorical(test, num_classes=numofCategories)

        print(test)
        print(test2)

        maskImagesNP = maskImagesNP - 1
        maskForTheModel = to_categorical(maskImagesNP, num_classes=numofCategories)

        print(maskForTheModel.dtype)
        maskForTheModel = maskForTheModel.astype("int32")
        print(maskForTheModel.dtype)
        print(maskImagesNP.dtype)

        #Create the final training testing dataset for training
        X_train, X_val, y_train, y_val = train_test_split(allImagesNP, maskForTheModel, test_size=0.1, random_state=42)

        print(X_train.shape)
        print(y_train.shape)
        print("hi", X_train.dtype)
        print(y_train.dtype)

        print(X_val.shape)
        print(y_val.shape)

        #Initialize image shape and classes
        shape = (160, 160, 3)
        num_classes = 3

        #Modifiable parameters
        lr = 1e-4
        batch_size = 4
        epochs = 10

        #Build U_Net model
        model = Model.build_unet(shape, num_classes)
        print(model.summary())
        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr), metrics=['accuracy'])

        #Setup training parameters
        stepsPerEpoch = np.ceil(len(X_train) / batch_size)
        validationSteps = np.ceil(len(X_val) / batch_size)

        #Setup training to save the best model and adapt during training
        best_model_file = modName
        callbacks = [
            ModelCheckpoint(best_model_file, verbose=1, save_best_only=True),
            ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
            EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        ]

        print("hi ", stepsPerEpoch.dtype)
        print(validationSteps.dtype)
        print(X_train.dtype)
        print(y_train.dtype)
        print(X_val.dtype)
        print(y_val.dtype)
        print(X_train.shape)
        print(y_train.shape)
        print(X_val.shape)
        print(y_val.shape)

        #Train the U-Net model
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                            validation_data=(X_val, y_val), shuffle=True, callbacks=callbacks)

        #Extract accuracy and loss in training and validation over epochs
        #Plot and save as an image

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'r', label="Train_Accuracy")
        plt.plot(epochs, val_acc, 'b', label="Validation_Accuracy")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title("Train and Validation Accuracy")
        plt.legend(loc='lower right')
        plt.savefig(self.sname2 + "\\Train_Acc_plot.jpg")

        plt.plot(epochs, loss, 'r', label="Train_Loss")
        plt.plot(epochs, val_loss, 'b', label="Validation_Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("Train and Validation Loss")
        plt.legend(loc='upper right')
        plt.savefig(self.sname2 + "\\Train_Los_plot.jpg")

#File location intake functions

    def openFile1(self):
        self.fname = QFileDialog.getExistingDirectory(None,  "Select one or more files to open", "C:\\Windows")

        if self.fname:
            self.lineEdit.setText(str(self.fname))

    def openFile2(self):
        self.sname = QFileDialog.getExistingDirectory(None,  "Select one or more files to open", "C:\\Windows")

        if self.sname:
            self.lineEdit_8.setText(str(self.sname))

    def openFile3(self):
        self.fname1 = QFileDialog.getExistingDirectory(None,  "Select one or more files to open", "C:\\Windows")

        if self.fname1:
            self.lineEdit_10.setText(str(self.fname1))

    def openFile4(self):
        self.sname1 = QFileDialog.getExistingDirectory(None,  "Select one or more files to open", "C:\\Windows")

        if self.sname1:
            self.lineEdit_9.setText(str(self.sname1))

    def openFile5(self):
        global numberF
        self.fname2 = QFileDialog.getExistingDirectory(None,  "Select one or more files to open", "C:\\Windows")

        if self.fname2:
            self.lineEdit_12.setText(str(self.fname2))
            afiles = [f for f in listdir(self.fname2 + "\\goodBF\\") if isfile(join(self.fname2 + "\\goodBF\\", f))]
            numberF = len(afiles)
            self.label_26.setText("Dataset size: " + str(numberF))

    def openFile6(self):
        self.sname2 = QFileDialog.getExistingDirectory(None,  "Select one or more files to open", "C:\\Windows")

        if self.sname2:
            self.lineEdit_27.setText(str(self.sname2))

    def openFile7(self):
        self.fname3 = QFileDialog.getExistingDirectory(None,  "Select one or more files to open", "C:\\Windows")

        if self.fname3:
            self.lineEdit_28.setText(str(self.fname3))

    def openFile8(self):
        self.sname3 = QFileDialog.getOpenFileName(None,  "Select one or more files to open", "C:\\Windows")[0]

        if self.sname3:
            self.lineEdit_29.setText(str(self.sname3))

    def openFile9(self):
        self.fname4 = QFileDialog.getExistingDirectory(None,  "Select one or more files to open", "C:\\Windows")

        if self.fname4:
            self.lineEdit_31.setText(str(self.fname4))

    def openFile10(self):
        self.sname4 = QFileDialog.getOpenFileName(None,  "Select one or more files to open", "C:\\Windows")[0]

        if self.sname4:
            self.lineEdit_30.setText(str(self.sname4))

    def openFile11(self):
        self.fname5 = QFileDialog.getExistingDirectory(None,  "Select one or more files to open", "C:\\Windows")

        if self.fname5:
            self.lineEdit_32.setText(str(self.fname5))

    #Plotting function that combines all results and locations and stitches nanowell images together
    def combiner(self):
        if self.fname5 == "":
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure an Image Directory is provided.")
            dialog.setWindowTitle("Error")
            dialog.exec()
            return

        #Combine fluorescent results and nanowell locations into one spreadsheet and save
        df1 = pd.read_csv(self.fname5 + "\\FluoroResults.csv")
        df2 = pd.read_csv(self.fname5 + "\\NanowellLocations.csv")
        merged = df1.merge(df2, how = 'inner', on = 'Name')
        merged.to_csv(self.fname5 + "\\CombinedResults.csv", index=False)

        #Extract fluorescent intensities and locations, and maximum intensities from combine results
        namess = merged['Name'].tolist()
        smallbead = merged['2.8Int'].tolist()
        smallbead = list(map(float, smallbead))
        largebead = merged['4.5Int'].tolist()
        largebead = list(map(float, largebead))
        Xloc = merged['COMX'].tolist()
        Xloc = list(map(int, Xloc))
        Yloc = merged['COMY'].tolist()
        Yloc = list(map(int, Yloc))
        tiffer = merged['TIFF'].tolist()
        print(smallbead)
        print(largebead)
        maxint28 = max(smallbead)
        maxint45 = max(largebead)
        if(maxint28 == 0):
            maxint28 = 1
        if(maxint45 == 0):
            maxint45 = 1

        #Store the results for each image in an array where they are sorted based on tiff image
        locations = []
        intensities = []
        tiffImg = []
        tiffys = []
        allnames = []
        for j in range(len(smallbead)):
            tiffname = self.fname5 + "\\Tiffs\\" + tiffer[j]
            if tiffname in tiffImg:
                locations[tiffImg.index(tiffname)].append([Xloc[j], Yloc[j]])
                intensities[tiffImg.index(tiffname)].append([smallbead[j], largebead[j]])
                allnames[tiffImg.index(tiffname)].append(namess[j])

            else:
                tiffys.append(tiffer[j])
                tiffImg.append(tiffname)
                locations.append([[Xloc[j], Yloc[j]]])
                intensities.append([[smallbead[j], largebead[j]]])
                allnames.append([namess[j]])

        #Stitch the boxes of normalized intensities and the predicted segmentations with normalized intensities
        half_size = 80
        ast = len(tiffImg)
        for p in range(len(tiffImg)):
            #Setup progress bar
            self.label_400.setText("Progress: " + str((int(100*p/ast))))
            QtWidgets.qApp.processEvents()

            #Initialize the brightfield image for each setting being stitched
            ret, images = cv2.imreadmulti(tiffImg[p], [], cv2.IMREAD_ANYDEPTH)
            br1 = cv2.cvtColor((images[0]/256).astype("uint8"), cv2.COLOR_GRAY2BGR)
            br2 = cv2.cvtColor((images[0]/256).astype("uint8"), cv2.COLOR_GRAY2BGR)
            br3 = cv2.cvtColor((images[0]/256).astype("uint8"), cv2.COLOR_GRAY2BGR)
            br4 = cv2.cvtColor((images[0]/256).astype("uint8"), cv2.COLOR_GRAY2BGR)
            img1 = cv2.cvtColor((images[0]/256).astype("uint8"), cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor((images[0]/256).astype("uint8"), cv2.COLOR_GRAY2BGR)
            img3 = cv2.cvtColor((images[0]/256).astype("uint8"), cv2.COLOR_GRAY2BGR)
            img4 = cv2.cvtColor((images[0]/256).astype("uint8"), cv2.COLOR_GRAY2BGR)
            predImg = cv2.cvtColor((images[0]/256).astype("uint8"), cv2.COLOR_GRAY2BGR)
            predImg2 = cv2.cvtColor((images[0]/256).astype("uint8"), cv2.COLOR_GRAY2BGR)

            #Create a blank pocture for stitching images together before overlaying
            ggst = np.expand_dims(np.zeros((predImg.shape[0], predImg.shape[1])), axis=-1)
            canvas = np.concatenate([ggst, ggst, ggst], axis=2)
            canvas2 = np.concatenate([ggst, ggst, ggst], axis=2)
            for k in range(len(locations[p])):
                arr28 = np.asarray(intensities[p])
                new28 = arr28[:,0]
                maxa28 = max(new28)
                new45 = arr28[:,1]
                maxa45 = max(new45)
                if (maxa28 == 0):
                    maxa28 = 1
                if (maxa45 == 0):
                    maxa45 = 1

                #Normalized intensity overall and local
                gamma1 = intensities[p][k][0]/maxint28
                gamma2 = intensities[p][k][1]/maxint45
                gamma3 = intensities[p][k][0]/maxa28
                gamma4 = intensities[p][k][1]/maxa45
                x = locations[p][k][0]
                y = locations[p][k][1]
                brt = cv2.imread(self.fname5 + "\\predictedMaskCalibPred\\" + allnames[p][k], cv2.IMREAD_GRAYSCALE)

                #Stitch the prediction masks with different colors and intensities together

                ret, GG = cv2.threshold(brt.copy(), 10, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                ret, GG1 = cv2.threshold(brt.copy(), 200, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                GGF = cv2.subtract(GG, GG1)
                GGF = np.expand_dims(GGF.copy(), axis=-1)
                bst = np.expand_dims(np.zeros((brt.shape[0], brt.shape[1])), axis=-1)
                HH_img = np.concatenate([gamma2*GGF, gamma2*GGF, bst], axis=2)
                HH_img = HH_img.astype("uint8")
                a = GG1.copy()
                GG1 = np.expand_dims(GG1.copy(), axis=-1)
                DD_img = np.concatenate([gamma1*GG1, bst, gamma1*GG1], axis=2)
                DD_img = DD_img.astype("uint8")
                finalimg = cv2.add(DD_img, HH_img)
                canvas[y - half_size:y + half_size, x - half_size:x + half_size] = finalimg

                HH_img1 = np.concatenate([gamma4*GGF, gamma4*GGF, bst], axis=2)
                HH_img1 = HH_img1.astype("uint8")
                GG11 = np.expand_dims(a, axis=-1)
                DD_img1 = np.concatenate([gamma3*GG11, bst, gamma3*GG11], axis=2)
                DD_img1 = DD_img1.astype("uint8")
                finalimg1 = cv2.add(DD_img1, HH_img1)
                canvas2[y - half_size:y + half_size, x - half_size:x + half_size] = finalimg1

                #Stitch the boxes onto the image with normalized intensities
                br1 = cv2.rectangle(br1, (x - half_size, y - half_size), (x + half_size, y + half_size), (128, 128, 255*gamma1), -1)
                br2 = cv2.rectangle(br2, (x - half_size, y - half_size), (x + half_size, y + half_size), (128, 128, 255*gamma2), -1)
                br3 = cv2.rectangle(br3, (x - half_size, y - half_size), (x + half_size, y + half_size), (128, 128, 255*gamma3), -1)
                br4 = cv2.rectangle(br4, (x - half_size, y - half_size), (x + half_size, y + half_size), (128, 128, 255*gamma4), -1)

            #Stitch the prediction masks with normalized intensities
            canvas = canvas.astype("uint8")
            predImg = cv2.addWeighted(canvas, 0.6, predImg*5, 0.4, 0)
            canvas2 = canvas2.astype("uint8")
            predImg2 = cv2.addWeighted(canvas2, 0.6, predImg2*5, 0.4, 0)
            img1 = cv2.addWeighted(br1, 0.6, img1, 0.4, 0)
            img2 = cv2.addWeighted(br2, 0.6, img2, 0.4, 0)
            img3 = cv2.addWeighted(br3, 0.6, img3, 0.4, 0)
            img4 = cv2.addWeighted(br4, 0.6, img4, 0.4, 0)

            #Save the stitched images
            naem = [*tiffys[p]]
            for i in range(4):
                del naem[-1]
            tiffname = ''.join(naem)
            if not os.path.exists(self.fname5 + "\\Intensity28OVR\\"):
                os.makedirs(self.fname5 + "\\Intensity28OVR\\")
            cv2.imwrite(self.fname5 + "\\Intensity28OVR\\" + tiffname + ".jpg", img1)

            if not os.path.exists(self.fname5 + "\\Intensity45OVR\\"):
                os.makedirs(self.fname5 + "\\Intensity45OVR\\")
            cv2.imwrite(self.fname5 + "\\Intensity45OVR\\" + tiffname + ".jpg", img2)

            if not os.path.exists(self.fname5 + "\\Intensity28LOC\\"):
                os.makedirs(self.fname5 + "\\Intensity28LOC\\")
            cv2.imwrite(self.fname5 + "\\Intensity28LOC\\" + tiffname + ".jpg", img3)

            if not os.path.exists(self.fname5 + "\\Intensity45LOC\\"):
                os.makedirs(self.fname5 + "\\Intensity45LOC\\")
            cv2.imwrite(self.fname5 + "\\Intensity45LOC\\" + tiffname + ".jpg", img4)

            if not os.path.exists(self.fname5 + "\\OverallPreds\\"):
                os.makedirs(self.fname5 + "\\OverallPreds\\")
            cv2.imwrite(self.fname5 + "\\OverallPreds\\" + tiffname + ".jpg", predImg)

            if not os.path.exists(self.fname5 + "\\LocPreds\\"):
                os.makedirs(self.fname5 + "\\LocPreds\\")
            cv2.imwrite(self.fname5 + "\\LocPreds\\" + tiffname + ".jpg", predImg2)

    #Calibrates by extracting the fluorescent intensity for different biomolecule concentrations
    def calibrater(self):

        #Error checking user inputs

        if self.fname4 == "":
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure an Image Directory is provided.")
            dialog.setWindowTitle("Error")
            dialog.exec()
            return

        if self.sname4 == "":
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure a model files (.keras) is provided.")
            dialog.setWindowTitle("Error")
            dialog.exec()
            return

        threshpix = 0
        if (self.lineEdit_200.text() != ''):
            try:
                 threshpix = int(self.lineEdit_200.text())
            except ValueError:
                print("Please enter integer!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers for the thresh.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (threshpix < 0):
            threshpix = 0

        filt45 = False
        filt28 = False
        if (self.radioButton_200.isChecked()):
            filt45 = True
        elif (self.radioButton_201.isChecked()):
            filt28 = True
        else:
            print("Both Chosen")

        #open the model file for predictions
        best_model_file = self.sname4
        model = tf.keras.models.load_model(best_model_file)
        print(model.summary())

        #Initialize images
        allTestImagesNP = [f for f in listdir(self.fname4 + "\\NanoBFCalibPred\\") if isfile(join(self.fname4 + "\\NanoBFCalibPred\\", f))]
        concents = []
        fluovals = []
        breakconcents = []

        #Write results to a CSV
        with open(self.fname4 + '\\FluoroResults.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Glu", "Ins", "2.8Int", "4.5Int"])

            # Generate segmentation predictions with model and extract the fluorescent intensity from the stored 16 bit images
            for i in range(len(allTestImagesNP)):
                self.label_108.setText("Progress: " + str(int(i*100/len(allTestImagesNP))))
                QtWidgets.qApp.processEvents()
                namea = allTestImagesNP[i]
                Image12 = cv2.imread(self.fname4 + "\\NanoBFCalibPred\\" + allTestImagesNP[i])  # , cv2.IMREAD_COLOR)#)
                splitup = [*allTestImagesNP[i]]
                for j in range(3):
                    del splitup[-1]
                word1 = str(''.join(splitup))
                nameb = word1 + "tif"
                fluo1 = cv2.imread(self.fname4 + "\\Nano28FlCalibPred\\" + nameb, cv2.IMREAD_ANYDEPTH)
                fluo2 = cv2.imread(self.fname4 + "\\Nano45FlCalibPred\\" + nameb, cv2.IMREAD_ANYDEPTH)

                #Extract biomolecule concentrations from nanowell image names

                while splitup[-1] != "_":
                    del splitup[-1]

                del splitup[-1]

                index = 0
                glut = []
                insult = []
                switcho = True
                for g in range(len(splitup)):
                    if (switcho):
                        glut.append(splitup[g])
                    else:
                        insult.append(splitup[g])

                    if (splitup[g] == '_'):
                        index = g
                        switcho = False

                del glut[-1]
                insulconc = str(''.join(insult))
                glutconc = str(''.join(glut))

                #Conduct segmentation predictions

                Image = Image12 / 255.0
                Image = Image.astype(np.float32)

                img = Image.copy()
                imgForModel = np.expand_dims(img, axis=0)

                p = model.predict(imgForModel)
                resultMask = p[0]
                resultMask = np.argmax(resultMask, axis=-1)
                resultMask = np.expand_dims(resultMask, axis=-1)
                resultMask = resultMask * (255 / 3)
                resultMask = resultMask.astype(np.uint8)
                x = cv2.resize(resultMask, (16, 16), interpolation=cv2.INTER_NEAREST)
                predictedMaskImg = np.concatenate([resultMask, resultMask, resultMask], axis=2)
                predictedMaskImg = cv2.cvtColor(predictedMaskImg, cv2.COLOR_BGR2GRAY)
                predictedMaskImg = ((predictedMaskImg / 170) * 255).astype("uint8")
                print(predictedMaskImg.shape)

                #Extract the fluorescent intensities using the 2.8um mask and the 4.5um mask and take the average

                ret, GG1 = cv2.threshold(predictedMaskImg.copy(), 200, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                gl = np.multiply(fluo1, (GG1 / 255).astype("uint16"))
                if (len(gl[gl>threshpix]) == 0):
                    aa28 = 0
                else:
                    aa28 = np.sum(gl[gl>threshpix]) / len(gl[gl>threshpix])

                ret, GC1 = cv2.threshold(predictedMaskImg.copy(), 200, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                ret, GC = cv2.threshold(predictedMaskImg.copy(), 10, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                GA1 = cv2.subtract(GC, GC1)
                kl = np.multiply(fluo2, (GA1 / 255).astype("uint16"))
                if (len(kl[kl>threshpix]) == 0):
                    aa45 = 0
                else:
                    aa45 = np.sum(kl[kl>threshpix]) / len(kl[kl>threshpix])

                #Filter results for to remove incorrect segmentations from calculations
                if (not(filt28 and aa28 != 0) and not(filt45 and aa45 != 0)):
                    newname = str(''.join(splitup))
                    if newname in concents:
                        fluovals[concents.index(newname)].append([aa28, aa45])
                    else:
                        concents.append(newname)
                        breakconcents.append([glutconc,insulconc])
                        fluovals.append([[aa28, aa45]])

                    #Save prediction results and write results to a csv
                    if not os.path.exists(self.fname4 + "\\predictedMaskCalibPred\\"):
                        os.makedirs(self.fname4 + "\\predictedMaskCalibPred\\")
                    cv2.imwrite(self.fname4 + "\\predictedMaskCalibPred\\" + namea, predictedMaskImg)

                    if not os.path.exists(self.fname4 + "\\NIS28\\"):
                        os.makedirs(self.fname4 + "\\NIS28\\")
                    cv2.imwrite(self.fname4 + "\\NIS28\\" + nameb, gl)
                    print(gl.dtype)

                    if not os.path.exists(self.fname4 + "\\NIS45\\"):
                        os.makedirs(self.fname4 + "\\NIS45\\")
                    cv2.imwrite(self.fname4 + "\\NIS45\\" + nameb,kl)
                    print(kl.dtype)


        #Calculate the average results for each biomolecule concentration and write a CSV
        with open(self.fname4 + '\\FluoroAvgResults.csv', 'w', newline='') as file1:
            writer1 = csv.writer(file1)
            writer1.writerow(["Glu", "Ins", "2.8Int",  "2.8STD", "4.5Int", "4.5STD"])
            for jp in range(len(concents)):
                js = np.asarray(fluovals[jp])
                aat = js[:, 0]
                bbt = js[:, 1]
                writer1.writerow([breakconcents[jp][0], breakconcents[jp][1], np.mean(aat),  np.std(aat), np.mean(bbt), np.std(bbt)])

    #This function has the same execution as the calibrator function
    # Except does not have filtering and average results calculations since this is for predictions
    def predictor(self):

        if self.fname4 == "":
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure an Image Directory is provided.")
            dialog.setWindowTitle("Error")
            dialog.exec()
            return

        if self.sname4 == "":
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure a model files (.keras) is provided.")
            dialog.setWindowTitle("Error")
            dialog.exec()
            return

        threshpix = 0
        if (self.lineEdit_200.text() != ''):
            try:
                 threshpix = int(self.lineEdit_200.text())
            except ValueError:
                print("Please enter integer!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers for the thresh.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (threshpix < 0):
            threshpix = 0

        filt45 = False
        filt28 = False
        if (self.radioButton_200.isChecked()):
            filt45 = True
        elif (self.radioButton_201.isChecked()):
            filt28 = True
        else:
            print("Both Chosen")

        best_model_file = self.sname4
        model = tf.keras.models.load_model(best_model_file)
        print(model.summary())

        allTestImagesNP = [f for f in listdir(self.fname4 + "\\NanoBFCalibPred\\") if isfile(join(self.fname4 + "\\NanoBFCalibPred\\", f))]
        concents = []
        fluovals = []
        breakconcents = []
        with open(self.fname4 + '\\FluoroResults.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "2.8Int", "4.5Int"])

            for i in range(len(allTestImagesNP)):
                self.label_108.setText("Progress: " + str(int(i*100/len(allTestImagesNP))))
                QtWidgets.qApp.processEvents()
                namea = allTestImagesNP[i]
                Image12 = cv2.imread(self.fname4 + "\\NanoBFCalibPred\\" + allTestImagesNP[i])  # , cv2.IMREAD_COLOR)#)
                # Image = cv2.cvtColor(Image12, cv2.COLOR_GRAY2BGR)
                splitup = [*allTestImagesNP[i]]
                for j in range(3):
                    del splitup[-1]
                word1 = str(''.join(splitup))
                nameb = word1 + "tif"
                fluo1 = cv2.imread(self.fname4 + "\\Nano28FlCalibPred\\" + nameb, cv2.IMREAD_ANYDEPTH)
                fluo2 = cv2.imread(self.fname4 + "\\Nano45FlCalibPred\\" + nameb, cv2.IMREAD_ANYDEPTH)

                while splitup[-1] != "_":
                    del splitup[-1]

                del splitup[-1]

                index = 0
                glut = []
                insult = []
                switcho = True
                for g in range(len(splitup)):
                    if (switcho):
                        glut.append(splitup[g])
                    else:
                        insult.append(splitup[g])

                    if (splitup[g] == '_'):
                        index = g
                        switcho = False

                del glut[-1]
                insulconc = str(''.join(insult))
                glutconc = str(''.join(glut))

                Image = Image12 / 255.0
                Image = Image.astype(np.float32)

                img = Image.copy()
                imgForModel = np.expand_dims(img, axis=0)

                p = model.predict(imgForModel)
                resultMask = p[0]
                resultMask = np.argmax(resultMask, axis=-1)
                resultMask = np.expand_dims(resultMask, axis=-1)
                resultMask = resultMask * (255 / 3)
                resultMask = resultMask.astype(np.uint8)

                x = cv2.resize(resultMask, (16, 16), interpolation=cv2.INTER_NEAREST)
                predictedMaskImg = np.concatenate([resultMask, resultMask, resultMask], axis=2)
                predictedMaskImg = cv2.cvtColor(predictedMaskImg, cv2.COLOR_BGR2GRAY)
                predictedMaskImg = ((predictedMaskImg / 170) * 255).astype("uint8")
                print(predictedMaskImg.shape)

                ret, GG1 = cv2.threshold(predictedMaskImg.copy(), 200, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                gl = np.multiply(fluo1, (GG1 / 255).astype("uint16"))
                if (len(gl[gl>threshpix]) == 0):
                    aa28 = 0
                else:
                    aa28 = np.sum(gl[gl>threshpix]) / len(gl[gl>threshpix])

                ret, GC1 = cv2.threshold(predictedMaskImg.copy(), 200, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                ret, GC = cv2.threshold(predictedMaskImg.copy(), 10, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                GA1 = cv2.subtract(GC, GC1)
                kl = np.multiply(fluo2, (GA1 / 255).astype("uint16"))
                if (len(kl[kl>threshpix]) == 0):
                    aa45 = 0
                else:
                    aa45 = np.sum(kl[kl>threshpix]) / len(kl[kl>threshpix])

                if not os.path.exists(self.fname4 + "\\predictedMaskCalibPred\\"):
                    os.makedirs(self.fname4 + "\\predictedMaskCalibPred\\")
                cv2.imwrite(self.fname4 + "\\predictedMaskCalibPred\\" + namea, predictedMaskImg)
                writer.writerow([namea, aa28, aa45])

                if not os.path.exists(self.fname4 + "\\NIS28\\"):
                    os.makedirs(self.fname4 + "\\NIS28\\")
                cv2.imwrite(self.fname4 + "\\NIS28\\" + nameb, gl)
                print(gl.dtype)

                if not os.path.exists(self.fname4 + "\\NIS45\\"):
                    os.makedirs(self.fname4 + "\\NIS45\\")
                cv2.imwrite(self.fname4 + "\\NIS45\\" + nameb, kl)
                print(kl.dtype)


    def testManager(self):

        #Error checking user input

        if self.fname3 == "":
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure an Image Directory is provided.")
            dialog.setWindowTitle("Error")
            dialog.exec()
            return

        if self.sname3 == "":
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure a model files (.keras) is provided.")
            dialog.setWindowTitle("Error")
            dialog.exec()
            return

        best_model_file = self.sname3
        model = tf.keras.models.load_model(best_model_file)
        print(model.summary())

        Height = 160
        Width = 160
        NumOfCategories = 3

        allTestImagesNP = [f for f in listdir(self.fname3 + "\\TestBF\\") if isfile(join(self.fname3 + "\\TestBF\\", f))]

        if not os.path.exists(self.fname3 + "\\TestPreds\\"):
            os.makedirs(self.fname3 + "\\TestPreds\\")

        #Initialize Dice score arrays
        D1 = []
        D2 = []

        #Predict and calculate Dice Scores
        for i in range(len(allTestImagesNP)):

            img = cv2.imread(self.fname3 + "\\TestBF\\" + allTestImagesNP[i], cv2.IMREAD_COLOR)
            img = img / 255.0
            img = img.astype(np.float32)
            imgForModel = np.expand_dims(img, axis=0)

            maskTestImagesNP = cv2.imread(self.fname3 + "\\TestGT\\" + allTestImagesNP[i], cv2.IMREAD_GRAYSCALE)
            p = model.predict(imgForModel)
            resultMask = p[0]
            resultMask = np.argmax(resultMask, axis=-1)
            resultMask = np.expand_dims(resultMask, axis=-1)
            resultMask = resultMask * (255 / NumOfCategories)
            resultMask = resultMask.astype(np.uint8)

            #Checkpoint for prediction masks
            x = cv2.resize(resultMask, (16, 16), interpolation=cv2.INTER_NEAREST)
            print(x)
            x = cv2.resize(maskTestImagesNP, (16, 16), interpolation=cv2.INTER_NEAREST)
            print(x)

            #Save predictions
            predictedMaskImg = np.concatenate([resultMask, resultMask, resultMask], axis=2)
            cv2.imwrite(self.fname3 + "\\TestPreds\\" + allTestImagesNP[i], predictedMaskImg)

            #Calculate Dice Scores for 2.8 um beads and 4.5 um beads for each nanowell
            a = maskTestImagesNP.astype("uint8")
            ret, GG = cv2.threshold(a.copy(), 10, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
            ret, GGa = cv2.threshold(a.copy(), 200, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
            GG = cv2.subtract(GG, GGa)
            ret, GG1 = cv2.threshold(a.copy(), 200, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
            b = ((cv2.cvtColor(predictedMaskImg, cv2.COLOR_BGR2GRAY) / 170) * 255).astype("uint8")
            ret, GC = cv2.threshold(b.copy(), 10, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
            ret, GCa = cv2.threshold(b.copy(), 200, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
            GC = cv2.subtract(GC, GCa)
            ret, GC1 = cv2.threshold(b.copy(), 200, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
            gg = np.multiply((GG / 255), (GC / 255))
            total1 = np.sum(gg.astype("uint8"))
            aa = np.multiply((GG1 / 255), (GC1 / 255))
            total2 = np.sum(aa.astype("uint8"))
            bb = np.sum((GG / 255).astype("uint8"))
            cc = np.sum((GC / 255).astype("uint8"))
            atot = bb + cc
            bb1 = np.sum((GG1 / 255).astype("uint8"))
            cc1 = np.sum((GC1 / 255).astype("uint8"))
            btot = bb1 + cc1

            print("Dice1:", 2 * total1 / atot)
            print("Dice2:", 2 * total2 / btot)
            if atot > 0:
                ast = 2 * total1 / atot
            else:
                ast = 1

            if btot > 0:
                bsts = 2 * total2 / btot
            else:
                bsts = 1

            D1.append(ast)
            D2.append(bsts)

        #{lot the dice scores, show the average dice scores, standard deviatinos, and minimums
        plt.boxplot([D2, D1])
        plt.savefig(self.fname3 + "\\DicePlot.jpg")
        self.label_60.setText("Dice 4.5um: " + str(np.round(np.asarray(D1).mean(),2)) + ", SD: " + str(np.round(np.asarray(D1).std(),2)) + ", Min: " + str(np.round(min(D1),2)))
        self.label_59.setText("Dice 2.8um: " + str(np.round(np.asarray(D2).mean(),2)) + ", SD: " + str(np.round(np.asarray(D2).std(),2)) + ", Min: " + str(np.round(min(D2),2)))
        print(np.asarray(D1).mean())
        print(np.asarray(D2).mean())
        print(np.asarray(D1).std())
        print(np.asarray(D2).std())
        print(min(D1))
        print(min(D2))


    #Setup for cropping of nanowells
    def generateSeg(self):
        global minA, maxA, Xmin, Xmax, Ymin, Ymax, oiter, k1, k2, Br, T28, T45, imgNum, im2b16, im3b16, BF_img, FF_img, FF_img2, onlyfiles, executed

        #Error checking the user input

        if self.fname == "":
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure an Image Directory is provided.")
            dialog.setWindowTitle("Error")
            dialog.exec()
            return

        if self.sname == "":
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure a save Directory is provided.")
            dialog.setWindowTitle("Error")
            dialog.exec()
            return

        if (self.lineEdit_2.text() != ''):
            try:
                minA = int(self.lineEdit_2.text())
            except ValueError:
                print("Please enter float!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (self.lineEdit_3.text() != ''):
            try:
                maxA = int(self.lineEdit_3.text())
            except ValueError:
                print("Please enter float!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (self.lineEdit_4.text() != ''):
            try:
                Xmin = int(self.lineEdit_4.text())
            except ValueError:
                print("Please enter float!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (self.lineEdit_5.text() != ''):
            try:
                Xmax = int(self.lineEdit_5.text())
            except ValueError:
                print("Please enter float!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (self.lineEdit_7.text() != ''):
            try:
                Ymin = int(self.lineEdit_7.text())
            except ValueError:
                print("Please enter float!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (self.lineEdit_6.text() != ''):
            try:
                Ymax = int(self.lineEdit_6.text())
            except ValueError:
                print("Please enter float!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (self.lineEdit_6.text() != ''):
            try:
                Ymax = int(self.lineEdit_6.text())
            except ValueError:
                print("Please enter float!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (self.lineEdit_6.text() != ''):
            try:
                Ymax = int(self.lineEdit_6.text())
            except ValueError:
                print("Please enter float!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (self.lineEdit_6.text() != ''):
            try:
                Ymax = int(self.lineEdit_6.text())
            except ValueError:
                print("Please enter float!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (self.lineEdit_101.text() != ''):
            try:
                Br = int(self.lineEdit_101.text())
            except ValueError:
                print("Please enter float!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (self.lineEdit_102.text() != ''):
            try:
                T28 = int(self.lineEdit_102.text())
            except ValueError:
                print("Please enter float!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (self.lineEdit_100.text() != ''):
            try:
                T45 = int(self.lineEdit_100.text())
            except ValueError:
                print("Please enter float!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (self.lineEdit_103.text() != ''):
            try:
                imgNum = int(self.lineEdit_103.text())
            except ValueError:
                print("Please enter float!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (self.lineEdit_300.text() != ''):
            try:
                k1 = int(self.lineEdit_300.text())
            except ValueError:
                print("Please enter integer!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (self.lineEdit_301.text() != ''):
            try:
                oiter = int(self.lineEdit_301.text())
            except ValueError:
                print("Please enter integer!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (self.lineEdit_302.text() != ''):
            try:
                k2 = int(self.lineEdit_302.text())
            except ValueError:
                print("Please enter integer!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return


        if (Br <= 0 or T28 <= 0 or T45 <= 0 or imgNum <= 0):
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure Brightfield, 2.8 um, 4.5 um, and image Numer are greater than 0.")
            dialog.setWindowTitle("Error")
            dialog.exec()
            return

        if (k1 <= 0 or oiter <= 0 or k2 <= 0):
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure C-k1, C-iter, O-k2 are greater than 0.")
            dialog.setWindowTitle("Error")
            dialog.exec()
            return


        if (maxA <= minA or Xmax <= Xmin or Ymax <= Ymin):
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure the maximums are greater than the minimums.")
            dialog.setWindowTitle("Error")
            dialog.exec()
            return

        onlyfiles = [f for f in listdir(self.fname) if isfile(join(self.fname, f))]
        print(onlyfiles)
        if (imgNum > len(onlyfiles)):
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure the Image Number is below or equal to " + str(len(onlyfiles)))
            dialog.setWindowTitle("Error")
            dialog.exec()
            return


        #Initialize image paths and saving paths
        img_path = self.fname + "\\" + onlyfiles[imgNum - 1]
        save = self.sname + "\\"
        save_path = self.sname + "\\BRex\\"  # save segmented nanowells
        save_path2 = self.sname + "\\FL1ex\\"  # save segmented nanowells
        save_path3 = self.sname + "\\FL2ex\\"  # save segmented nanowells
        ret, images = cv2.imreadmulti(img_path, [], cv2.IMREAD_ANYDEPTH)
        print(images[0].dtype)
        if (Br > len(images) or T28 > len(images) or T45 > len(images) or Br <= 0 or T28 <= 0 or T45 <= 0):
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure 0 < Bright field, 2.8um, 4.5um < " + str(len(images) + 1))
            dialog.setWindowTitle("Error")
            dialog.exec()
            return

        #Store the different layers of the multi-layer Tif
        img = images[Br - 1]
        img2 = images[T28 - 1]
        img3 = images[T45 - 1]
        im2b16 = img2.copy()
        im3b16 = img3.copy()

        #Convert to 8 bit images (for segmentation)
        img = (img/256).astype("uint8")
        img2 = (img2/256).astype("uint8")
        img3 = (img3/256).astype("uint8")
        BF_img = img
        FF_img = img2
        FF_img2 = img3

        ### image preprocessing-------------------------------------------------------------------
        ret, threshold = cv2.threshold(BF_img.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Creates a 4x4 rectangular structuring element for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k1, k1))
        # close small holes inside the foreground objects or small black points on the object.
        close = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=oiter)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (k2, k2))
        # remove small objects (it's good for removing noise)
        opening = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel2)

        centroidsBF, imagee = Ui_MainWindow.find_nanowell(BF_img, opening, save, show1 = True)  # BF_img
        #Show the proposed croppping of the image in the GUI
        self.label_2.setPixmap(imagee)#imagee)
        executed = True

    #Crops nanowells of all TIFF image stacks in a folder
    def cropNano(self):
        global offseta, k1, k2, oiter
        if (not(executed)):
            return
        onlyfiles = [f for f in listdir(self.fname) if isfile(join(self.fname, f))]
        print(onlyfiles)

        #Store the locations of the cropped nanowell images and the TIFF image which the nanowell image comes from
        with open(self.sname +"\\NanowellLocations.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "TIFF", "COMX", "COMY"])

            for j in range(len(onlyfiles)):
                self.label_104.setText("Progress: " + str(int(j*100/len(onlyfiles))))
                QtWidgets.qApp.processEvents()
                img_path = self.fname + "\\" + onlyfiles[j]
                save = self.sname + "\\"

                #Generate the saving paths for the images
                save_path = self.sname + "\\NanoBF\\"
                save_path2 = self.sname + "\\NanoGT\\"
                save_path3 = self.sname + "\\NanoQual\\"
                save_path4 = self.sname + "\\Nano28Fl\\"
                save_path5 = self.sname + "\\Nano45Fl\\"

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if not os.path.exists(save_path2):
                    os.makedirs(save_path2)
                if not os.path.exists(save_path3):
                    os.makedirs(save_path3)
                if not os.path.exists(save_path4):
                    os.makedirs(save_path4)
                if not os.path.exists(save_path5):
                    os.makedirs(save_path5)

                ret, images = cv2.imreadmulti(img_path, [], cv2.IMREAD_ANYDEPTH)
                print(images[0].dtype)
                if (Br > len(images) or T28 > len(images) or T45 > len(images) or Br <= 0 or T28 <= 0 or T45 <= 0):
                    dialog = QMessageBox(MainWindow)
                    dialog.setText("Please ensure 0 < Bright field, 2.8um, 4.5um < " + str(len(images) + 1))
                    dialog.setWindowTitle("Error")
                    dialog.exec()
                    return

                #Store the different layers of the multi-layer TIF
                img = images[Br - 1]
                img2 = images[T28 - 1]
                img3 = images[T45 - 1]
                im2b16 = img2.copy()
                im3b16 = img3.copy()

                img = (img/256).astype("uint8")
                img3a = (img2/256).astype("uint8")
                img2a = (img3/256).astype("uint8")
                BF_img = img
                FF_img = img2a
                FF_img2 = img3a

                ### image preprocessing-------------------------------------------------------------------
                ret, threshold = cv2.threshold(BF_img.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # Creates a 4x4 rectangular structuring element for morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k1, k1))
                # close small holes inside the foreground objects or small black points on the object.
                close = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=oiter)
                kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (k2, k2))
                # remove small objects (it's good for removing noise)
                opening = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel2)

                kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                ret, FF_img1 = cv2.threshold(FF_img.copy(), 10, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                FF1 = np.expand_dims(FF_img1.copy(), axis=-1)
                ret, FF_img23 = cv2.threshold(FF_img2.copy(), 10, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                FF2 = np.expand_dims(FF_img23.copy(), axis=-1)
                FF1 = FF1 / 2
                FF1 = FF1.astype("uint8")
                FF2 = FF2.astype("uint8")
                FF = cv2.add(FF1, FF2)

                #Generate the ground truths and the overlaid ground truths for quality inspection
                ret, GG = cv2.threshold(FF.copy(), 10, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                ret, GG1 = cv2.threshold(FF.copy(), 200, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                GGF = cv2.subtract(GG, GG1)
                BF = np.expand_dims(BF_img.copy(), axis=-1)
                GGF = np.expand_dims(GGF.copy(), axis=-1)
                bst = np.expand_dims(np.zeros((FF.shape[0], FF.shape[1])), axis=-1)
                print(bst.shape)
                print(GGF.shape)
                HH_img = np.concatenate([GGF, GGF, bst], axis=2)
                HH_img = HH_img.astype("uint8")
                JJ_img = np.concatenate([BF, BF, BF], axis=2)
                SS = cv2.addWeighted(HH_img, 0.15, JJ_img, 0.85, 0)
                GG1 = np.expand_dims(GG1.copy(), axis=-1)
                DD_img = np.concatenate([GG1, bst, GG1], axis=2)
                DD_img = DD_img.astype("uint8")
                HH = cv2.addWeighted(DD_img, 0.15, SS, 0.85, 0)

                #Crop the nanowell images
                centroidsBF, imagee = Ui_MainWindow.find_nanowell(BF_img, opening, save, show1 = False)  # BF_img

                if (not(self.radioButton_9.isChecked())):

                    #Initialize saving folders and store images for training and testing

                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    if not os.path.exists(save_path2):
                        os.makedirs(save_path2)
                    if not os.path.exists(save_path3):
                        os.makedirs(save_path3)
                    if not os.path.exists(save_path4):
                        os.makedirs(save_path4)
                    if not os.path.exists(save_path5):
                        os.makedirs(save_path5)

                    Ui_MainWindow.crop_squares(BF_img, centroidsBF, 160, save_path, False, offseta)
                    Ui_MainWindow.crop_squares(FF, centroidsBF, 160, save_path2, False, offseta)
                    Ui_MainWindow.crop_squares(HH, centroidsBF, 160, save_path3, False, offseta)
                    Ui_MainWindow.crop_squares(im2b16, centroidsBF, 160, save_path4, True, offseta)
                    Ui_MainWindow.crop_squares(im3b16, centroidsBF, 160, save_path5, True, offseta)
                    # print(offseta)
                    for g in range(len(centroidsBF)):
                        x, y = centroidsBF[g]
                        square_save = str(g + offseta) + '.jpg'
                        writer.writerow([square_save, onlyfiles[j], x, y])

                else:

                    #Initialize saving files, then save and store images for calibrations and predictions

                    splitup = [*onlyfiles[j]]
                    for i in range(4):
                        del splitup[-1]
                    while (splitup[0] != '_'):
                        del splitup[0]
                    del splitup[0]
                    insulin = str(''.join(splitup))

                    save_path6 = self.sname + "\\NanoBFCalibPred\\" + insulin + "_"  # save segmented nanowells
                    save_path7 = self.sname + "\\NanoGTCalibPred\\" + insulin + "_" # save segmented nanowells
                    save_path8 = self.sname + "\\NanoQualCalibPred\\" + insulin + "_" # save segmented nanowells
                    save_path9 = self.sname + "\\Nano28FlCalibPred\\" + insulin + "_" # save segmented nanowells
                    save_path10 = self.sname + "\\Nano45FlCalibPred\\" + insulin + "_" # save segmented nanowells

                    if not os.path.exists(self.sname + "\\NanoBFCalibPred\\"):
                        os.makedirs(self.sname + "\\NanoBFCalibPred\\")
                    if not os.path.exists(self.sname + "\\NanoGTCalibPred\\"):
                        os.makedirs(self.sname + "\\NanoGTCalibPred\\")
                    if not os.path.exists(self.sname + "\\NanoQualCalibPred\\"):
                        os.makedirs(self.sname + "\\NanoQualCalibPred\\")
                    if not os.path.exists(self.sname + "\\Nano28FlCalibPred\\"):
                        os.makedirs(self.sname + "\\Nano28FlCalibPred\\")
                    if not os.path.exists(self.sname + "\\Nano45FlCalibPred\\"):
                        os.makedirs(self.sname + "\\Nano45FlCalibPred\\")

                    Ui_MainWindow.crop_squares(BF_img, centroidsBF, 160, save_path6, False, offseta)
                    Ui_MainWindow.crop_squares(FF, centroidsBF, 160, save_path7, False, offseta)
                    Ui_MainWindow.crop_squares(HH, centroidsBF, 160, save_path8, False, offseta)
                    Ui_MainWindow.crop_squares(im2b16, centroidsBF, 160, save_path9, True, offseta)
                    Ui_MainWindow.crop_squares(im3b16, centroidsBF, 160, save_path10, True, offseta)
                    for g in range(len(centroidsBF)):
                        x, y = centroidsBF[g]
                        square_save = insulin + "_" + str(g + offseta) + '.jpg'
                        writer.writerow([square_save, onlyfiles[j], x, y])

                offseta = offseta + len(centroidsBF)

    #Display overlay images for quality inspection
    def qualInspect(self):
        global startPos, autoMax

        #Error checking user input

        if self.fname1 == "":
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure an Image Directory is provided.")
            dialog.setWindowTitle("Error")
            dialog.exec()
            return

        if self.sname1 == "":
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure a Save Directory is provided.")
            dialog.setWindowTitle("Error")
            dialog.exec()
            return

        if (self.lineEdit_11.text() != ''):
            try:
                startPos = int(self.lineEdit_11.text())
            except ValueError:
                print("Please enter float!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        if (self.lineEdit_105.text() != ''):
            try:
                autoMax = int(self.lineEdit_105.text())
            except ValueError:
                print("Please enter float!")
                dialog = QMessageBox(MainWindow)
                dialog.setText("Please ensure only integers are entered and all fields complete.")
                dialog.setWindowTitle("Error")
                dialog.exec()
                return

        #Initialize saving folder names

        save_path = self.fname1 + "\\NanoBF\\"
        save_path2 = self.fname1 + "\\NanoGT\\"
        save_path3 = self.fname1 + "\\NanoQual\\"
        save_path4 = self.fname1 + "\\Nano28Fl\\"
        save_path5 = self.fname1 + "\\Nano45Fl\\"
        save_path6 = self.sname1 + "\\goodBF\\"
        save_path7 = self.sname1 + "\\goodGT\\"
        save_path8 = self.sname1 + "\\goodQual\\"
        save_path9 = self.sname1 + "\\goodQual28FF\\"
        save_path10 = self.sname1 + "\\goodQual45FF\\"
        save_path11 = self.sname1 + "\\badQualBF\\"
        save_path12 = self.sname1 + "\\badQual28FF\\"
        save_path13 = self.sname1 + "\\badQual45FF\\"

        if not os.path.exists(save_path6):
            os.makedirs(save_path6)
        if not os.path.exists(save_path7):
            os.makedirs(save_path7)
        if not os.path.exists(save_path8):
            os.makedirs(save_path8)
        if not os.path.exists(save_path9):
            os.makedirs(save_path9)
        if not os.path.exists(save_path10):
            os.makedirs(save_path10)
        if not os.path.exists(save_path11):
            os.makedirs(save_path11)
        if not os.path.exists(save_path12):
            os.makedirs(save_path12)
        if not os.path.exists(save_path13):
            os.makedirs(save_path13)

        allfiles = [f for f in listdir(save_path) if isfile(join(save_path, f))]
        print(allfiles)
        if (startPos > len(allfiles) or startPos <=0):
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure the Image Number is below or equal to " + str(len(allfiles)))
            dialog.setWindowTitle("Error")
            dialog.exec()
            return

        #Testing functionality: extract pixel sums for determining the auto screening number
        totalSum = []
        if self.check_105.isChecked():
            for gg in range(len(allfiles)):
                img2 = cv2.imread(save_path2 + allfiles[gg])
                FF_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                adder = FF_img.copy() / 120
                adderSum = np.sum(adder)
                print(adderSum)
                totalSum.append(adderSum)

            with open(self.sname1 + "\\DataQual.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Total"])

                for pp in totalSum:
                    writer.writerow([pp])
            print(len(totalSum))

        else:

            #Manual quality inspection that responds to keyboard presses and places the images in bad and good folders

            rej = 0
            good = 0
            total = len(allfiles) - startPos + 1
            imleft = total
            self.label_12.setText("Total: " + str(total))
            self.label_13.setText("Rejections: " + str(rej))
            self.label_11.setText("Accepted: " + str(good))
            self.label_14.setText("Images Left: " + str(imleft))
            QtWidgets.qApp.processEvents()
            for ll in range(startPos-1, len(allfiles)):
                imleft = imleft - 1
                img = cv2.imread(save_path + allfiles[ll])
                img2 = cv2.imread(save_path2 + allfiles[ll])
                img3 = cv2.imread(save_path3 + allfiles[ll])
                splitup = [*allfiles[ll]]
                for i in range(3):
                    del splitup[-1]
                joint = str(''.join(map(str,splitup)))
                img4 = cv2.imread(save_path4 + joint + "tif", cv2.IMREAD_UNCHANGED)
                img5 = cv2.imread(save_path5 + joint + "tif", cv2.IMREAD_UNCHANGED)

                #Auto screening
                FF_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                adder = FF_img.copy() / 120
                adderSum = np.sum(adder)
                print(ll, adderSum, autoMax)
                totalSum.append(adderSum)
                self.label_14.setText("Images Left: " + str(imleft))
                QtWidgets.qApp.processEvents()

                if (adderSum<autoMax):
                    #Display image in the GUI
                    ab = (cv2.resize(img3, (970, 760))*3).astype("uint8")
                    cv2.imwrite(self.sname1 + "\\pic1.jpg", ab)
                    pixmap = QPixmap(self.sname1 + "\\pic1.jpg")
                    self.label_18.setPixmap(pixmap)
                    QtWidgets.qApp.processEvents()
                    while True:
                        #Reject image
                        if keyboard.read_key() == "w":
                            rej = rej + 1

                            self.label_13.setText("Rejections: " + str(rej))
                            QtWidgets.qApp.processEvents()

                            shutil.move(save_path + allfiles[ll], save_path11 + allfiles[ll])
                            shutil.move(save_path4 + joint + "tif", save_path12 + joint + "tif")
                            shutil.move(save_path5 + joint + "tif", save_path13 + joint + "tif")
                            time.sleep(0.1)
                            break

                        #Accept image
                        if keyboard.read_key() == "e":
                            good = good + 1
                            self.label_11.setText("Accepted: " + str(good))
                            QtWidgets.qApp.processEvents()

                            shutil.move(save_path + allfiles[ll], save_path6 + allfiles[ll])
                            shutil.move(save_path2 + allfiles[ll], save_path7 + allfiles[ll])
                            shutil.move(save_path3 + allfiles[ll], save_path8 + allfiles[ll])
                            shutil.move(save_path4 + joint + "tif", save_path9 + joint + "tif")
                            shutil.move(save_path5 + joint + "tif", save_path10 + joint + "tif")
                            time.sleep(0.1)
                            break
                else:
                    #auto rejection
                    rej = rej + 1

                    self.label_13.setText("Rejections: " + str(rej))
                    QtWidgets.qApp.processEvents()

                    shutil.move(save_path + allfiles[ll], save_path11 + allfiles[ll])
                    shutil.move(save_path4 + joint + "tif", save_path12 + joint + "tif")
                    shutil.move(save_path5 + joint + "tif", save_path13 + joint + "tif")



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
