import csv
import os
import shutil
import time
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
#
minA = 3500
maxA = 30000
Xmin = 100
Xmax = 2000
Ymin = 100
Ymax = 2000
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

class Ui_MainWindow(QMainWindow):
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
        self.label_103.setGeometry(QtCore.QRect(40, 600, 100, 16))
        self.label_103.setObjectName("label_103")
        self.lineEdit_103 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_103.setGeometry(QtCore.QRect(30, 620, 150, 22))
        self.lineEdit_103.setObjectName("lineEdit_103")
        self.label_104 = QtWidgets.QLabel(self.tab)
        self.label_104.setGeometry(QtCore.QRect(40, 750, 100, 16))
        self.label_104.setObjectName("label_104")



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
        self.radioButton_9.setGeometry(QtCore.QRect(90, 270, 95, 20))
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
        self.line_7.setGeometry(QtCore.QRect(30, 360, 241, 16))
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.label_63 = QtWidgets.QLabel(self.tab_5)
        self.label_63.setGeometry(QtCore.QRect(30, 350, 141, 16))
        self.label_63.setObjectName("label_63")
        self.lineEdit_30 = QtWidgets.QLineEdit(self.tab_5)
        self.lineEdit_30.setGeometry(QtCore.QRect(30, 450, 141, 22))
        self.lineEdit_30.setObjectName("lineEdit_30")
        self.pushButton_24 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_24.setGeometry(QtCore.QRect(180, 450, 93, 28))
        self.pushButton_24.setObjectName("pushButton_24")
        self.label_64 = QtWidgets.QLabel(self.tab_5)
        self.label_64.setGeometry(QtCore.QRect(30, 430, 201, 16))
        self.label_64.setObjectName("label_64")
        self.label_65 = QtWidgets.QLabel(self.tab_5)
        self.label_65.setGeometry(QtCore.QRect(30, 380, 201, 16))
        self.label_65.setObjectName("label_65")
        self.pushButton_25 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_25.setGeometry(QtCore.QRect(180, 400, 93, 28))
        self.pushButton_25.setObjectName("pushButton_25")
        self.lineEdit_31 = QtWidgets.QLineEdit(self.tab_5)
        self.lineEdit_31.setGeometry(QtCore.QRect(30, 400, 141, 22))
        self.lineEdit_31.setObjectName("lineEdit_31")
        self.pushButton_26 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_26.setGeometry(QtCore.QRect(30, 500, 251, 28))
        self.pushButton_26.setObjectName("pushButton_26")
        self.pushButton_27 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_27.setGeometry(QtCore.QRect(180, 640, 93, 28))
        self.pushButton_27.setObjectName("pushButton_27")
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
        print("a")
        # self.setFocusPolicy(Qt.StrongFocus)
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


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
        self.radioButton_9.setText(_translate("MainWindow", "Calibration"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Nanowell Slicer"))
        self.label_11.setText(_translate("MainWindow", "Accepted:"))
        self.label_12.setText(_translate("MainWindow", "Total:"))
        self.label_13.setText(_translate("MainWindow", "Rejections: "))
        self.label_14.setText(_translate("MainWindow", "Images Left:"))
        self.pushButton_5.setText(_translate("MainWindow", "Search"))
        self.label_15.setText(_translate("MainWindow", "Image Directory (TIFF MultiLayer)"))
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
        self.label_63.setText(_translate("MainWindow", "Fluorescent Calibration"))
        self.pushButton_24.setText(_translate("MainWindow", "Search"))
        self.label_64.setText(_translate("MainWindow", "Model File"))
        self.label_65.setText(_translate("MainWindow", "Data Directory"))
        self.pushButton_25.setText(_translate("MainWindow", "Search"))
        self.pushButton_26.setText(_translate("MainWindow", "Predict"))
        self.pushButton_27.setText(_translate("MainWindow", "Search"))
        self.label_66.setText(_translate("MainWindow", "Fluorescent Predictions"))
        self.label_67.setText(_translate("MainWindow", "Data Directory"))
        self.label_101.setText(_translate("MainWindow", "Brightfield"))
        self.label_102.setText(_translate("MainWindow", "2.8 um"))
        self.label_100.setText(_translate("MainWindow", "4.5 um"))
        self.label_103.setText(_translate("MainWindow", "Image Number"))
        self.label_104.setText(_translate("MainWindow", "Progress: "))
        self.label_105.setText(_translate("MainWindow", "Auto"))
        self.lineEdit_105.setText(_translate("MainWindow", "9000"))
        self.lineEdit_105.setPlaceholderText(_translate("MainWindow", "9000"))
        self.check_105.setText(_translate("MainWindow", "Test"))
        self.label_106.setText(_translate("MainWindow", "Offset"))
        self.pushButton_28.setText(_translate("MainWindow", "Plot"))
        self.pushButton_106.setText(_translate("MainWindow", "Enter"))
        self.label_107.setText(_translate("MainWindow", "Name:"))
        self.label_108.setText(_translate("MainWindow", "Progress: "))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "Testing and Prediction"))
    #     self.Runner.ImageUpdate.connect(self.ImageUpdateSlot)
    #
    # def ImageUpdateSlot(self):
    #     self.label_18.setPixmap(QPixmap.fromImage(Image))

    def keyPressEvent(self, event):
        if isinstance(event, QKeyEvent):
            key_text = event.text()
            print(str(key_text))

    def keyReleaseEvent(self, event):
        if isinstance(event, QKeyEvent):
            key_text = event.text()
            print(str(key_text))

    def event(self, event):
        if (event.type() == QEvent.KeyPress) and (event.key() == Qt.Key_Space):
            print('Parent handling space')
            return True
        return QWidget.event(self, event)

    def eventFilter(self, widget, event):
        if (event.type() == QEvent.KeyPress) and (event.key() == Qt.Key_Space):
            print('Sending space event to parent...')
            self.event(event)
            return True
        return super(Ui_MainWindow, self).eventFilter(widget, event)


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

        # fig, ax = plt.subplots(figsize=(10, 10))
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
                    # ax.add_patch(square)

        # ax.imshow(BF_img, cmap='gray')
        # print('total=', image_number)
        # plt.show()
        if show1:

            imgcopy = (cv2.resize(imgcopy, (850, 750), interpolation = cv2.INTER_LINEAR)*4).astype("uint8")

            cv2.imwrite(save+"pic.jpg", imgcopy)
            pixmap = QPixmap(save+"pic.jpg")
            return centroidsBF, pixmap
        else:
            return centroidsBF,None

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

    def trainSize(self):
        global trainS, numberF, moveOn

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

        self.label_27.setText("Testing Size:" + str(numberF-trainS))
        moveOn = True


    def trainer(self):
        global trainS, numberF, moveOn

        if (not(moveOn)):
            return

        modName = self.sname2 + "\\" + self.modelName + ".keras"

        totalfiles = [f for f in listdir(self.fname2 + "\\goodBF\\") if isfile(join(self.fname2 + "\\goodBF\\", f))]
        afiles = []
        testfiles = []
        for p in range(len(totalfiles)):
            if (p < trainS):
                afiles.append(totalfiles[p])
            else:
                testfiles.append(totalfiles[p])

        basedir1 = self.fname2 + "\\goodBF\\"
        basedir2 = self.fname2 + "\\goodGT\\"

        allImagesBF = []
        allImagesGT = []

        allTestImagesBF = []
        allTestImagesGT = []
        print("b")

        for ss in range(len(afiles)):
            img = cv2.imread(basedir1 + afiles[ss], cv2.IMREAD_COLOR)
            img2 = cv2.imread(basedir2 + afiles[ss], cv2.IMREAD_GRAYSCALE)
            allImagesBF.append(img)
            allImagesGT.append(img2)

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

        print("c")
        save_patha = self.sname2 + "\\TestBF\\"
        save_pathb = self.sname2 + "\\TestGT\\"
        if not os.path.exists(save_patha):
            os.makedirs(save_patha)
        if not os.path.exists(save_pathb):
            os.makedirs(save_pathb)

        for kk in range(len(testfiles)):
            img = cv2.imread(basedir1 + testfiles[kk], cv2.IMREAD_COLOR)
            img2 = cv2.imread(basedir2 + testfiles[kk], cv2.IMREAD_GRAYSCALE)
            cv2.imwrite( save_patha + testfiles[kk], img)
            cv2.imwrite( save_pathb + testfiles[kk], img2)
            allTestImagesBF.append(img)
            allTestImagesGT.append(img2)

        model = Model.build_unet((160, 160, 3), 2)
        print(model.summary())

        Height = 160
        Width = 160
        NumofCategories = 3

        allImages = []
        maskImages = []

        allTestImages = []
        maskTestImages = []


        for img1 in allImagesBF:
            img = (img1 / 255.0)
            img = img.astype(np.float32)
            allImages.append(img)

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

        allImagesNP = np.array(allImages)
        maskImagesNP = np.array(maskImages)
        maskImagesNP = maskImagesNP.astype(int)

        print(allImagesNP.shape)
        print(allImagesNP.dtype)

        print(maskImagesNP.shape)
        print(maskImagesNP.dtype)
        print(maskImagesNP[0].dtype)

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

        for img1 in allTestImagesBF:
            img = img1 / 255.0
            img = img.astype(np.float32)
            allTestImages.append(img)

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

        allTestImagesNP = np.array(allTestImages)
        maskTestImagesNP = np.array(maskTestImages)
        maskTestImagesNP = maskTestImagesNP.astype(int)

        print(allTestImagesNP.shape)
        print(allTestImagesNP.dtype)

        print(maskTestImagesNP.shape)
        print(maskTestImagesNP.dtype)

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



        np.save(self.sname2 + "\\TestBF.npy" , allTestImagesNP)
        np.save(self.sname2 + "\\TestGT.npy", maskTestImagesNP)

        Weight = 160
        Width = 160
        numofCategories = 3

        # from keras.utils import np_utils
        # from keras.utils.np_utils import to_categorical


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


        X_train, X_val, y_train, y_val = train_test_split(allImagesNP, maskForTheModel, test_size=0.1, random_state=42)

        print(X_train.shape)
        print(y_train.shape)
        print("hi", X_train.dtype)
        print(y_train.dtype)

        print(X_val.shape)
        print(y_val.shape)

        shape = (160, 160, 3)
        num_classes = 3
        lr = 1e-4
        batch_size = 4
        epochs = 10

        model = Model.build_unet(shape, num_classes)
        print(model.summary())
        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr), metrics=['accuracy'])

        stepsPerEpoch = np.ceil(len(X_train) / batch_size)
        validationSteps = np.ceil(len(X_val) / batch_size)

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

        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                            validation_data=(X_val, y_val), shuffle=True, callbacks=callbacks)


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


    def calibrater(self):
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

        best_model_file = self.sname4
        model = tf.keras.models.load_model(best_model_file)
        print(model.summary())

        allTestImagesNP = [f for f in listdir(self.fname4 + "\\NanoBFCalib\\") if isfile(join(self.fname4 + "\\NanoBFCalib\\", f))]

        with open(self.fname4 + '\\FluoroResults.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Ins", "2.8Int", "4.5Int"])

            for i in range(len(allTestImagesNP)):
                self.label_108.setText("Progress: " + str(int(i*100/len(allTestImagesNP))))
                QtWidgets.qApp.processEvents()
                namea = allTestImagesNP[i]
                Image12 = cv2.imread(self.fname4 + "\\NanoBFCalib\\" + allTestImagesNP[i])  # , cv2.IMREAD_COLOR)#)
                # Image = cv2.cvtColor(Image12, cv2.COLOR_GRAY2BGR)
                splitup = [*allTestImagesNP[i]]
                for j in range(3):
                    del splitup[-1]
                word1 = str(''.join(splitup))
                nameb = word1 + "tif"
                fluo1 = cv2.imread(self.fname4 + "\\Nano28FlCalib\\" + nameb, cv2.IMREAD_ANYDEPTH)
                fluo2 = cv2.imread(self.fname4 + "\\Nano45FlCalib\\" + nameb, cv2.IMREAD_ANYDEPTH)
                # fluo1 = cv2.cvtColor(fluo1.copy(), cv2.COLOR_BGR2GRAY)
                # fluo2 = cv2.cvtColor(fluo2.copy(), cv2.COLOR_BGR2GRAY)

                while splitup[-1] != "_":
                    del splitup[-1]

                del splitup[-1]
                conc = str(''.join(splitup))

                Image = Image12 / 255.0
                Image = Image.astype(np.float32)

                img = Image.copy()
                imgForModel = np.expand_dims(img, axis=0)

                p = model.predict(imgForModel)
                # print(p)

                resultMask = p[0]
                # print(resultMask.shape)

                resultMask = np.argmax(resultMask, axis=-1)
                # print(resultMask.shape)

                resultMask = np.expand_dims(resultMask, axis=-1)
                # print(resultMask.shape)

                resultMask = resultMask * (255 / 3)
                resultMask = resultMask.astype(np.uint8)

                x = cv2.resize(resultMask, (16, 16), interpolation=cv2.INTER_NEAREST)
                # print(x)

                predictedMaskImg = np.concatenate([resultMask, resultMask, resultMask], axis=2)
                predictedMaskImg = cv2.cvtColor(predictedMaskImg, cv2.COLOR_BGR2GRAY)
                predictedMaskImg = ((predictedMaskImg / 170) * 255).astype("uint8")
                print(predictedMaskImg.shape)

                ret, GG1 = cv2.threshold(predictedMaskImg.copy(), 200, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                gl = np.multiply(fluo1, (GG1 / 255).astype("uint16"))
                aa28 = np.sum(gl) / np.count_nonzero(gl)

                ret, GC1 = cv2.threshold(predictedMaskImg.copy(), 200, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                ret, GC = cv2.threshold(predictedMaskImg.copy(), 10, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                GA1 = cv2.subtract(GC, GC1)
                kl = np.multiply(fluo2, (GA1 / 255).astype("uint16"))
                aa45 = np.sum(kl) / np.count_nonzero(kl)


                if not os.path.exists(self.fname4 + "\\predictedMaskCalib\\"):
                    os.makedirs(self.fname4 + "\\predictedMaskCalib\\")
                cv2.imwrite(self.fname4 + "\\predictedMaskCalib\\" + namea, predictedMaskImg)
                writer.writerow([namea, conc, aa28, aa45])


    def testManager(self):
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


        # from keras.utils import to_categorical
        # maskImagesForModel = to_categorical(allTestImagesNP, num_classes=NumOfCategories)
        # maskImagesForModel = maskImagesForModel.astype(int)
        # brigths = []
        # groundTs = []
        # predictions = []
        # cv2_imshow(allTestImagesNP[1] * 255)
        # cv2_imshow(maskTestImagesNP[1] * 128)

        D1 = []
        D2 = []

        for i in range(len(allTestImagesNP)):


            img = cv2.imread(self.fname3 + "\\TestBF\\" + allTestImagesNP[i], cv2.IMREAD_COLOR)
            img = img / 255.0
            img = img.astype(np.float32)
            imgForModel = np.expand_dims(img, axis=0)

            maskTestImagesNP = cv2.imread(self.fname3 + "\\TestGT\\" + allTestImagesNP[i], cv2.IMREAD_GRAYSCALE)

            # maskTestImagesNP = maskTestImagesNP - 1

            p = model.predict(imgForModel)
            # print(p)

            resultMask = p[0]
            # print(resultMask.shape)

            resultMask = np.argmax(resultMask, axis=-1)
            # print(resultMask.shape)

            resultMask = np.expand_dims(resultMask, axis=-1)
            # print(resultMask.shape)

            resultMask = resultMask * (255 / NumOfCategories)
            resultMask = resultMask.astype(np.uint8)

            x = cv2.resize(resultMask, (16, 16), interpolation=cv2.INTER_NEAREST)
            print(x)

            x = cv2.resize(maskTestImagesNP, (16, 16), interpolation=cv2.INTER_NEAREST)
            print(x)

            predictedMaskImg = np.concatenate([resultMask, resultMask, resultMask], axis=2)

            cv2.imwrite(self.fname3 + "\\TestPreds\\" + allTestImagesNP[i], predictedMaskImg)

            # brigths.append((img * 900).astype(np.uint8))
            # groundTs.append((maskTestImagesNP[i] * 128).astype(np.uint8))
            # predictions.append(predictedMaskImg)

            # maskTestImagesNP = (maskTestImagesNP * 128).astype(np.uint8)



            a = maskTestImagesNP.astype("uint8")
            ret, GG = cv2.threshold(a.copy(), 10, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
            ret, GGa = cv2.threshold(a.copy(), 200, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
            GG = cv2.subtract(GG, GGa)
            # cv2_imshow(GG)
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
            # if (ast < 0.6):
            #     cv2_imshow(GG)
            #     cv2_imshow(GC)
            #     cv2_imshow(Brights[i])
            #     ast = 1

            if btot > 0:
                bsts = 2 * total2 / btot
            else:
                bsts = 1

            D1.append(ast)
            D2.append(bsts)


        plt.boxplot([D2, D1])
        plt.savefig(self.fname3 + "\\DicePlot.jpg")
        # plt.boxplot(D2)
        # plt.show()
        self.label_60.setText("Dice 4.5um: " + str(np.round(np.asarray(D1).mean(),2)) + ", SD: " + str(np.round(np.asarray(D1).std(),2)) + ", Min: " + str(np.round(min(D1),2)))
        self.label_59.setText("Dice 2.8um: " + str(np.round(np.asarray(D2).mean(),2)) + ", SD: " + str(np.round(np.asarray(D2).std(),2)) + ", Min: " + str(np.round(min(D2),2)))
        print(np.asarray(D1).mean())
        print(np.asarray(D2).mean())
        print(np.asarray(D1).std())
        print(np.asarray(D2).std())
        print(min(D1))
        print(min(D2))



    def generateSeg(self):
        global minA, maxA, Xmin, Xmax, Ymin, Ymax, Br, T28, T45, imgNum, im2b16, im3b16, BF_img, FF_img, FF_img2, onlyfiles, executed

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

        if (Br <= 0 or T28 <= 0 or T45 <= 0 or imgNum <= 0):
            dialog = QMessageBox(MainWindow)
            dialog.setText("Please ensure Brightfield, 2.8 um, 4.5 um, and image Numer are greater than 0.")
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


        img = images[Br - 1]
        img2 = images[T28 - 1]
        img3 = images[T45 - 1]
        im2b16 = img2.copy()
        im3b16 = img3.copy()

        img = (img/256).astype("uint8")
        img2 = (img2/256).astype("uint8")
        img3 = (img3/256).astype("uint8")
        BF_img = img#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        FF_img = img2#cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        FF_img2 = img3#cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

        #        ret, FF_img = cv2.threshold(FF_img.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        ### image preprocessing-------------------------------------------------------------------
        ret, threshold = cv2.threshold(BF_img.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Creates a 4x4 rectangular structuring element for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        # close small holes inside the foreground objects or small black points on the object.
        close = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=5)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
        # remove small objects (it's good for removing noise)
        opening = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel2)

        centroidsBF, imagee = Ui_MainWindow.find_nanowell(BF_img, opening, save, show1 = True)  # BF_img
        self.label_2.setPixmap(imagee)
        executed = True

    def cropNano(self):
        global offseta
        if (not(executed)):
            return
        onlyfiles = [f for f in listdir(self.fname) if isfile(join(self.fname, f))]
        print(onlyfiles)
        with open(self.sname +"\\NanowellLocations.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "TIFF", "COMX", "COMY"])

            for j in range(len(onlyfiles)):
                self.label_104.setText("Progress: " + str(int(j*100/len(onlyfiles))))
                QtWidgets.qApp.processEvents()
                img_path = self.fname + "\\" + onlyfiles[j]
                save = self.sname + "\\"
                save_path = self.sname + "\\NanoBF\\"  # save segmented nanowells
                save_path2 = self.sname + "\\NanoGT\\"  # save segmented nanowells
                save_path3 = self.sname + "\\NanoQual\\"  # save segmented nanowells
                save_path4 = self.sname + "\\Nano28Fl\\"  # save segmented nanowells
                save_path5 = self.sname + "\\Nano45Fl\\"  # save segmented nanowells

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


                img = images[Br - 1]
                img2 = images[T28 - 1]
                img3 = images[T45 - 1]
                im2b16 = img2.copy()
                im3b16 = img3.copy()

                img = (img/256).astype("uint8")
                img3a = (img2/256).astype("uint8")
                img2a = (img3/256).astype("uint8")
                BF_img = img#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                FF_img = img2a#cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                FF_img2 = img3a#cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

                #        ret, FF_img = cv2.threshold(FF_img.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                ### image preprocessing-------------------------------------------------------------------
                ret, threshold = cv2.threshold(BF_img.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # Creates a 4x4 rectangular structuring element for morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
                # close small holes inside the foreground objects or small black points on the object.
                close = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=5)
                kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
                # remove small objects (it's good for removing noise)
                opening = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel2)



                kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                # FF_img = cv2.morphologyEx(FF_img.copy(), cv2.MORPH_OPEN, kernel3)
                # FF_img = cv2.GaussianBlur(FF_img.copy(),(11,11),0)
                # HH_img = np.multiply(FF_img/255, BF_img)
                # HH_img = HH_img.astype("uint8")
                ret, FF_img1 = cv2.threshold(FF_img.copy(), 10, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                FF1 = np.expand_dims(FF_img1.copy(), axis=-1)
                ret, FF_img23 = cv2.threshold(FF_img2.copy(), 10, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                FF2 = np.expand_dims(FF_img23.copy(), axis=-1)
                FF1 = FF1 / 2
                FF1 = FF1.astype("uint8")
                FF2 = FF2.astype("uint8")
                FF = cv2.add(FF1, FF2)

                ret, GG = cv2.threshold(FF.copy(), 10, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                ret, GG1 = cv2.threshold(FF.copy(), 200, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
                GGF = cv2.subtract(GG, GG1)
                BF = np.expand_dims(BF_img.copy(), axis=-1)
                GGF = np.expand_dims(GGF.copy(), axis=-1)
                # cc = np.expand_dims(cv2.add(BF_img, FF), axis = -1)
                # cc = cc.astype("uint8")
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

                # print(bst.shape, "a")
                # print(HH_img.shape, "b")


                centroidsBF, imagee = Ui_MainWindow.find_nanowell(BF_img, opening, save, show1 = False)  # BF_img
                if (not(self.radioButton_9.isChecked())):

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

                    splitup = [*onlyfiles[j]]
                    for i in range(4):
                        del splitup[-1]
                    while (splitup[0] != '_'):
                        del splitup[0]
                    del splitup[0]
                    insulin = str(''.join(splitup))

                    save_path6 = self.sname + "\\NanoBFCalib\\" + insulin + "_"  # save segmented nanowells
                    save_path7 = self.sname + "\\NanoGTCalib\\" + insulin + "_" # save segmented nanowells
                    save_path8 = self.sname + "\\NanoQualCalib\\" + insulin + "_" # save segmented nanowells
                    save_path9 = self.sname + "\\Nano28FlCalib\\" + insulin + "_" # save segmented nanowells
                    save_path10 = self.sname + "\\Nano45FlCalib\\" + insulin + "_" # save segmented nanowells

                    if not os.path.exists(self.sname + "\\NanoBFCalib\\"):
                        os.makedirs(self.sname + "\\NanoBFCalib\\")
                    if not os.path.exists(self.sname + "\\NanoGTCalib\\"):
                        os.makedirs(self.sname + "\\NanoGTCalib\\")
                    if not os.path.exists(self.sname + "\\NanoQualCalib\\"):
                        os.makedirs(self.sname + "\\NanoQualCalib\\")
                    if not os.path.exists(self.sname + "\\Nano28FlCalib\\"):
                        os.makedirs(self.sname + "\\Nano28FlCalib\\")
                    if not os.path.exists(self.sname + "\\Nano45FlCalib\\"):
                        os.makedirs(self.sname + "\\Nano45FlCalib\\")

                    Ui_MainWindow.crop_squares(BF_img, centroidsBF, 160, save_path6, False, offseta)
                    Ui_MainWindow.crop_squares(FF, centroidsBF, 160, save_path7, False, offseta)
                    Ui_MainWindow.crop_squares(HH, centroidsBF, 160, save_path8, False, offseta)
                    Ui_MainWindow.crop_squares(im2b16, centroidsBF, 160, save_path9, True, offseta)
                    Ui_MainWindow.crop_squares(im3b16, centroidsBF, 160, save_path10, True, offseta)
                    # print(offseta)
                    for g in range(len(centroidsBF)):
                        x, y = centroidsBF[g]
                        square_save = insulin + "_" + str(g + offseta) + '.jpg'
                        writer.writerow([square_save, onlyfiles[j], x, y])


                offseta = offseta + len(centroidsBF)

    def qualInspect(self):
        global startPos, autoMax
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
                # print(img4.dtype)
                # print(img5.dtype)

                FF_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                adder = FF_img.copy() / 120
                adderSum = np.sum(adder)
                print(ll, adderSum, autoMax)
                totalSum.append(adderSum)
                self.label_14.setText("Images Left: " + str(imleft))
                QtWidgets.qApp.processEvents()
                if (adderSum<autoMax):
                    ab = (cv2.resize(img3, (970, 760))*3).astype("uint8")
                    cv2.imwrite(self.sname1 + "\\pic1.jpg", ab)
                    pixmap = QPixmap(self.sname1 + "\\pic1.jpg")
                    self.label_18.setPixmap(pixmap)
                    QtWidgets.qApp.processEvents()
                    while True:
                        if keyboard.read_key() == "w":
                            rej = rej + 1

                            self.label_13.setText("Rejections: " + str(rej))
                            QtWidgets.qApp.processEvents()

                            shutil.move(save_path + allfiles[ll], save_path11 + allfiles[ll])
                            shutil.move(save_path4 + joint + "tif", save_path12 + joint + "tif")
                            shutil.move(save_path5 + joint + "tif", save_path13 + joint + "tif")
                            time.sleep(0.1)
                            break

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
