# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:49:48 2024

@author: 贺国永
"""
import torch
import os
import PIL
from PIL import Image
from torchvision import models, transforms
from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog, QMainWindow, \
    QWidget, QComboBox, QTabWidget, QLabel, QFrame, QTextEdit, QLineEdit
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile, QSize, Qt, Signal, Slot, QObject, QThread
from PySide2.QtGui import QImage, QPixmap
#from PyQt5.QtWidgets import QMessageBox, QInputDialog
from Titan_predict import C_Titan
import Titan_predict
import openslide
import sys
import io
#import threading
import queue


class Worker(QObject):
    # 定义一个信号，用于传递字符串数据到主线程
    update_text = Signal(str)

    def __init__(self, LaiDunMedical):
        super(Worker, self).__init__()
        '''
        self.m_lineEdit_Hint = m_lineEdit_Hint
        self.图片文件 = fpath
        self.titan = titan
        '''
        self.LaiDunMedical = LaiDunMedical

    @Slot()
    def do_work(self):

        print('显示svs图片')
        self.LaiDunMedical.m_lineEdit_Hint.setText('显示svs图片')
        self.LaiDunMedical.m_lineEdit_Hint.update()
        self.LaiDunMedical.f_显示svs图片()
        print('正在加载Titan模型...')
        self.LaiDunMedical.m_lineEdit_Hint.setText('正在加载Titan模型...')
        self.LaiDunMedical.m_lineEdit_Hint.update()
        if self.LaiDunMedical.titan is None:
            self.LaiDunMedical.titan = C_Titan()
        l_路径 = os.path.dirname(self.LaiDunMedical.图片文件)
        l_文件全名 = os.path.basename(self.LaiDunMedical.图片文件)
        l_主文件名 = os.path.splitext(l_文件全名)[0]

        print('正在生成图像切片文件到'+l_路径)
        self.LaiDunMedical.m_lineEdit_Hint.setText('正在生成图像切片文件...')
        self.LaiDunMedical.m_lineEdit_Hint.update()
        self.LaiDunMedical.titan.f_生成图像切片(l_路径)
        print('正在提取切片特征到'+l_路径)
        self.LaiDunMedical.m_lineEdit_Hint.setText('正在提取切片特征...')
        self.LaiDunMedical.m_lineEdit_Hint.update()
        self.LaiDunMedical.titan.f_提取切片特征(l_路径)

        print('正在生成图像特征...')
        self.LaiDunMedical.m_lineEdit_Hint.setText('正在生成图像特征...')
        self.LaiDunMedical.m_lineEdit_Hint.update()
        l_slide_embedding = self.LaiDunMedical.titan.f_生成图像特征(l_路径 + '/feat/h5_files/' + l_主文件名 + '.h5')
        print('已生成图像特征')
        self.LaiDunMedical.m_lineEdit_Hint.setText('已生成图像特征')
        self.LaiDunMedical.m_lineEdit_Hint.update()
        l_分类结果 = self.LaiDunMedical.titan.m_classificationModel.predict(l_slide_embedding)
        l_分类概率 = self.LaiDunMedical.titan.m_classificationModel.predict_proba(l_slide_embedding)[:, :]
        # 通过enumerate获取索引和值，并构建字典
        l_dict_分类概率 = {i: val for i, val in enumerate(l_分类概率[0])}

        # 根据字典的值进行从大到小的排序
        sorted_items = dict(sorted(l_dict_分类概率.items(), key=lambda item: item[1], reverse=True))
        # 将排序后的项转换回字典类型
        sorted_dict = dict(sorted_items)
        print('排序后分类概率:')
        print(sorted_dict)
        l_result = ''

        for key, val in sorted_dict.items():
            #l_result += C_Titan.疾病分类[int(key)] + ': ' + f"{val:.7f}" + '\n'
            self.update_text.emit(C_Titan.疾病分类[int(key)] + ': ' + f"{val:.7f}" + '\n')


class LaiDunMedical:
    
    def __init__(self):

        qtUiMWindow = QFile('./Medical.ui')
        qtUiMWindow.open(QFile.ReadOnly)
        #qtUiMWindow.close()
        
        #加载UI文件
        self.m_uiMainWindow = QUiLoader().load(qtUiMWindow)
        self.m_uiMainWindow.btn_readImg.clicked.connect(self.f_Titan模型预测)
        
        #所有用到的控件
        #图片显示控件
        self.m_labelImg = self.m_uiMainWindow.findChild(QLabel, 'label_img')
        self.m_frameImg = self.m_uiMainWindow.findChild(QFrame, 'frame_img')
        self.m_textEdit_surfaceP = self.m_uiMainWindow.findChild(QTextEdit, 'textEdit_surfaceP')
        self.m_lineEdit_W = self.m_uiMainWindow.findChild(QLineEdit, 'lineEdit_W')
        self.m_lineEdit_H = self.m_uiMainWindow.findChild(QLineEdit, 'lineEdit_H')
        self.m_lineEdit_Hint = self.m_uiMainWindow.findChild(QLineEdit, 'lineEdit_Hint')
        self.图片文件 = ''
        self.titan = None

        self.Titan_thread = QThread()
        self.Titan_worker = None
        #self.m_titan_result_signal = Signal(str)
        #self.m_titan_result_signal.connect(self.update_text_edit)

        
        #if self.titan is None:
        #    self.titan = C_Titan()

    @Slot(str)
    def on_update_text(self, text):
        self.m_textEdit_surfaceP.append(text)


    #获取指定控件的所有父控件，直到顶级父控件为止
    def f_get_parent_list(self, widget):
        parent_list = []
        while widget:
            parent_list.append(widget)
            widget = widget.parentWidget()
        return parent_list    
        
        
    #获取文件路径
    def f_getFilePath(self):
        return QFileDialog.getOpenFileName(self.m_uiMainWindow, "选取单个文件", \
                                           "./", "All Files (*);;Text Files (*.txt)")
    
    def f_读取SVS文件(self):
        self.图片文件, ok = self.f_getFilePath()
        if not ok:
            QMessageBox.about(self.m_uiMainWindow, '提示','未指定文件路径')
            return False
        self.slide = openslide.OpenSlide(self.图片文件)

        return True

    def f_计算图片显示宽高(self, imgW, imgH):
        lableScale = self.m_labelImg.width()/self.m_labelImg.height()
        imgScale = imgW / imgH
        if lableScale > imgScale:
            h = self.m_labelImg.height()
            w = int(h * imgScale)
        else:
            w = self.m_labelImg.width()
            h = int(w // imgScale)
        return int(w), int(h)

    #显示图片
    def f_显示svs图片(self):
        downsample_factor = self.slide.level_downsamples
        max_value = float('-inf')
        max_index = -1
        # 遍历元组，寻找最大值及其下标
        for index, value in enumerate(downsample_factor):
            if value > max_value:
                max_value = value
                max_index = index

        location = (0, 0)
        size = self.slide.level_dimensions[index]
        image = self.slide.read_region(location, index, size).convert("RGB")
        # 创建一个字节流
        byte_arr = io.BytesIO()
        # 将 PIL 图像保存到字节流中
        image.save(byte_arr, format='PNG')
        # 将字节流的位置重置为开始
        byte_arr.seek(0)
        # 将PIL图像转换为QImage
        #qimage = QImage(image.tobytes(), size[0], size[1], QImage.Format_RGB888)
        qimage = QImage()
        qimage.loadFromData(byte_arr.read())

        w, h = self.f_计算图片显示宽高(size[0], size[1])
        # 缩放QImage到指定大小
        qimage_scaled = qimage.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.m_labelImg.setPixmap(QPixmap.fromImage(qimage_scaled))
        self.m_labelImg.update()
        return

    def f_Titan开始(self):
        print('显示svs图片')
        self.m_lineEdit_Hint.setText('显示svs图片')
        self.m_lineEdit_Hint.update()
        self.f_显示svs图片()
        print('正在加载Titan模型...')
        self.m_lineEdit_Hint.setText('正在加载Titan模型...')
        self.m_lineEdit_Hint.update()
        if self.titan is None:
            self.titan = C_Titan()
        l_路径 = os.path.dirname(self.图片文件)
        l_文件全名 = os.path.basename(self.图片文件)        
        l_主文件名 = os.path.splitext(l_文件全名)[0]

        print('正在生成图像切片文件...')
        self.m_lineEdit_Hint.setText('正在生成图像切片文件...')
        self.m_lineEdit_Hint.update()
        #self.titan.f_生成图像切片(l_路径)
        print('正在提取切片特征...')
        self.m_lineEdit_Hint.setText('正在提取切片特征...')
        self.m_lineEdit_Hint.update()
        #self.titan.f_提取切片特征(l_路径)

        print('正在生成图像特征...')
        self.m_lineEdit_Hint.setText('正在生成图像特征...')
        self.m_lineEdit_Hint.update()
        l_slide_embedding = self.titan.f_生成图像特征(l_路径 + '/feat/h5_files/' + l_主文件名 + '.h5')
        l_分类结果 = self.titan.m_classificationModel.predict(l_slide_embedding)
        l_分类概率 = self.titan.m_classificationModel.predict_proba(l_slide_embedding)[:, :]

        # 通过enumerate获取索引和值，并构建字典
        l_dict_分类概率 = {i: val for i, val in enumerate(l_分类概率[0])}

        # 根据字典的值进行从大到小的排序
        sorted_items = dict(sorted(l_dict_分类概率.items(), key=lambda item: item[1], reverse=True))
        # 将排序后的项转换回字典类型
        sorted_dict = dict(sorted_items)
        l_result = ''

        for key, val in sorted_dict.items():
            l_result += C_Titan.疾病分类[int(key)] + ': ' + f"{val:.7f}" + '\n'
        #res.put(l_result)
        self.m_titan_result_signal.emit(l_result)

    #Titan模型预测
    def f_Titan模型预测(self, img):
        print('读取SVS文件')
        self.m_lineEdit_Hint.setText('读取SVS文件')
        self.m_lineEdit_Hint.update()
        if not self.f_读取SVS文件():
            return

        '''
        # 创建一个队列对象，用于存储线程的返回值
        result_queue = queue.Queue()
        l_thread1 = threading.Thread(target=self.f_Titan开始)
        l_thread1.start()
        '''
        self.m_textEdit_surfaceP.setPlainText('')
        # 创建工作线程和Worker对象
        if self.Titan_thread is None:
            self.Titan_thread = QThread()
        if self.Titan_worker is None:
            self.Titan_worker = Worker(self)
        self.Titan_worker.moveToThread(self.Titan_thread)
        # 连接Worker的信号到主线程的槽
        self.Titan_worker.update_text.connect(self.on_update_text)
        # 在工作线程中启动Worker的工作
        #QObject.connect(thread, QObject.started, worker.do_work)

        self.Titan_thread.started.connect(self.Titan_worker.do_work)
        self.Titan_thread.start()


    #表面缺陷检测
    def f_surfaceDefectsDetecting(self):
        if self.m_SurfaceDefectsDetectingModel is None:
            #print('f_loadSurfaceDefectsDetectingModel')
            self.f_loadSurfaceDefectsDetectingModel()
            
        #定义图像预处理流程
        preprocess = transforms.Compose([
            transforms.Resize(224),#(224, 224)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        print('f_surfaceDefectsDetecting pic size:' + str(self.m_img.width) + ', ' + str(self.m_img.height))
        l_imgTensor = preprocess(self.m_img)
        l_imgTensor = l_imgTensor.unsqueeze(0)  # 添加一个 batch 维度
        l_imgTensor = l_imgTensor.to(self.device)
        # Step 4: 进行推理
        with torch.no_grad():
            l_outputs = self.m_SurfaceDefectsDetectingModel(l_imgTensor)
            #_, predicted = torch.max(outputs, 1)
            l_prediction = l_outputs.argmax(dim=1)
            l_prediction_probabilities = torch.softmax(l_outputs, dim=1)
            l_top3_indices = l_prediction_probabilities.topk(3)[1].tolist()[0]
            
        l_result = ''
        for idx in l_top3_indices:
            l_result += f"{self.m_dicSurfaceDefectsDetecting.get(str(idx)):<5} - {l_prediction_probabilities[0][idx]:.1f}" + '\n'
        print(l_result)
        self.m_textEdit_surfaceP.setPlainText(l_result)
    
    #加载模型
    def f_loadSurfaceDefectsDetectingModel(self):
        #加载预训练的模型
        self.m_SurfaceDefectsDetectingModel = torch.load('model_full.pth', self.device)
        self.m_SurfaceDefectsDetectingModel = self.m_SurfaceDefectsDetectingModel.to(self.device)
        self.m_SurfaceDefectsDetectingModel.eval()  # 将模型设置为评估模式

if __name__ == "__main__":
    g_app = QApplication([])
    widget = LaiDunMedical()
    widget.m_uiMainWindow.show()
    #g_app.exec_()
    sys.exit(g_app.exec_())








