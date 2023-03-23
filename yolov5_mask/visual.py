# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import shutil
import PyQt5.QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import os.path as osp
from models.common import DetectMultiBackend
from utils.datasets import LoadImages, LoadStreams
from utils.general import (LOGGER, check_img_size, check_imshow,
                           non_max_suppression, scale_coords, xyxy2xywh, increment_path)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Please Put on Your Mask')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon('USER/UIimages/logo.jpg'))
        # 读取图片
        self.output_size = 480
        self.img2predict = ""
        self.device = 'cpu'  # 可进行修改 默认用户使用cpu，改成0为启用gpu
        # 视频读取
        self.vid_source = ""  # 初始设置为空
        self.stopEvent = threading.Event()  # threading.Event 相当于贯穿多个线程之间的条件变量,用它能够很方便的来控制多个线程的行为
        self.webcam = True
        self.stopEvent.clear()
        self.model = self.load_model_pt(weights="runs/train/exp3/weights/best.pt",
                                        device=self.device)
        self.init_img()
        self.init_video()
        self.init_about()
        self.reset_vid()

    def setup_ui(self):
        self.load_model_pt()
        self.init_img()
        self.init_video()
        self.init_about()
        self.upload_img()
        self.detect_img()
        self.save_det_img()
        self.open_cam()
        self.open_mp4()
        self.detect_vid()
        self.reset_vid()
        self.close_vid()
        self.closeEvent()

    #  加载模型
    @torch.no_grad()
    def load_model_pt(self, weights="",
                      device='',
                      half=False,
                      ):
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device)
        half &= model.pt and device.type != 'cpu'
        if model.pt:
            model.model.half() if half else model.model.float()
        return model

    # 图片检测界面
    def init_img(self):
        # 图片检测
        # 相关字体
        title_size = QFont('楷体', 20)
        main_size = QFont('楷体', 14)

        # 整个界面
        all_widget_1 = QWidget()
        all_layout_1 = QVBoxLayout()

        # 第一行标题
        first_1 = QLabel("图片检测")
        first_1.setFont(title_size)

        # 第二行图片
        second_1_widget = QWidget()
        second_1_layout = QHBoxLayout()
        self.img_1 = QLabel()
        self.img_2 = QLabel()
        self.img_1.setPixmap(QPixmap("USER/left_mask.jpg"))
        self.img_2.setPixmap(QPixmap("USER/right_mask.jpg"))
        self.img_1.setAlignment(Qt.AlignCenter)
        self.img_2.setAlignment(Qt.AlignCenter)
        second_1_layout.addWidget(self.img_1)
        second_1_layout.addWidget(self.img_2)
        second_1_widget.setLayout(second_1_layout)

        # 第三行按钮

        third_1_widget = QWidget()
        third_1_layout = QHBoxLayout()

        up_button1 = QPushButton("打开图片")
        up_button1.setFont(main_size)
        up_button1.clicked.connect(self.upload_img)

        dec_button1 = QPushButton("开始检测")
        dec_button1.setFont(main_size)
        dec_button1.clicked.connect(self.detect_img)

        save_button1 = QPushButton("保存结果")
        save_button1.setFont(main_size)
        save_button1.clicked.connect(self.save_det_img)

        up_button1.setStyleSheet("QPushButton{color:rgb(26, 79, 163)}"
                                 "QPushButton:hover{background-color: rgb(2,110,208);}"
                                 "QPushButton{background-color:rgb(222, 230, 248)}"
                                 "QPushButton{border:6px}"
                                 "QPushButton{border-radius:15px}"
                                 "QPushButton{padding:15px 15px}"
                                 "QPushButton{margin:15px 15px}")

        dec_button1.setStyleSheet("QPushButton{color:rgb(26, 79, 163)}"
                                  "QPushButton:hover{background-color: rgb(2,110,208);}"
                                  "QPushButton{background-color:rgb(222, 230, 248)}"
                                  "QPushButton{border:6px}"
                                  "QPushButton{border-radius:15px}"
                                  "QPushButton{padding:15px 15px}"
                                  "QPushButton{margin:15px 15px}")

        save_button1.setStyleSheet("QPushButton{color:rgb(26, 79, 163)}"
                                   "QPushButton:hover{background-color: rgb(2,110,208);}"
                                   "QPushButton{background-color:rgb(222, 230, 248)}"
                                   "QPushButton{border:6px}"
                                   "QPushButton{border-radius:15px}"
                                   "QPushButton{padding:15px 15px}"
                                   "QPushButton{margin:15px 15px}")

        third_1_layout.addWidget(up_button1)
        third_1_layout.addWidget(dec_button1)
        third_1_layout.addWidget(save_button1)
        third_1_widget.setLayout(third_1_layout)

        all_layout_1.addWidget(first_1, alignment=Qt.AlignCenter)
        all_layout_1.addWidget(second_1_widget, alignment=Qt.AlignCenter)
        all_layout_1.addWidget(third_1_widget)
        all_widget_1.setLayout(all_layout_1)

        self.addTab(all_widget_1, '图片')
        self.setTabIcon(0, QIcon("USER/mask2.jpg"))

    # 视频检测界面
    def init_video(self):
        title_size = QFont('楷体', 20)
        main_size = QFont('楷体', 14)

        # 整个界面
        all_widgt_2 = QWidget()
        all_layout_2 = QHBoxLayout()

        # 标签
        labelwidget = QWidget()
        labellayout = QVBoxLayout()
        first_2 = QLabel("视频检测")
        first_2.setObjectName("first_2")
        first_2.setFont(title_size)
        first_2.setAlignment(Qt.AlignCenter)
        first_2_2 = QLabel("（预测结果预览会自动保存）\n*save_det_vid*")
        first_2_2.setFont(main_size)
        first_2_2.setAlignment(Qt.AlignCenter)
        labellayout.addWidget(first_2)
        labellayout.addWidget(first_2_2)
        labelwidget.setLayout(labellayout)

        # 上传代表图

        self.second_2 = QLabel()
        self.second_2.setPixmap(QPixmap("USER/right_mask3.jpg"))
        self.second_2.setAlignment(Qt.AlignCenter)

        # 三个按钮
        button_widget = QWidget()
        button_layout = QVBoxLayout()

        self.third_2 = QPushButton("打开摄像头")
        self.third_2.setFont(main_size)

        self.fourth_2 = QPushButton("打开视频")
        self.fourth_2.setFont(main_size)

        self.fifth_2 = QPushButton("停止检测")
        self.fifth_2.setFont(main_size)

        # self.sixth_2 = QPushButton("保存检测结果")
        # self.sixth_2.setFont(main_size)

        self.third_2.setStyleSheet("QPushButton{color:rgb(26, 79, 163)}"
                                   "QPushButton:hover{background-color: rgb(2,110,208);}"
                                   "QPushButton{background-color:rgb(222, 230, 248)}"
                                   "QPushButton{border:6px}"
                                   "QPushButton{border-radius:15px}"
                                   "QPushButton{padding:15px 15px}"
                                   "QPushButton{margin:15px 15px}")

        self.fourth_2.setStyleSheet("QPushButton{color:rgb(26, 79, 163)}"
                                    "QPushButton:hover{background-color: rgb(2,110,208);}"
                                    "QPushButton{background-color:rgb(222, 230, 248)}"
                                    "QPushButton{border:6px}"
                                    "QPushButton{border-radius:15px}"
                                    "QPushButton{padding:15px 15px}"
                                    "QPushButton{margin:15px 15px}")

        self.fifth_2.setStyleSheet("QPushButton{color:rgb(26, 79, 163)}"
                                   "QPushButton:hover{background-color: rgb(2,110,208);}"
                                   "QPushButton{background-color:rgb(222, 230, 248)}"
                                   "QPushButton{border:6px}"
                                   "QPushButton{border-radius:15px}"
                                   "QPushButton{padding:15px 15px}"
                                   "QPushButton{margin:15px 15px}")

        # self.sixth_2.setStyleSheet("QPushButton{color:rgb(26, 79, 163)}"
        #                            "QPushButton:hover{background-color: rgb(2,110,208);}"
        #                            "QPushButton{background-color:rgb(222, 230, 248)}"
        #                            "QPushButton{border:6px}"
        #                            "QPushButton{border-radius:15px}"
        #                            "QPushButton{padding:15px 15px}"
        #                            "QPushButton{margin:15px 15px}")

        self.third_2.clicked.connect(self.open_cam)
        self.fourth_2.clicked.connect(self.open_mp4)
        self.fifth_2.clicked.connect(self.close_vid)

        button_layout.addWidget(self.third_2)
        button_layout.addWidget(self.fourth_2)
        button_layout.addWidget(self.fifth_2)
        button_widget.setLayout(button_layout)

        all_layout_2.addWidget(labelwidget)
        all_layout_2.addWidget(self.second_2)
        all_layout_2.addWidget(button_widget)
        all_widgt_2.setLayout(all_layout_2)

        self.addTab(all_widgt_2, '视频')
        self.setTabIcon(1, QIcon('USER/mask2.jpg'))

    # 相关信息
    def init_about(self):
        title_size = QFont('楷体', 20)
        main_size = QFont('楷体', 14)

        all_widegt_3 = QWidget()
        all_layout_3 = QVBoxLayout()

        # 标签
        first_3 = QLabel("感谢使用基于yolov5的口罩检测系统")
        first_3.setFont(title_size)
        first_3.setAlignment(Qt.AlignCenter)

        # 图片
        second_3 = QLabel()
        second_3.setObjectName("second_3")
        second_3.setPixmap(QPixmap('USER/left_mask_3.jpg'))
        second_3.setAlignment(Qt.AlignCenter)

        # 呼吁大家带上口罩
        third_3 = QLabel("众志成城抗疫情，平凡人生见真情!\n愿早日战胜疫情，城市早日正常运转。")
        third_3.setFont(main_size)

        # 作者及相关信息
        fourth_3 = QLabel()
        fourth_3.setText("绝不可掉以轻心\n绝不可置身事外"
                         )
        fourth_3.setFont(main_size)
        fourth_3.setAlignment(Qt.AlignRight)

        # 添加widegt
        all_layout_3.addWidget(first_3)
        all_layout_3.addWidget(second_3)
        all_layout_3.addWidget(third_3)
        all_layout_3.addWidget(fourth_3)
        all_widegt_3.setLayout(all_layout_3)

        self.addTab(all_widegt_3, '关于')
        self.setTabIcon(2, QIcon('USER/mask2.jpg'))

    """
    打开图片
    """

    # 这里采用的是开源代码改良后进行上传图片操作
    def upload_img(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("USER/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("USER/tmp/upload_show_result.jpg", im0)
            self.img2predict = fileName
            self.img_1.setPixmap(QPixmap("USER/tmp/upload_show_result.jpg"))
            self.img_2.setPixmap(QPixmap("USER/right_mask.jpg"))

    """
    检测图片
    """

    # 对开源代码进行改良达成目的
    def detect_img(self):
        model = self.model
        output_size = self.output_size
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        imgsz = [480, 480]  # inference size (pixels)
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        view_img = False  # show results
        save_txt = False  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 3  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        print(source)

        if source == "":
            QMessageBox.warning(self, "Upload Photo", "there is NO IMAGES to detect")
        else:
            source = str(source)
            device = select_device(self.device)
            webcam = False
            stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            save_img = not nosave and not source.endswith('.txt')  # save inference images
            # Dataloader
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = 1  # batch_size
            # Run inference
            if pt and device.type != 'cpu':
                model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
            dt, seen = [0.0, 0.0, 0.0], 0

            # 只使用我们需要的部分
            for path, im, im0s, vid_cap, s in dataset:
                t1 = time_sync()
                im = torch.from_numpy(im).to(device)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                t2 = time_sync()
                dt[0] += t2 - t1
                # Inference
                pred = model(im, augment=augment, visualize=visualize)
                t3 = time_sync()
                dt[1] += t3 - t2
                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                dt[2] += time_sync() - t3
                for i, det in enumerate(pred):  # per image
                    seen += 1

                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    self.p = Path(p)

                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))

                    # Print time (inference-only)
                    LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                    # Stream results
                    self.im0 = annotator.result()
                    resize_scale = output_size / im0.shape[0]
                    self.im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                    cv2.imwrite("USER/tmp/single_result.jpg", self.im0)
                    self.img_2.setPixmap(QPixmap("USER/tmp/single_result.jpg"))

    # 保存图片检测结果的动作
    def save_det_img(self):
        name = 'detcet'
        project = ROOT / 'USER/save_det_img'
        exist_ok = False
        save_txt = False
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        source = self.img2predict
        if source == "":
            QMessageBox.warning(self, "Upload Photo", "there is NO IMAGES to save")

        else:
            p = self.p
            save_path = str(save_dir / p.name)
            cv2.imwrite(save_path, self.im0)
            QMessageBox.warning(self, "where you can find the resualts",
                                "the resualts are in save_det_img after quiting this projiect")

    """
    开启摄像头检测
    """

    def open_cam(self):
        self.third_2.setEnabled(False)
        self.fourth_2.setEnabled(False)
        self.fifth_2.setEnabled(True)
        self.vid_source = '0'
        self.webcam = True
        th = threading.Thread(target=self.detect_vid)
        th.start()

    """
    开启视频文件检测
    """

    def open_mp4(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.third_2.setEnabled(False)
            self.fourth_2.setEnabled(False)
            self.fifth_2.setEnabled(True)
            self.vid_source = fileName
            self.webcam = False
            th = threading.Thread(target=self.detect_vid)
            th.start()

    '''
    视频检测 
    '''

    # 视频和摄像头的主函数一样的。
    def detect_vid(self):
        model = self.model
        output_size = self.output_size
        imgsz = [640, 640]  # inference size (pixels)
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        # device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results
        save_txt = False  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 3  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference

        project = ROOT / 'USER/save_det_vid'
        name = 'exp'
        exist_ok = False
        source = str(self.vid_source)
        webcam = self.webcam

        device = select_device(self.device)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = 10  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs
        # Run inference
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                -1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                # Print time (inference-only)
                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                # Stream results

                im0 = annotator.result()
                frame = im0

            if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                self.stopEvent.clear()
                self.fifth_2.setEnabled(True)
                self.fourth_2.setEnabled(True)
                self.reset_vid()
                break

            if vid_path[i] != save_path:  # new video
                vid_path[i] = save_path
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path += '.mp4'
                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[i].write(im0)

            resize_scale = output_size / frame.shape[0]
            frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("USER/tmp/single_result_vid.jpg", frame_resized)
            self.second_2.setPixmap(QPixmap("USER/tmp/single_result_vid.jpg"))

    # 在每次重新启动时进行界面重置
    def reset_vid(self):
        self.third_2.setEnabled(True)
        self.third_2.setEnabled(True)
        self.fourth_2.setEnabled(True)
        self.second_2.setPixmap(QPixmap("USER/right_mask3.jpg"))
        self.vid_source = ""
        self.webcam = True

    # 对视频进行重置
    def close_vid(self):
        if self.vid_source == "":
            QMessageBox.warning(self, 'please upload video', 'there is no source to detect')
        else:
            self.stopEvent.set()
            self.reset_vid()

    # 退出界面
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'quiting',
                                     "Before quiting, "
                                     " please put on your MASK",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
