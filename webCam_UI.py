import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtWidgets import QTextEdit, QGridLayout, QFrame, QSizePolicy, QScrollArea, QComboBox
from PyQt6.QtGui import QImage, QPixmap, QColor, QPalette, QTextCursor
from PyQt6.QtCore import Qt, QTimer
import cv2
import time

from yolo_detection import load_model, connect_camera, get_frame_with_detections

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        
        # 模型路徑
        self.model_path = "models/train_s_50gen_16batch_640_AUG_HP4.onnx"
        self.conf_thres = 0.45
        self.iou_thres = 0.5
        
        # 初始化下拉式選單資訊
        self.requirements = {
            "Surgical Instruments - Option A": {"6-Babcock-Tissue-Forceps": 1, "8-Babcock-Tissue-Forceps": 1},
            "Surgical Instruments - Option B": {"7-Metzenbaum-Scissors": 1, "8-Mayo-Needle-Holder": 1},
            "Surgical Instruments - Option C": {"Adson-Smooth-Tissue-Forceps":1, "Knife-Handle-No3":1, "Towel-Clamp": 1}
        }
        
        self.setWindowTitle("Surgical Tools Detection")
        self.setGeometry(100, 100, 800, 600)
        
        # 主佈局
        mainLayout = QHBoxLayout(self)

        # 左側：顯示攝影機畫面的畫布
        leftLayout = QVBoxLayout()
        self.cameraView = QLabel()
        self.cameraView.setFixedSize(640, 640)
        leftLayout.addWidget(self.cameraView)
        mainLayout.addLayout(leftLayout)

        # 右側：放置按鈕和訊息/物件顯示區域
        rightLayout = QVBoxLayout()

        # 右上部分 - 按鈕水平佈局，並排顯示按鈕
        buttonLayout = QHBoxLayout()
        self.loadModelBtn = QPushButton("Load Model")
        self.loadModelBtn.setFixedSize(120, 50)
        self.loadModelBtn.clicked.connect(self.load_model_and_start_camera)
        buttonLayout.addWidget(self.loadModelBtn)

        self.pauseCameraBtn = QPushButton("Start/Stop Camera")
        self.pauseCameraBtn.setFixedSize(120, 50)
        self.pauseCameraBtn.clicked.connect(self.pause_camera)
        self.pauseCameraBtn.setEnabled(False)  # 初始設為無效
        buttonLayout.addWidget(self.pauseCameraBtn)
        rightLayout.addLayout(buttonLayout)

        # 新增需求選單
        self.dropdown = QComboBox()
        self.dropdown.addItems(["Surgical Instruments - Option A",
                                "Surgical Instruments - Option B",
                                "Surgical Instruments - Option C"])
        self.dropdown.currentIndexChanged.connect(self.update_camera_view)
        self.dropdown.setEnabled(False)  # 初始設為無效
        rightLayout.addWidget(self.dropdown)

        # 右下部分 - 放置訊息框和物件類別資訊
        statusLayout = QVBoxLayout()

        # 即時顯示提示訊息框
        self.messageBox = QTextEdit()
        self.messageBox.setReadOnly(True)
        self.messageBox.setFixedHeight(150)  # 限制訊息框的高度
        self.messageBox.setFixedWidth(300)   # 限制訊息框的寬度
        statusLayout.addWidget(self.messageBox)

        # 獨立的QVBoxLayout來放置QGridLayout
        infoLayout = QVBoxLayout()

        #顏色說明標籤
        status_legend_layout = QHBoxLayout()
        # 完成 (淺綠色)
        complete_label = QLabel("Done")
        complete_label.setFixedSize(80, 30)
        complete_label.setStyleSheet("background-color: lightgreen; font-size: 12px;")
        complete_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_legend_layout.addWidget(complete_label)

        # 待收集 (橘色)
        pending_label = QLabel("Undone")
        pending_label.setFixedSize(80, 30)
        pending_label.setStyleSheet("background-color: sandybrown; font-size: 12px;")
        pending_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_legend_layout.addWidget(pending_label)

        # 不需要 (紅色)
        not_required_label = QLabel("Unnecessary")
        not_required_label.setFixedSize(80, 30)
        not_required_label.setStyleSheet("background-color: lightcoral; font-size: 12px;")
        not_required_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_legend_layout.addWidget(not_required_label)

        # 將圖例布局添加到 infoLayout 的最上方
        infoLayout.insertLayout(0, status_legend_layout)

        # 顯示物件類別和數量的表格
        self.infoGrid = QGridLayout()
        self.infoFrame = QFrame()
        self.infoFrame.setLayout(self.infoGrid)
        self.infoFrame.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.infoFrame.setFixedHeight(400)  # 限制物件資訊表格高度
        self.infoFrame.setFixedWidth(300)   # 限制物件資訊表格的寬度

        # 防止表格隨內容變化大小
        self.infoGrid.setSizeConstraint(QGridLayout.SizeConstraint.SetFixedSize)  # 設置固定大小
        self.infoGrid.setColumnStretch(0, 1)  # 設置列的伸縮因子
        self.infoGrid.setRowStretch(0, 1)     # 設置行的伸縮因子

        # 添加一個滾動區域以防止內容增減時的跳動
        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)  # 使滾動區域內部的widget可以調整大小
        scrollArea.setFixedHeight(400)  # 限制滾動區域的高度
        scrollArea.setWidget(self.infoFrame)  # 設置滾動區域的widget

        infoLayout.addWidget(scrollArea)  # 將滾動區域添加到infoLayout
        statusLayout.addLayout(infoLayout)

        rightLayout.addLayout(statusLayout)
        mainLayout.addLayout(rightLayout)
        
        # 其他屬性
        self.camera_active = False
        self.camera_paused = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_view)
        self.cap = None
        self.yoloseg = None  # 模型變數
        self.prev_time = 0
        self.current_time = 0
        self.fps = 0
        self.detected_objects = {}
        
    def load_model_and_start_camera(self):
        # 載入模型並啟動攝影機
        self.add_message("Loading model and starting camera...")
        self.yoloseg = load_model(self.model_path, self.conf_thres, self.iou_thres)
        self.cap = connect_camera()
        
        if self.yoloseg and self.cap:  # 確認模型和攝影機成功載入
            self.camera_active = True
            self.timer.start(30)  # 設定30ms的更新頻率
            self.add_message("Model loaded and camera started.")
            
            # 啟用其他按鈕
            self.pauseCameraBtn.setEnabled(True)
            self.dropdown.setEnabled(True)
        else:
            self.add_message("Failed to load model or connect to camera.")
        
    def pause_camera(self):
        """暫停或恢復攝影機畫面更新。"""
        if self.camera_active:
            if self.camera_paused:
                # 如果目前為暫停狀態，則恢復攝影機更新
                self.timer.start(30)
                self.pauseCameraBtn.setText("Pause Camera")
                self.add_message("Camera resumed.")
            else:
                # 如果目前為活動狀態，則暫停攝影機更新
                self.timer.stop()
                self.pauseCameraBtn.setText("Resume Camera")
                self.add_message("Camera paused.")
            self.camera_paused = not self.camera_paused

    def update_camera_view(self):
        self.current_time = time.time()
        # 取得處理後的影像
        if self.cap and self.cap.isOpened() and self.yoloseg:
            frame, object_counts = get_frame_with_detections(self.yoloseg, self.cap)
            self.fps = 1 / (self.current_time - self.prev_time)
            self.prev_time = self.current_time
            cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            if frame is not None:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.cameraView.setPixmap(pixmap)
                
                self.update_object_info(object_counts)

    def update_object_info(self, detected_objects=None):
        if detected_objects is not None:
            self.detected_objects = detected_objects
        elif not isinstance(self.detected_objects, dict):
            self.detected_objects = {}
            
        for i in reversed(range(self.infoGrid.count())):
            widget = self.infoGrid.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        selected_option = self.dropdown.currentText()
        requirements = self.requirements.get(selected_option, {})
        
        row = 0
        for item, required_count in requirements.items():
            current_count = self.detected_objects.get(item, 0)
            #類別標籤
            label_text = f"{item}"
            label = QLabel(label_text)
            label.setFixedSize(200, 30)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if current_count >= required_count:
                label.setStyleSheet("background-color: lightgreen; font-size: 12px;")
            else:
                label.setStyleSheet("background-color: sandybrown; font-size: 12px;")
            
            # 類別數量標籤
            count_text = f"{current_count}/{required_count}"
            count_label = QLabel(count_text)
            count_label.setFixedSize(50, 30)
            count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            count_label.setStyleSheet("font-size: 12px;")
            
            self.infoGrid.addWidget(label, row, 0)
            self.infoGrid.addWidget(count_label, row, 1)
            
            row += 1
        
        for item, count in self.detected_objects.items():
            if item not in requirements:
                #類別標籤
                label_text = QLabel(f"{item}")
                label_text.setFixedSize(200, 30)  # 寬250px，高30px
                label_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label_text.setStyleSheet("background-color: lightcoral; font-size: 12px;")
                                
                # 類別數量標籤
                count_text = f"{count}/0"
                count_label = QLabel(count_text)
                count_label.setFixedSize(50, 30)  # 寬250px，高30px
                count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                count_label.setStyleSheet("font-size: 12px;")
                
                self.infoGrid.addWidget(label_text, row, 0)
                self.infoGrid.addWidget(count_label, row, 1)

                row += 1
    
    def add_message(self, message):
        # 在 messageBox 中顯示訊息並自動滾動到最新訊息
        self.messageBox.append(message)
        self.messageBox.moveCursor(QTextCursor.MoveOperation.End)  # 自動滾動到最新訊息

if __name__ == "__main__":  
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())
