import cv2
import time
from yoloseg import YOLOSeg

# 提供的 class_names 列表
class_names = [
    '6-Babcock-Tissue-Forceps', '6-Mayo-Needle-Holder', '7-Metzenbaum-Scissors', 
    '8-Babcock-Tissue-Forceps', '8-Mayo-Needle-Holder', '9-Metzenbaum-Scissors', 
    'Adson-Smooth-Tissue-Forceps', 'Adson-Teeth-Tissue-Forceps', 
    'Allis-Grasping-Forceps', 'Bonneys-Non-Toothed-Dissector', 
    'Fritsch-Abdominal-Retractor', 'Kelly-Forceps-Cvd', 
    'Kelly-Forceps-Str', 'Knife-Handle-No3', 'Knife-Handle-No4', 
    'Knife-Handle-No7', 'Kocher-Forceps', 'Mastoid-Retractor', 
    'Mayo-Scissors-Cvd', 'Mosquito-Forceps-Cvd', 'Patten-Retractor', 
    'Ring-Forceps', 'Smooth-Tissue-Forceps', 'Suction-Tube-Fr', 
    'Suction-Tube-Po', 'Suction-Tube-Ya', 'Suture-Scissors', 
    'Teeth-Tissue-Forceps', 'Towel-Clamp'
]

def load_model(model_path, conf_thres=0.45, iou_thres=0.5):
    """載入YOLO模型."""
    print('Loading model...')
    yoloseg = YOLOSeg(model_path, conf_thres=conf_thres, iou_thres=iou_thres)
    return yoloseg

def connect_camera(camera_id=0, width=1280, height=720, fps=60):
    """連接攝影機並設定參數."""
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Requested Width: {width}, Actual Width: {actual_width}")
    print(f"Requested Height: {height}, Actual Height: {actual_height}")
    print(f"Requested FPS: {fps}, Actual FPS: {actual_fps}")
    return cap

def get_frame_with_detections(yoloseg, cap):
    """從攝影機獲取影像並進行物件偵測，返回帶有標註的影像."""
    ret, frame = cap.read()
    height, width, channels = frame.shape
    #print(width, height)
    frame = cv2.resize(frame, (640, 640))
    if not ret:
        return None

    # 偵測物件
    boxes, scores, class_ids, masks = yoloseg(frame)
    
    # 根據 class ID 獲取物件名稱
    sorted_names = get_object_names_by_id(class_ids, class_names)
    object_counts = Counter(sorted_names)
    '''
    # 輸出結果
    for object_name, count in object_counts.items():
        print(f"{object_name}: {count}")
        '''
    combined_img = yoloseg.draw_masks(frame)
    return combined_img, object_counts

# 自定義 Counter 函數
def Counter(items):
    counts = {}
    for item in items:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    return counts
    
# 根據 class ID 獲取物件名稱
def get_object_names_by_id(class_ids, class_names):
    """根據 class 名稱返回物件名稱，並按照名稱的順序排列。

    參數:
        class_ids: 檢測到的物件 ID 列表
        class_names: 物件名稱列表

    返回:
        sorted_names: 按照名稱排序的物件名稱列表
    """
    # 將 class_ids 與對應的 class_names 組合成一個列表
    id_name_pairs = [(id, class_names[id]) for id in class_ids if id < len(class_names)]
    
    # 按照名稱排序
    sorted_pairs = sorted(id_name_pairs, key=lambda x: x[1])
    
    # 只返回名稱部分
    sorted_names = [name for _, name in sorted_pairs]
    return sorted_names
    
'''
# webcam測試用

def start_detection(yoloseg, cap):
    """啟動物件偵測並顯示影像."""
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

    prev_time = 0
    while cap.isOpened():
        current_time = time.time()
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 640))
        
        if not ret:
            break

        boxes, scores, class_ids, masks = yoloseg(frame)
        combined_img = yoloseg.draw_masks(frame)

        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(combined_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Detected Objects", combined_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


model_path = "models/29_s_40gen_16batch_1024.onnx"
yoloseg = load_model(model_path)
cap = connect_camera()
start_detection(yoloseg, cap)
'''