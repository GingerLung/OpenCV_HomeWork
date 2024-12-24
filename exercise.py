import cv2
import numpy as np

# 影片路徑
video_path = r"C:\Users\TMP214\Desktop\zelda.mp4"

# 初始化影片
cap = cv2.VideoCapture(video_path)

# 取得影片的fps（每秒幀數）
fps = cap.get(cv2.CAP_PROP_FPS)

# 定義時間區段及對應的處理方法
segments = [
    (0, 7, 'original'),
    (8, 20, 'sift'),
    (21, 27, 'gradient'),
    (28, 30, 'color_balance'),
    (31, 38, 'oirginal'),
    (39, 54, 'tophat'),
    (55, 59, 'color_balance'),
]

# 創建SIFT檢測器
sift = cv2.SIFT_create()

# 初始化DOG (Difference of Gaussian) 方法
def dog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用高斯模糊來計算DOG
    blur1 = cv2.GaussianBlur(gray, (3, 3), 0)
    blur2 = cv2.GaussianBlur(gray, (5, 5), 0)    
    return blur1 - blur2

# 讀取影片並進行處理
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 當前幀的時間戳（秒）
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # 根據時間選擇處理方法
    for start, end, method in segments:
        if start <= current_time < end:
            if method == 'original':
                cv2.imshow('Frame - Original', frame)

            elif method == 'sift':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                keypoints, descriptors = sift.detectAndCompute(gray, None)
                frame_sift = cv2.drawKeypoints(frame, keypoints, None)
                cv2.imshow('Frame - SIFT', frame_sift)

            elif method == 'dog':
                dog_image = dog(frame)
                cv2.imshow('Frame - DOG', dog_image)

            elif method == 'tophat':
                k=np.ones((5,5),np.uint8)
                tophat_image=cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, k)
                cv2.imshow('Frame - TOPHAT', tophat_image)

            elif method == 'gradient':
                kernel = np.ones((3, 3), np.uint8)
                gradient = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel)
                cv2.imshow('Frame - GRADIENT', gradient)
            elif method == 'color_balance':    
                (b, g, r) = cv2.split(frame)        # 彩色影像均衡化,需要分解通道 對每一個通道均衡化
                bH = cv2.equalizeHist(b)
                gH = cv2.equalizeHist(g)
                rH = cv2.equalizeHist(r)
                color = cv2.merge((bH, gH, rH))    # 合併每一個通道
                cv2.imshow('Frame - COLOR', color)
                    

    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
