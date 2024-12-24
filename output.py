import cv2
import numpy as np

# 影片路徑
video_path = r"C:\Users\TMP214\Desktop\zelda.mp4"

# 初始化影片
cap = cv2.VideoCapture(video_path)

# 取得影片的fps（每秒幀數）
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 創建 VideoWriter 用來保存處理後的影片
output_video_path = r"C:\Users\TMP214\Desktop\processed_zleda.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v編碼
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 定義時間區段及對應的處理方法
segments = [
    (0, 7, 'original'),
    (7, 20, 'sift'),
    (20, 27, 'gradient'),
    (27, 38, 'color_balance'),
    (38, 54, 'tophat'),
    (54, 59, 'dog'),
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
            # 在左上角顯示方法名稱
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Method: {method}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if method == 'original':
                pass  # 原始影像，不做任何變化

            elif method == 'sift':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                keypoints, descriptors = sift.detectAndCompute(gray, None)
                frame = cv2.drawKeypoints(frame, keypoints, None)

            elif method == 'dog':
                dog_image = dog(frame)
                frame = cv2.cvtColor(dog_image, cv2.COLOR_GRAY2BGR)  # 轉換回彩色影像
                cv2.putText(frame, "DOG", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            elif method == 'tophat':
                k = np.ones((5, 5), np.uint8)
                tophat_image = cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, k)
                frame = tophat_image

            elif method == 'gradient':
                kernel = np.ones((3, 3), np.uint8)
                gradient = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel)
                frame = gradient

            elif method == 'color_balance':
                (b, g, r) = cv2.split(frame)  # 彩色影像均衡化，對每一個通道均衡化
                bH = cv2.equalizeHist(b)
                gH = cv2.equalizeHist(g)
                rH = cv2.equalizeHist(r)
                frame = cv2.merge((bH, gH, rH))  # 合併每一個通道

            # 寫入處理後的幀到影片中
            out.write(frame)

            # 顯示處理後的影片幀
            cv2.imshow('Processed Frame', frame)

    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
