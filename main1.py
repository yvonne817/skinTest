# 先打開終端機安裝所需套件
# pip install streamlit ultralytics opencv-python-headless pillow

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from ultralytics import YOLO

# --- 設定頁面標題 ---
st.set_page_config(page_title="皮膚偵測 AI 系統", layout="wide")
st.title("🔍 皮膚偵測與分析系統")
st.write("上傳圖片並調整亮度，即可進行即時 AI 偵測")

# --- 載入模型 (快取處理) ---
@st.cache_resource
def load_model():
    # 請確保 best.pt 放在與 app.py 同一個資料夾下
    return YOLO("best.pt")

model = load_model()

# --- 側邊欄設定 ---
st.sidebar.header("參數設定")
# 亮度滑桿：範圍 0.5 到 2.0，預設 1.0 (不變)
brightness = st.sidebar.slider("圖片亮度調整", 0.5, 2.0, 1.0, 0.1)
# 信心度門檻
conf_threshold = st.sidebar.slider("AI 信心度門檻", 0.1, 1.0, 0.25, 0.05)

# --- 圖片上傳區域 ---
uploaded_file = st.file_uploader("請選擇一張皮膚照片 (jpg, png, jpeg)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 讀取圖片
    image = Image.open(uploaded_file)
    
    # 1. 調整亮度 (使用 PIL ImageEnhance)
    enhancer = ImageEnhance.Brightness(image)
    processed_image = enhancer.enhance(brightness)
    
    # 建立左右對照畫面
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("待測圖片 (已調亮度)")
        st.image(processed_image, use_container_width=True)
    
    # 2. 進行 YOLOv8 偵測
    # 將 PIL 轉為 OpenCV 格式供模型使用
    img_array = np.array(processed_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    if st.button("開始 AI 偵測"):
        with st.spinner('AI 正在分析中...'):
            results = model.predict(source=img_bgr, conf=conf_threshold)
            
            # 取得結果物件的第一項 (因為我們只傳一張圖)
            result = results[0]
            
            # 取得畫好框的圖片 (BGR 轉 RGB)
            annotated_img = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("偵測結果")
                st.image(annotated_img, use_container_width=True)
                
                # --- 新增程式碼：顯示詳細數據 ---
                # 判斷是否有偵測到物件
                if len(result.boxes) > 0:
                    st.divider() # 分隔線
                    st.caption("詳細偵測數據列表：")
                    
                    # 建立一個列表來儲存數據
                    detection_data = []
                    
                    # 取得類別名稱的字典 {0: 'acne', 1: 'spot', ...}
                    names = result.names
                    
                    # 遍歷每一個偵測到的框
                    for box in result.boxes:
                        # 取得類別 ID (整數)
                        cls_id = int(box.cls[0])
                        # 取得類別名稱
                        class_name = names[cls_id]
                        # 取得信心指數 (浮點數)
                        confidence = float(box.conf[0])
                        
                        # 加入列表
                        detection_data.append({
                            "偵測類別": class_name,
                            "信心指數": f"{confidence:.2%}" # 轉為百分比格式，例如 85.20%
                        })
                    
                    # 使用 Streamlit 的表格元件顯示
                    st.table(detection_data)
                
            # 顯示偵測統計文字
            num_detections = len(result.boxes)
            if num_detections > 0:
                st.success(f"偵測完成！共發現 {num_detections} 處目標。")
            else:
                st.warning("未偵測到任何目標，建議調整亮度或降低信心度門檻。")

#打開終端機執行指令
# streamlit run main.py --server.fileWatcherType none