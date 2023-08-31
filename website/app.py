import os
import cv2
import streamlit as st
import streamlit.components.v1 as components
from st_on_hover_tabs import on_hover_tabs
from streamlit_option_menu import option_menu
from pathlib import Path
from PIL import Image
import numpy as np
import tempfile
import torch
import wget
import glob
import time
import av

# ======================================== 웹 사이트 기능
cctv = {
    'CCTV 지역을 선택하세요.': 'pass',
    '부산우체국': 'http://61.43.246.225:1935/rtplive/cctv_4.stream/playlist.m3u8',
    '부산 소방서 삼거리': 'https://streamits.gyeongju.go.kr:1935/live/live42.stream/playlist.m3u8',
    '부산 개성중(하)': 'http://61.43.246.225:1935/rtplive/cctv_84.stream/playlist.m3u8',
    '인천 만수사거리': 'http://61.40.94.13:1935/cctv/L1003.stream/playlist.m3u8',
    '포항 두호 롯데아파트': 'http://220.122.218.201:1935/live/62.stream/playlist.m3u8',
    '포항 마산사거리': 'http://220.122.218.201:1935/live/78.stream/playlist.m3u8',
    '포항 송도교사거리': 'http://220.122.218.201:1935/live/7.stream/playlist.m3u8',
    '포항 대도사거리': 'http://220.122.218.201:1935/live/14.stream/playlist.m3u8',
    '경주 안강네거리': 'https://streamits.gyeongju.go.kr:1935/live/live59.stream/playlist.m3u8',
    '광주 계림오거리': 'https://gjtic.go.kr/cctv6/livehttp/2024_video2/chunklist.m3u8',
    '광주 서방사거리': 'https://gjtic.go.kr/cctv5/livehttp/1032_video2/chunklist.m3u8',
    '대구 대구역내거리': 'http://210.91.152.35:1935/live2/_definst_/ch87.stream/playlist.m3u8',
    '대구 덕천치안센터': 'http://210.91.152.35:1935/live4/_definst_/ch274.stream/playlist.m3u8'
}
# 모델 불러오기
def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    print("model to ", device)
    return model_

def get_frames_from_m3u8(url):
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame



yolov5_model_path = 'models/yolov5_all800_best.pt'
yolov8_model_path = 'models/val_epoch100.pt'

# ========================================
st.markdown("<h1 style='text-align: center;'>Traffic Police</h1>", unsafe_allow_html=True)


selected2 = option_menu(None, ["Home", "Tracking","Dashboard", "info"], 
    icons=['house', 'gear',"bi-bar-chart", "list-task"], 
    menu_icon="cast", default_index=0, orientation="horizontal")


if selected2 == "Home":
    # 해당 유튜브 영상의 임베드 링크
    youtube_link = "https://www.youtube.com/embed/cHDLvp_NPOk"
    # Streamlit에 임베드
    st.write(f"""
    <style>
        .videoWrapper {{
            position: relative;
            padding-bottom: 56.25%; /* 16:9 */
            padding-top: 25px;
            height: 0;
            text-align: center; /* 중앙 정렬 */
        }}
        .videoWrapper iframe {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }}
    </style>
    <div class="videoWrapper">
        <iframe width="560" height="315" src="{youtube_link}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>""", unsafe_allow_html=True)
    # 비디오 재생
    # video_file = open('myvideo.mp4', 'rb')
    # video_bytes = video_file.read()
    # st.video(video_bytes)

    st.subheader('YOLO (You Only Look Once) :desktop_computer:')
    st.markdown("""
    <p>
        실시간 객체 탐지를 위한 혁신적인 딥 러닝 기반의 알고리즘입니다. 
        기존의 많은 객체 탐지 알고리즘들이 이미지 내의 여러 영역을 반복적으로 검사하는 반면, 
        YOLO는 이미지를 한 번만 보고 여러 객체를 탐지합니다. 이 접근 방식 덕분에 YOLO는 빠른 속도와 높은 정확도를 동시에 
        제공하며 실시간 탐지 작업에 매우 적합합니다.
    </p>
    """, unsafe_allow_html=True)


elif selected2 == "Tracking":
        st.header("Tracking")
        location = st.selectbox("Select a CCTV location:", list(cctv.keys()))
        
        if location != 'CCTV 지역을 선택하세요.':
            st.session_state.video_url = cctv[location] 
            video_capture = cv2.VideoCapture(st.session_state.video_url)

            st.write(f"Displaying stream for: {location}")
            frame_window = st.image([])
            st.write("영상 제공 : 경찰청 도시교통정보센터(UTIC)")

            st.sidebar.title("Tracking option")
            # cpu와 cuda을 사용할지 설정하는 항목
            if torch.cuda.is_available():
                device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
            else:
                device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

            # YOLO v5 모델링 tracking 작업
            if st.sidebar.button("YoLo v5 RUN", key="YoLo v5"):
                stop_button = st.sidebar.button("STOP", key="btn_yolo_STOP")
                model = load_model(yolov5_model_path, device_option)
                placeholder = st.empty()
                for frame in get_frames_from_m3u8(st.session_state.video_url):
                    if stop_button:
                        break
                    results = model(frame)
                    labeled_frame = np.array(results.render()[0])
                    placeholder.image(labeled_frame, channels="BGR", use_column_width=True)
                    
            if 'stop_button' in globals() and stop_button:
                st.session_state.clear()
            # YoLo v8
            if st.sidebar.button("YoLo v8 RUN", key="YoLo v8"):
                model = load_model(yolov8_model_path, device_option)

            while True:
                ret, frame = video_capture.read()
                if not ret:
                    st.warning("Error retrieving the video frame.")
                    break
                frame_window.image(frame, channels="BGR")



elif selected2 == "Dashboard":
    st.header("Dashboard Page")

        
elif selected2 == "info":
    st.header("Information Page")
    # 여기에 Reference 페이지의 추가 내용을 추가하세요.