import os
import cv2
import streamlit as st
import streamlit.components.v1 as components
from st_on_hover_tabs import on_hover_tabs
from streamlit_option_menu import option_menu
from pathlib import Path
import PIL
from PIL import Image
import numpy as np
import tempfile
import torch
import wget
import glob
import time
import av
from ultralytics import YOLO

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


confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

# ========================================
st.markdown('''<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Anton&display=swap" rel="stylesheet">
            <h1 style='text-align: center;'>Traffic Police</h1>''', unsafe_allow_html=True)


selected2 = option_menu(None, ["Home", "Tracking","Dashboard", "info"], 
    icons=['house', 'gear',"bi-bar-chart", "list-task"], 
    menu_icon="cast", default_index=0, orientation="horizontal")


if selected2 == "Home":
    st.subheader('Real-time traffic ovject detection using YoLo model')

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

    st.markdown("""
                <h3> 프로젝트 설명 </h3>
                <p> 최근 개인형 이동장치의 사용향이 증가함에 따라 교통법규 위반으로 인한 다수의 사고가 발생하고 있다.
                그 반면 위법 행위 적발 수치는 낮은 편이다.
                이에 'A' 기간은 교통법규 위반 탐지 모델을 생성하여 적방 수치를 높임으로써 사고를 방지하고, 시민들이 보다
                안전하게 생활할 수 있는 환경을 조성하고자 한다.<br></p>
    """, unsafe_allow_html=True)
    st.markdown("""
                <h3> 팀 목표 </h3>
                <p> 실시간 CCTV로 딥러닝 모델을 이용하여 ovject detection 실시간하여 해당 지역의 위법 상황 지표를
                만들 수 있는 데이터 수집 모델을 제작<br></p>
    """, unsafe_allow_html=True)
    st.markdown("""
                <h3> Object Detection </h3>
                <p> Object Detection 객체 인식은 이미지 또는 비디오에서 개체를 식별하고 찾는 것과 관련된 컴퓨터 비전 작업이다.
                감지된 물제는 사람, 자동차, 건물, 동물일 수 있다. 기존 객체 인식은 다양한 접근 방식으로 시도되면서 데이터 제한 및 모델링 문제를 해결하려고 했습니다.
                하지만 단일 알고리즘 실행을 통해 객체를 감지하는 것이 어려웠던 차에, YOLO 알고리즘이 등장하게 되었죠.<br>
                 </p>
    """, unsafe_allow_html=True)
    st.markdown("""
                <h3> You Only Look Once </h3>
                <p> YOLO(You Only Look Once)는 최첨단 실시간 Object Detection 시스템입니다. 
                기존의 모델보다 빠르고 정확한 데이터 처리 속도를 자랑하며 화제를 몰고 왔죠. YOLO는 물체 감지와 객체 인식에 대한 딥러닝 기반 접근 방식입니다.
                <br>
                간단하게 말하자면, YOLO는 입력된 이미지를 일정 분할로 그리드한 다음, 신경망을 통과하여 바운딩 박스와 클래스 예측을 생성하여 최종 감지 출력을 결정합니다. 실제 이미지 및 비디오에서 테스트하기 전에 먼저 전체 데이터 세트에 대해 여러 인스턴스를 학습하죠.
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
                        st.session_state.clear()
                        break
                    results = model(frame)
                    labeled_frame = np.array(results.render()[0])
                    placeholder.image(labeled_frame, channels="BGR", use_column_width=True)
                    

            # YoLo v8
            if st.sidebar.button("YoLo v8 RUN", key="YoLo v8"):
                stop_button = st.sidebar.button("STOP", key="btn_yolo_STOP")
                model2 = YOLO(yolov8_model_path, device_option)
                placeholder2 = st.empty()
                for frame in get_frames_from_m3u8(st.session_state.video_url):
                    if stop_button:
                        st.session_state.clear()
                        break
                    res = model2.predict(frame,
                        conf=confidence)
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    placeholder2.image(res_plotted, channels="BGR", use_column_width=True)

            if 'stop_button' in globals() and stop_button:
                st.session_state.clear()

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