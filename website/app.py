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
import copy
from ultralytics import YOLO
import base64
import matplotlib.pyplot as plt
from collections import Counter

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

def load_yolov5_model(model_path, device='cuda'):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path).to(device)
    model.eval()
    return model

def detect_and_save_video(input_path, output_path, model, device='cuda'):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        rendered_frame = results.render()[0]

        out.write(rendered_frame)

    cap.release()
    out.release()

def apply_alpha_to_color(image, mask, color, alpha):
    overlay = image.copy()
    cv2.fillPoly(overlay, [mask], color)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def line(a, b, c, d, image):
    start_point = (a, b)
    end_point = (c, d)
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    width, height, _ = image.shape
    if dx != 0:
        slope = dy / dx  # 기울기
        if dx > 0:
            x1, y1 = start_point
            x2, y2 = width - 1, int((width - 1 - x1) * slope + y1)
        else:
            x1, y1 = end_point
            x2, y2 = 0, int(-x1 * slope + y1)
    else:
        x1, y1 = start_point
        x2, y2 = start_point[0], height - 1
    return x2, y2

yolov8_model_path = 'models/yolov8_all5000_added.pt'
title_image_path = 'static/title.png'
current_id = {}
same_id = []
real_results = []
segment_frame = 8

# ========================================
st.image(title_image_path,use_column_width=True)

selected2 = option_menu(None, ["Home", "Tracking", "info"], 
    icons=['house', 'gear', "list-task"], 
    menu_icon="cast", default_index=0, orientation="horizontal")


if selected2 == "Home":
    st.header("개인형 이동장치 위법 탐지 서비스")
    file_ = open('static/tracked_output.gif', "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" style="width: 100%">',
    unsafe_allow_html=True,)
    # 로고나 대표 이미지
    st.subheader("서비스 소개")
    st.write("""
    최근 개인형 이동장치의 사용량이 증가함에 따라 교통법규 위반으로 인한 다수의 사고가 발생하고 있습니다.
    이 서비스는 실시간 CCTV를 통해 YOLO 딥러닝 모델을 이용하여 위법 행위를 탐지하고, 안전한 환경 조성을 위한 데이터를 수집합니다.
    """)
    st.subheader("사용방법")
    st.write("""
    1. 위의 Tracking 페이지로 이동합니다.
    2. CCTV 영상을 선택합니다.
    3. YOLO 모델 실행을 선택하면 위법 행위를 탐지합니다.
    4. STOP 버튼을 누르면 트래킹이 종료됩니다.
    """)
    
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
                    
            # YoLo v8
            if st.sidebar.button("YoLo v8 RUN", key="YoLo v8"):
                stop_button = st.sidebar.button("STOP", key="btn_yolo_STOP")
                confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100
                st.sidebar.write('Chart on & off')
                bar_on = st.sidebar.toggle('Bar chart')
                pie_on = st.sidebar.toggle('Pie chart')

                model = YOLO(yolov8_model_path, device_option)
                placeholder = st.empty()

                col1, col2 = st.columns(2)
                id_class = {} # id와 class를 확보하기위한 dictionary 생성
                vehicle_counts = {'Motorcycle':0, 'Bicycle':0, 'Kickboard':0, 'Car':0, 'Heavy_Vehicle':0}
                # 이미지 엘리먼트 변수 생성
                frame_window_col1 = col1.empty()
                frame_window_col2 = col2.empty()
                frame_window_col3 = col1.empty()
                frame_window_col4 = col2.empty()
                frame_window_col5 = col1.empty()
                frame_window_col6 = col2.empty()
                frame_window_col7 = col1.empty()
                frame_window_col8 = col2.empty()

                for frame in get_frames_from_m3u8(st.session_state.video_url):
                    if stop_button:
                        st.session_state.clear()
                        st.stop()
                        break
                    
                    id_lst = []
                    if len(real_results) >= 2:
                        real_results = [real_results[0]]
                    if len(same_id) == segment_frame:
                        same_id=[]

                    results = model.track(frame, persist=True, tracker="./trackers/botsort.yaml")
                    real_results.append(copy.deepcopy(results[0].names))
                    annotated_frame = results[0].plot()
                    boxes_tensor = results[0].boxes
                    if len(boxes_tensor) > 0:
                        bbox_class = boxes_tensor.boxes[0][-1]
                        bbox_xyxy = boxes_tensor.xyxy.tolist()
                        bbox_cls = boxes_tensor.cls.tolist()
                        bbox_xywh = boxes_tensor.xywh.tolist()
                        if boxes_tensor.id != None:
                            bbox_id = boxes_tensor.id
                            current_id = {}
                            for i in range(len(bbox_id.tolist())):
                                current_id[bbox_id.tolist()[i]] = bbox_xyxy[i]
                            same_id.append([current_id])
                            if len(same_id) == segment_frame:
                                for i in range(len(list(same_id[0][0].keys()))):
                                    for j in range(len(list(same_id[segment_frame-1][0].keys()))):
                                        if list(same_id[0][0].keys())[i] == list(same_id[segment_frame-1][0].keys())[j]:
                                            id_lst.append(list(same_id[0][0].keys())[i])
                                for i in id_lst:
                                    first = same_id[0][0][i]
                                    second = same_id[segment_frame-1][0][i]        
                                    a1, a3 = line(first[1], first[0], second[1], second[0], annotated_frame)
                                    a2, a4 = line(first[3], first[0], second[3], second[0], annotated_frame)
                                    if first[1] - second[1] <= 15:
                                        final = [[0, 0], [0, 0], [0, 0], [0, 0]]
                                    elif first[1] < second[1]:
                                        final = [[int(first[0]), int(first[1])], [int(a3), int(a1)], [int(a4), int(a2)], [int(first[0]), int(first[3])]]
                                    elif first[1] > second[1]:
                                        final = [[int(a3), int(a1)], [int(first[0]), int(first[1])], [int(first[0]), int(first[3])], [int(a4), int(a2)]]
                                    
                                    annotated_frame = apply_alpha_to_color(annotated_frame, np.array(final), (192, 192, 192), 0.3)
                                    if 23 in bbox_cls:
                                        point_inside = cv2.pointPolygonTest(np.array(final), bbox_xywh[bbox_cls.index(23)][:2], measureDist=False)
                                        if point_inside >= 0:
                                            for i in range(len(results[0].names)):
                                                if i != 23 and i != 24:
                                                    results[0].names[i] = "Normal"
                                        elif point_inside < 0:
                                            for i in range(len(results[0].names)):
                                                results[0].names[i] = real_results[0][i]
                                
                    placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
                    if stop_button:
                        st.session_state.clear()
                        st.stop()
                        break
                    # Display the annotated frame
                    ids = results[0].boxes.id # id 확보
                    if ids is not None : # id가 존재할 때만 코드 실행
                        class_list = results[0].boxes.cls.cpu().numpy().astype(int)
                        ids_list = ids.cpu().numpy().astype(int)

                        for id, class_num in zip(ids_list, class_list) :
                            if id not in id_class : # 새로 등장한 id일 경우 key 생성
                                id_class[int(id)] = [int(class_num)]
                                
                                # 새로 등장한 id일 경우 vehicle count에 적용
                                if class_num >= 0 and class_num <= 6 :
                                    vehicle_counts['Motorcycle'] += 1
                                elif class_num >= 7 and class_num <= 14 :
                                    vehicle_counts['Bicycle'] += 1
                                elif class_num >= 15 and class_num <= 22 :
                                    vehicle_counts['Kickboard'] += 1
                                elif class_num == 26 :
                                    vehicle_counts['Car'] += 1
                                elif class_num == 27 :
                                    vehicle_counts['Heavy_Vehicle'] += 1
                                
                            else :
                                if int(class_num) in id_class[int(id)] :
                                    pass
                                else :
                                    id_class[int(id)].append(int(class_num))
                    if bar_on:
                        with col1:
                            if len(id_class) >= 1 :
                                    classes = []
                                    for class_list in id_class.values() :
                                        if 0 in class_list and len(class_list) >= 2 :
                                            classes.extend([x for x in class_list if x != 0])
                                        else :
                                            classes.extend(class_list)

                                        # 각 값의 빈도수 계산
                                        value_counts = Counter(classes)
                                        mc_counts = {key: value for key, value in value_counts.items() if key >= 0 and key <= 6}
                                        bc_counts = {key: value for key, value in value_counts.items() if key >= 7 and key <= 14}
                                        kb_counts = {key: value for key, value in value_counts.items() if key >= 15 and key <= 22}

                                        # motorcycle bar chart
                                        values = list(mc_counts.keys())
                                        counts = list(mc_counts.values())
                                        
                                        fig, ax = plt.subplots()
                                        plt.bar(values, counts)
                                        plt.xlabel('Class')
                                        plt.ylabel('Counts')
                                        plt.title('Motorcycle Value Counts')
                                        
                                        frame_window_col3.pyplot(fig)

                                        # bicycle bar chart
                                        values = list(bc_counts.keys())
                                        counts = list(bc_counts.values())
                                        
                                        fig, ax = plt.subplots()
                                        plt.bar(values, counts)
                                        plt.xlabel('Class')
                                        plt.ylabel('Counts')
                                        plt.title('Bicycle Value Counts')
                                        
                                        frame_window_col5.pyplot(fig)

                                        # kickboard bar chart
                                        values = list(kb_counts.keys())
                                        counts = list(kb_counts.values())
                                        
                                        fig, ax = plt.subplots()
                                        plt.bar(values, counts)
                                        plt.xlabel('Class')
                                        plt.ylabel('Counts')
                                        plt.title('Kickboard Value Counts')
                                        
                                        frame_window_col7.pyplot(fig)

                    # 두 번째 열 : pie chart
                    if pie_on:
                        with col2 :
                            if len(id_class) >= 1 :

                                # Vehicle 파이 차트 그리기
                                fig, ax = plt.subplots()
                                labels = [vehicle for vehicle in vehicle_counts.keys()]
                                sizes = [counts for counts in vehicle_counts.values()]
                                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                                ax.axis('equal')  # 원이 보존되도록 설정
                                plt.title('Vehicle Counts')

                                frame_window_col2.pyplot(fig)

                                # Motorcycle pie chart
                                fig, ax = plt.subplots()
                                labels = [vehicle for vehicle in mc_counts.keys()]
                                sizes = [counts for counts in mc_counts.values()]
                                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                                ax.axis('equal')  # 원이 보존되도록 설정
                                plt.title('Motorcycle Violation Ratio')

                                frame_window_col4.pyplot(fig)

                                # Bicycle pie chart
                                fig, ax = plt.subplots()
                                labels = [vehicle for vehicle in bc_counts.keys()]
                                sizes = [counts for counts in bc_counts.values()]
                                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                                ax.axis('equal')  # 원이 보존되도록 설정
                                plt.title('Bicycle Violation Ratio')

                                frame_window_col6.pyplot(fig)

                                # Kickboard pie chart
                                fig, ax = plt.subplots()
                                labels = [vehicle for vehicle in kb_counts.keys()]
                                sizes = [counts for counts in kb_counts.values()]
                                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                                ax.axis('equal')  # 원이 보존되도록 설정
                                plt.title('Kickboard Violation Ratio')

                                frame_window_col8.pyplot(fig)


            if 'stop_button' in globals() and stop_button:
                st.session_state.clear()

            while True:
                ret, frame = video_capture.read()
                if not ret:
                    st.warning("Error retrieving the video frame.")
                    break
                frame_window.image(frame, channels="BGR")

        
elif selected2 == "info":
    st.subheader("모델 학습 데이터",divider='rainbow')
    st.image('/Users/bumsoojoe/Desktop/팀프로젝트2_교통법규/Traffic_Police/website/static/image_data.png',use_column_width=True)
    st.markdown("""
    **< Image Data >**
    - 출처 : AI HUB 개인형 이동장치 안전 데이터
    - 총 300시간에 대응하는 60만장의 도로 교통 사진
    - 이미지 크기 : 1920 * 1080
    - 종류 : CCTV & 연출
    - 시간 : 낮과 밤, 맑음과 우천
""")
                
    col1, col2 = st.columns(2)
    col1.image('/Users/bumsoojoe/Desktop/팀프로젝트2_교통법규/Traffic_Police/website/static/label_1.png',use_column_width=True)
    col2.image('/Users/bumsoojoe/Desktop/팀프로젝트2_교통법규/Traffic_Police/website/static/label_2.png',use_column_width=True)
    st.markdown("""
    **< Label >**
    - 형식 : JSON 파일
    - Meta Data : 각 이미지에 대한 정보 (촬영 날짜, 이미지 크기 등)
    - Polygon : 횡단보도, 인도 등 교통 환경을 다각형 표시한 라벨링·좌표
    - Bounding Box : 개인형 이동장치를 사각형 표시한 라벨링·좌표 
                

""")
    st.markdown("""**< Class >** """)
    col1, col2, col3 = st.columns(3)
    col1.markdown("""
    **오토바이 탑승자**
    - 정상 주행
    - 보행자도로 통행 위반
    - 안전모 미착용 위반
    - 무단 횡단 위반
    - 신호 위반
    - 정지선 위반
    - 횡단보도 주행 위반""")
    col2.markdown("""
    **자전거 탑승자**
    - 정상 주행
    - 자전거 운반
    - 보행자도로 통행 위반
    - 안전모 미착용 위반
    - 무단 횡단 위반
    - 신호 위반
    - 정지선 위반
    - 횡단보도 주행 위반""")
    col3.markdown("""
    **킥보드 탑승자**
    - 정상 주행
    - 킥보드 운반
    - 보행자도로 통행 위반
    - 안전모 미착용 위반
    - 무단 횡단 위반
    - 신호 위반
    - 정지선 위반
    - 횡단보도 주행 위반""")
    st.header("YOLO (You Only Look Once)",divider='rainbow')
    st.write("""
    YOLO는 실시간 객체 탐지 알고리즘 중 하나로, 이미지 전체를 한 번만 보고 객체의 종류와 위치를 파악할 수 있습니다. 
    YOLO는 전통적인 객체 탐지 방식과는 달리 이미지를 여러 번 스캔하지 않기 때문에 빠른 성능을 자랑합니다.
    """)
    st.subheader("YOLO의 핵심 원리 및 특징")
    st.markdown("""
    - **단일 네트워크**: YOLO는 단일 컨볼루션 네트워크를 사용하여 객체 탐지를 수행합니다. 
      이 네트워크는 이미지 전체를 한 번만 보고, 객체의 위치와 분류를 동시에 예측합니다.
    - **그리드 시스템**: 이미지는 \( S \times S \) 그리드로 분할됩니다. 각 그리드 셀은 해당 영역에 
      객체가 존재하는 확률과 객체의 경계 상자 (bounding box) 정보를 예측합니다.
    - **다중 경계 상자 예측**: 각 그리드 셀은 여러 개의 경계 상자를 예측할 수 있습니다. 
      이는 다양한 모양과 크기의 객체를 탐지하는 데 도움이 됩니다.
    - **경계 상자 특성**: 각 경계 상자는 다음과 같은 정보를 포함합니다: 
        - 중심점의 x, y 좌표
        - 너비와 높이
        - 객체가 존재하는 확률
    - **클래스 확률**: 각 그리드 셀은 해당 영역에 있는 객체의 클래스 확률도 예측합니다. 
      클래스 확률은 전체 이미지에 대한 것이 아니라 해당 그리드 셀에 대한 것입니다.
    - **손실 함수**: YOLO는 위치, 크기, 객체 존재 확률, 클래스 확률 등을 동시에 예측하기 때문에 
      복잡한 손실 함수를 사용합니다. 이를 통해 모델은 다양한 예측 타스크를 동시에 학습할 수 있습니다.
    - **실시간 성능**: YOLO는 실시간 객체 탐지에 적합하게 설계되었습니다. 따라서 비디오 스트리밍과 
      같은 실시간 애플리케이션에서 효과적으로 사용할 수 있습니다.
    """)
    st.write("""
    YOLO는 시간이 지나면서 여러 버전으로 발전했습니다. YOLOv1, YOLOv2 (YOLO9000), YOLOv3, YOLOv4 및 
    YOLOv5와 같은 여러 버전이 있으며, 각 버전은 성능 향상과 함께 여러 개선 사항을 도입했습니다.
    """)
    st.write("""
    요약하면, YOLO는 실시간 객체 탐지를 위한 강력하고 빠른 알고리즘입니다. 이미지를 한 번만 보고 객체의 
    위치와 분류를 동시에 예측하는 능력으로 인해 많은 관심을 받았습니다.
    """)
