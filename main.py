from collections import deque
import mediapipe as mp
import numpy as np
import pyautogui
import torch
import time
import platform
import json
import dlib
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import undetected_chromedriver as uc

from models.model_architecture import model
from utils import *
from imutils import face_utils

# Mediapipe ve OpenCV ile el hareketlerini algılama
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# WebDriver ayarları (Brave kontrolü için)
# BU ALANI DEĞİŞTİRMEYİN
options = webdriver.ChromeOptions()
options.binary_location = '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser'

# Sisteme göre Brave tarayıcısının yolunu belirleyin
system_platform = platform.system()

# JSON dosyasını yükleyin
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

try:
    if system_platform == 'Darwin':
        driver = uc.Chrome(options=options)
    elif system_platform == 'Windows':
        driver = uc.Chrome(service=Service(config['chromedriver_path']), options=options)
    else:
        print('Bu sistem desteklenmiyor.')
        exit()
    print('Brave tarayıcısı başarıyla başlatıldı.')
except Exception as e:
    print(f'Tarayıcı başlatılamadı: {e}')
    exit()

def setup_browser():
    try:
        driver.get('https://www.youtube.com')
        time.sleep(5)  # Sayfa yüklenirken beklemek için
        print('YouTube başarıyla açıldı.')
    except Exception as e:
        print(f'YouTube açılamadı: {e}')
        exit()
# --------------------------------------------BURAYA KADAR DEĞİŞTİRMEYİN--------------------------------------------


#Asıl iş burada başlıyor

mode = 0
CSV_PATH = 'data/gestures.csv'

# Camera settings
WIDTH = 1028//2
HEIGHT = 720//2

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)


# important keypoints (wrist + tips coordinates)
# for training the model
TRAINING_KEYPOINTS = [keypoint for keypoint in range(0, 21, 4)]


# Mouse mouvement stabilization
SMOOTH_FACTOR = 6
PLOCX, PLOCY = 0, 0 # previous x, y locations
CLOX, CLOXY = 0, 0 # current x, y locations

# Hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75)

# Hand landmarks drawing
mp_drawing = mp.solutions.drawing_utils

# Load saved model for hand gesture recognition
GESTURE_RECOGNIZER_PATH = 'models/model.pth'
model.load_state_dict(torch.load(GESTURE_RECOGNIZER_PATH))

# Load Label
LABEL_PATH = 'data/label.csv'
labels = pd.read_csv(LABEL_PATH, header=None).values.flatten().tolist()

# confidence threshold(required to translate gestures into commands)
CONF_THRESH = 0.9

# history to track the n last detected commands
GESTURE_HISTORY = deque([])

# general counter (for volum up/down; forward/backward)
GEN_COUNTER = 0

# Face detection (absence feature)
mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.75)

IS_ABSENT = None # only for testing purposes...change this by video status (paused or playing)
ABSENCE_COUNTER = 0
ABSENCE_COUNTER_THRESH = 20

# Frontal face + face landmarks (sleepness feature)
SHAPE_PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
lStart, lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
rStart, rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


IS_SLEEPING = None # only for testing purposes...change this by video status (paused or playing)
SLEEP_COUNTER = 0
SLEEP_COUNTER_THRESH = 20
EAR_THRESH = 0.21 
EAR_HISTORY = deque([])


while True:
    key = cv.waitKey(1) 
    if key == ord('q'):
        break

    
    # choose mode (normal or recording)
    mode = select_mode(key, mode=mode)

    # class id for recording
    class_id = get_class_id(key)

    # read camera
    has_frame, frame = cap.read()
    if not has_frame:
        break

    # horizontal flip and color conversion for mediapipe
    frame = cv.flip(frame, 1)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Detection and mouse zones
    det_zone, m_zone = det_mouse_zones(frame)



    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # get landmarks coordinates
            coordinates_list = calc_landmark_coordinates(frame_rgb, hand_landmarks)
            important_points = [coordinates_list[i] for i in TRAINING_KEYPOINTS]


            # Conversion to relative coordinates and normalized coordinates
            preprocessed = pre_process_landmark(important_points)
            
            # compute the needed distances to add to coordinate features
            d0 = calc_distance(coordinates_list[0], coordinates_list[5])
            pts_for_distances = [coordinates_list[i] for i in [4, 8, 12]]
            distances = normalize_distances(d0, get_all_distances(pts_for_distances)) 
            

            # Write to the csv file "keypoint.csv"(if mode == 1)
            # logging_csv(class_id, mode, preprocessed)
            features = np.concatenate([preprocessed, distances])
            draw_info(frame, mode, class_id)
            logging_csv(class_id, mode, features, CSV_PATH)

            
            # inference
            conf, pred = predict(features, model)
            gesture = labels[pred]

####################################################### YOUTUBE PLAYER CONTROL ###########################################################                       
                
            # check if middle finger mcp is inside the detection zone and prediction confidence is higher than a given threshold
            if cv.pointPolygonTest(det_zone, coordinates_list[9], False) == 1 and conf >= CONF_THRESH: 

                # track command history
                gest_hist = track_history(GESTURE_HISTORY, gesture)

                if len(gest_hist) >= 2:
                    before_last = gest_hist[len(gest_hist) - 2]
                else:
                    before_last = gest_hist[0]

            ############### mouse gestures ##################
                if gesture == 'Move_mouse':
                    x, y = mouse_zone_to_screen(coordinates_list[9], m_zone)
                    
                    # smoothe mouse movements
                    CLOX = PLOCX + (x - PLOCX) / SMOOTH_FACTOR
                    CLOXY = PLOCY + (y - PLOCY) / SMOOTH_FACTOR
                    pyautogui.moveTo(CLOX, CLOXY)
                    PLOCX, PLOCY = CLOX, CLOXY

                if gesture == 'Right_click' and before_last != 'Right_click':
                    pyautogui.rightClick()

                if gesture == 'Left_click' and before_last != 'Left_click':
                    pyautogui.click()   


            ############### Other gestures ################## 
                if gesture == 'Play_Pause' and before_last != 'Play_Pause':
                    pyautogui.press('space')
                
                elif gesture == 'Vol_up_gen':
                    pyautogui.press('volumeup')

                elif gesture == 'Vol_down_gen':
                    pyautogui.press('volumedown')

                elif gesture == 'Vol_up_ytb':
                    GEN_COUNTER += 1
                    if GEN_COUNTER % 4 == 0:
                        pyautogui.press('up')

                elif gesture == 'Vol_down_ytb':
                    GEN_COUNTER += 1
                    if GEN_COUNTER % 4 == 0:
                        pyautogui.press('down')

                elif gesture == 'Forward':
                    GEN_COUNTER += 1
                    if GEN_COUNTER % 4 == 0:
                        pyautogui.press('right')
                
                elif gesture == 'Backward':
                    GEN_COUNTER += 1
                    if GEN_COUNTER % 4 == 0:
                        pyautogui.press('left')
                
                elif gesture == 'fullscreen' and before_last != 'fullscreen':
                    pyautogui.press('f')
                
                elif gesture == 'Cap_Subt' and before_last != 'Cap_Subt':
                    pyautogui.press('c')

                elif gesture == 'Neutral':
                    GEN_COUNTER = 0 

                # show detected gesture
                cv.putText(frame, f'{gesture} | {conf: .2f}', (int(WIDTH*0.05), int(HEIGHT*0.07)),
                    cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv.LINE_AA)

cap.release()
cv.destroyAllWindows()                
