from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import torch
import time
import platform
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import undetected_chromedriver as uc

from models.model_architecture import MLP
from utils import (
    pre_process_landmark,
    predict,
    get_gesture_name,
    calc_landmark_coordinates,
    draw_info,
    track_history,
    det_mouse_zones,
    mouse_zone_to_screen
)

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

# Hareket-Sınıf İşlev Eşleştirmesi
def handle_gesture_action(gesture_name):
    try:
        if gesture_name == "Forward":
            driver.execute_script("document.querySelector('video').currentTime += 10;")
            print("İleri sarma işlemi gerçekleştirildi.")
        elif gesture_name == "Backward":
            driver.execute_script("document.querySelector('video').currentTime -= 10;")
            print("Geri sarma işlemi gerçekleştirildi.")
        elif gesture_name == "Vol_up_ytb":
            driver.execute_script("document.querySelector('video').volume = Math.min(1, document.querySelector('video').volume + 0.1);")
            print("YouTube ses seviyesi artırıldı.")
        elif gesture_name == "Vol_down_ytb":
            driver.execute_script("document.querySelector('video').volume = Math.max(0, document.querySelector('video').volume - 0.1);")
            print("YouTube ses seviyesi azaltıldı.")
        elif gesture_name == "FullScreen":
            play_pause_button = driver.find_element('css selector', '.ytp-play-button')
            play_pause_button.click()
            print("Oynat/Duraklat tuşu tetiklendi.")
        elif gesture_name == "Play_Pause":
            like_button = driver.find_element('css selector', 'button[aria-label="Beğen"]')
            like_button.click()
            print("Beğen tuşu tıklandı.")
    except Exception as e:
        print(f"Komut işlenirken bir hata oluştu: {e}")

# Modeli Yükleme
model = MLP(n_features=13, n_classes=13, hidden_size=[32, 16])
model.load_state_dict(torch.load("models/model.pth"))
model.eval()

cap = cv2.VideoCapture(0)
history = deque(maxlen=3)  # Hareket geçmişini izlemek için bir deque

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5) as hands:
    setup_browser()
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print('Web kamerasından görüntü alınamadı.')
            break

        # BGR'yi RGB'ye dönüştür
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # El hareketlerini tespit et
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Landmark koordinatlarını çıkar ve normalize et
                landmark_coordinates = calc_landmark_coordinates(image, hand_landmarks)
                features = pre_process_landmark(landmark_coordinates)

                # Modelden tahmin al
                confidence, prediction = predict(features, model)
                gesture_name = get_gesture_name(prediction)
                print(f"Tahmin edilen hareket: {gesture_name} (Sınıf {prediction}), Güven: {confidence:.2f}")

                # Hareket geçmişini takip et
                track_history(history, gesture_name)

                # Komutları yürüt
                if confidence > 0.8:  # Güven eşiği
                    handle_gesture_action(gesture_name)

        # OpenCV penceresi
        cv2.imshow('El Hareketleri Algılama', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
driver.quit()
cv2.destroyAllWindows()
