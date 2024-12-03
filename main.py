import platform
import os

# İşletim sistemi kontrolü
system_platform = platform.system()

if system_platform == "Windows":
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Ortak kütüphaneler
from collections import deque
import mediapipe as mp
import numpy as np
import torch
import pandas as pd
import cv2 as cv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
import pyautogui
import time

from models.model_architecture import model
from utils import *

# Global constants and configurations
WIDTH, HEIGHT = 1028, 720
CSV_PATH = 'data/gestures.csv'
GESTURE_RECOGNIZER_PATH = 'models/model.pth'
LABEL_PATH = 'data/label.csv'
CONF_THRESH = 0.9
TRAINING_KEYPOINTS = [keypoint for keypoint in range(0, 21, 4)]
SMOOTH_FACTOR = 6
ABSENCE_COUNTER_THRESH = 20

# State variables
PLOCX, PLOCY = 0, 0
CLOX, CLOXY = 0, 0
GEN_COUNTER = 0
GESTURE_HISTORY = deque([])
ABSENCE_COUNTER = 0
IS_ABSENT = None  # Placeholder for video status

# Initialize Mediapipe and other components
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75)
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.75)

################################################## BU KISIM TARAYICI AYARLARIDIR ###############################
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

def load_config(file_path):
    """
    JSON dosyasını yükler ve içeriğini döndürür.
    """
    try:
        with open(file_path, 'r') as config_file:
            config = json.load(config_file)
        print("Config dosyası başarıyla yüklendi.")
        return config
    except FileNotFoundError:
        print(f"Hata: {file_path} dosyası bulunamadı.")
        exit()
    except json.JSONDecodeError as e:
        print(f"JSON dosyası yüklenirken hata oluştu: {e}")
        exit()

def get_browser_choice():
    """
    Kullanıcıdan tarayıcı seçimi alır.
    """
    print("Kullanmak istediğiniz tarayıcıyı seçin:\n1 - Chrome\n2 - Brave")
    choice = input("Seçiminiz: ")
    if choice == "1":
        return "chrome"
    elif choice == "2":
        return "brave"
    else:
        print("Geçersiz seçim. Program sonlandırılıyor.")
        exit()

def start_browser(config, browser_choice):
    """
    Seçilen tarayıcıyı başlatır.
    """
    options = webdriver.ChromeOptions()
    if browser_choice == "chrome":
        if not config['chrome_path']:
            print("Chrome path config.json'da belirtilmemiş.")
            exit()
        options.binary_location = config['chrome_path']
    elif browser_choice == "brave":
        if not config['brave_browser_path']:
            print("Brave path config.json'da belirtilmemiş.")
            exit()
        options.binary_location = config['brave_browser_path']

    try:
        service = Service(config['chromedriver_path'])  # Chromedriver yolunu config.json'dan alıyoruz
        driver = webdriver.Chrome(service=service, options=options)
        print(f'{browser_choice.capitalize()} başarıyla başlatıldı.')
        return driver
    except Exception as e:
        print(f'Tarayıcı başlatılamadı: {e}')
        exit()

def open_url(driver, url):
    """
    Verilen URL'yi açar.
    """
    try:
        driver.get(url)
        time.sleep(5)  # Sayfa yüklenirken beklemek için
        print(f'{url} başarıyla açıldı.')
    except Exception as e:
        print(f'{url} açılamadı: {e}')
        exit()

##############################################################################################################

def handle_volume(gesture, volume_sensitivity=0.05):
    """
    Ses kontrolü işlemleri için platforma göre davranış.
    """
    if system_platform == "Windows":
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            current_volume = volume.GetMasterVolumeLevelScalar()

            if gesture == "Vol_up_gen":
                volume.SetMasterVolumeLevelScalar(min(1.0, current_volume + volume_sensitivity), None)
            elif gesture == "Vol_down_gen":
                volume.SetMasterVolumeLevelScalar(max(0.0, current_volume - volume_sensitivity), None)
        except Exception as e:
            print(f"Windows ses kontrolünde hata: {e}")
    elif system_platform == "Darwin":  # MacOS
        try:
            if gesture == "Vol_up_gen":
                os.system("osascript -e 'set volume output volume ((output volume of (get volume settings)) + 5)'")
            elif gesture == "Vol_down_gen":
                os.system("osascript -e 'set volume output volume ((output volume of (get volume settings)) - 5)'")
        except Exception as e:
            print(f"MacOS ses kontrolünde hata: {e}")
    else:
        print(f"{system_platform} platformu için ses kontrolü desteklenmiyor.")

# Load the model and labels
def load_model_and_labels():
    model.load_state_dict(torch.load(GESTURE_RECOGNIZER_PATH))
    labels = pd.read_csv(LABEL_PATH, header=None).values.flatten().tolist()
    return labels

labels = load_model_and_labels()

# Setup video capture
def setup_camera():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    return cap

# Process hand landmarks
def process_hand_landmarks(frame_rgb, frame, mode, class_id, driver):
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            coordinates_list = calc_landmark_coordinates(frame_rgb, hand_landmarks)
            important_points = [coordinates_list[i] for i in TRAINING_KEYPOINTS]

            preprocessed = pre_process_landmark(important_points)
            d0 = calc_distance(coordinates_list[0], coordinates_list[5])
            pts_for_distances = [coordinates_list[i] for i in [4, 8, 12]]
            distances = normalize_distances(d0, get_all_distances(pts_for_distances))

            features = np.concatenate([preprocessed, distances])
            draw_info(frame, mode, class_id)
            logging_csv(class_id, mode, features, CSV_PATH)

            conf, pred = predict(features, model)
            gesture = labels[pred]
            if conf > CONF_THRESH:
                handle_gesture(driver, gesture)
                cv.putText(frame, f"Gesture: {gesture} ({conf:.2f})",
                           (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display and update the frame
def display_frame(cap, driver, mode):
    global frame
    global frame_rgb
    while True:

        key = cv.waitKey(1)
        if key == ord('q'):
            break

        mode = select_mode(key, mode=mode)
        class_id = get_class_id(key)
        has_frame, frame = cap.read()
        if not has_frame:
            break

        frame = cv.flip(frame, 1)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        process_hand_landmarks(frame_rgb, frame, mode, class_id, driver)
        cv.imshow("Hand Gesture Recognition", frame)

# Main function
def main():
    # Tarayıcı ayarları ve YouTube açılışı
    try:
        print("Tarayıcı ayarları başlatılıyor...")
        config_path = 'config.json'
        config = load_config(config_path)  # Config dosyasını yükle
        browser_choice = get_browser_choice()  # Tarayıcı seçimini al
        driver = start_browser(config, browser_choice)  # Seçilen tarayıcıyı başlat
        open_url(driver, 'https://www.youtube.com')  # YouTube'u aç

    except Exception as e:
        print(f"Tarayıcı işlemleri sırasında hata oluştu: {e}")
        driver = None  # Driver başlatılamadıysa None yap

    # Kamera kurulumu ve video işleme
    try:
        print("Kamera kurulumu başlatılıyor...")
        cap = setup_camera()
        print("Video akışı başlatılıyor...")
        display_frame(cap, driver, mode=0)
    except Exception as e:
        print(f"Kamera veya video işleme sırasında hata oluştu: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv.destroyAllWindows()
        print("Tüm işlemler tamamlandı ve kaynaklar serbest bırakıldı.")

######################################## YouTube Kontrol ###############################################################

# Ekran boyutunu al
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

def move_mouse_with_coordinates(frame_rgb):
    global PLOCX, PLOCY
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            coordinates_list = calc_landmark_coordinates(frame_rgb, hand_landmarks)
            important_points = [coordinates_list[i] for i in TRAINING_KEYPOINTS]

            preprocessed = pre_process_landmark(important_points)
            d0 = calc_distance(coordinates_list[0], coordinates_list[5])
            pts_for_distances = [coordinates_list[i] for i in [4, 8, 12]]
            distances = normalize_distances(d0, get_all_distances(pts_for_distances))

            features = np.concatenate([preprocessed, distances])

            # Orta parmak ucu için normalize edilmiş koordinatları al
            x_norm, y_norm = coordinates_list[9]  # Orta parmak ucu
            x_norm, y_norm = x_norm / WIDTH, y_norm / HEIGHT  # Çerçeveye göre normalizasyon

            # Ekran boyutuna dönüştür
            x_screen = x_norm * SCREEN_WIDTH
            y_screen = y_norm * SCREEN_HEIGHT

            # Hareketi yumuşat
            CLOX = PLOCX + (x_screen - PLOCX) / SMOOTH_FACTOR
            CLOXY = PLOCY + (y_screen - PLOCY) / SMOOTH_FACTOR

            # Fareyi hareket ettir
            pyautogui.moveTo(CLOX, CLOXY)
            PLOCX, PLOCY = CLOX, CLOXY

LAST_GESTURE_TIME = 0
GESTURE_DELAY = 1.0  # 1 saniye tampon süresi

def calc_landmark_coordinates(frame_rgb, hand_landmarks):
    """
    Mediapipe'den dönen el işaretlerini (landmarks) çerçeve boyutuna göre 
    piksel koordinatlarına dönüştürür.
    """
    h, w, _ = frame_rgb.shape
    coordinates = [(int(landmark.x * w), int(landmark.y * h)) for landmark in hand_landmarks.landmark]
    return coordinates

def handle_gesture(driver, gesture):
    global LAST_GESTURE_TIME
    global frame_rgb

    current_time = time.time()  # Geçerli zaman

    # Move_mouse jesti için zaman kontrolü yapılmaz
    if gesture == 'Move_mouse':
        try:
            move_mouse_with_coordinates(frame_rgb)
        except Exception as e:
            print(f"'Move_mouse' jestinde hata: {e}")
        return

    # Diğer jestler için zaman kontrolü yapılır
    if current_time - LAST_GESTURE_TIME < GESTURE_DELAY:
        return  # Harekete izin verilmez

    LAST_GESTURE_TIME = current_time  # Son hareketin zamanını güncelle
    volume_sensitivity = 0.05

    try:
        if gesture == "Play_Pause":
            # Oynat/Duraklat işlemi
            driver.find_element("css selector", "button.ytp-play-button").click()
        elif gesture == "Vol_up_ytb":
            # YouTube sesini artır
            driver.execute_script(
                f"document.querySelector('video').volume = Math.min(1, document.querySelector('video').volume + {volume_sensitivity})"
            )
        elif gesture == "Vol_down_ytb":
            # YouTube sesini azalt
            driver.execute_script(
                f"document.querySelector('video').volume = Math.max(0, document.querySelector('video').volume - {volume_sensitivity})"
            )
        elif gesture in ["Vol_up_gen", "Vol_down_gen"]:
            # Genel ses kontrolü
            handle_volume(gesture, volume_sensitivity)
        elif gesture == "fullscreen":
            # Tam ekran geçişi yap
            driver.find_element("css selector", "button.ytp-fullscreen-button").click()
        elif gesture == "Forward":
            # Videoyu ileri sar
            driver.execute_script("document.querySelector('video').currentTime += 10")
        elif gesture == "Backward":
            # Videoyu geri sar
            driver.execute_script("document.querySelector('video').currentTime -= 10")
        elif gesture == "Cap_Subt":
            # Altyazıları aç/kapat
            driver.find_element("css selector", "button.ytp-subtitles-button").click()
        elif gesture == 'Right_click':
            pyautogui.rightClick()
        elif gesture == 'Left_click':
            pyautogui.click()
        else:
            print(f"'{gesture}' için tanımlı bir işlem bulunamadı.")
    except Exception as e:
        print(f"Gesture işleminde hata: {e}")

if __name__ == "__main__":
    main()
