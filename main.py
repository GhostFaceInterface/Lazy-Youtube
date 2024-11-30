from collections import deque
import mediapipe as mp
import numpy as np
import torch
import pandas as pd
import cv2 as cv
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from pynput.mouse import Controller, Button
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
# Model and utility imports
from models.model_architecture import model
from utils import *
import time
# Global constants and configurations
WIDTH, HEIGHT = 1028 // 2, 720 // 2
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

##################################################BU KISIM TARAYICI AYARLARIDIR###############################
import platform
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

def configure_chrome_options():
    """
    İşletim sistemine göre Chrome tarayıcı seçeneklerini yapılandırır.
    """
    system_platform = platform.system()
    options = webdriver.ChromeOptions()

    if system_platform == 'Darwin':  # macOS
        options.binary_location = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
    elif system_platform == 'Windows':  # Windows
        options.binary_location = "C:\\Users\\zahit\\Downloads\\chrome-win64\\chrome-win64\\chrome.exe"
    else:
        print("Bu sistem desteklenmiyor.")
        exit()

    print(f"{system_platform} için Chrome seçenekleri başarıyla yapılandırıldı.")
    return options

def start_browser(config, options):
    """
    WebDriver hizmetini başlatır ve Chrome tarayıcısını döndürür.
    """
    try:
        service = Service(config['chromedriver_path'])  # Chromedriver yolunu config.json'dan alıyoruz
        driver = webdriver.Chrome(service=service, options=options)  
        print('Google Chrome başarıyla başlatıldı.')
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
def process_hand_landmarks(frame_rgb, frame, mode, class_id,driver):
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
def display_frame(cap,driver,mode):
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
        config = load_config(config_path)
        options = configure_chrome_options()
        driver = start_browser(config, options)
        open_url(driver, 'https://www.youtube.com')
    except Exception as e:
        print(f"Tarayıcı işlemleri sırasında hata oluştu: {e}")
    
    # Kamera kurulumu ve video işleme
    try:
        print("Kamera kurulumu başlatılıyor...")
        cap = setup_camera()
        print("Video akışı başlatılıyor...")
        display_frame(cap,driver,mode=0)
    except Exception as e:
        print(f"Kamera veya video işleme sırasında hata oluştu: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv.destroyAllWindows()
        print("Tüm işlemler tamamlandı ve kaynaklar serbest bırakıldı.")

########################################Youtube Kontrol########################################################################

LAST_GESTURE_TIME = 0
GESTURE_DELAY = 1.0  # 1 saniye tampon süresi

def handle_gesture(driver, gesture):
    global LAST_GESTURE_TIME

    current_time = time.time()  # Geçerli zaman
    # Eğer son hareketle geçen süre tampon süresinden kısa ise, işlem yapılmaz
    if current_time - LAST_GESTURE_TIME < GESTURE_DELAY:
        return  # Harekete izin verilmez

    LAST_GESTURE_TIME = current_time  # Son hareketin zamanını güncelle
    volume_sensitivity=0.05
    try:
        if gesture == "Play_Pause":
            # Oynat/Duraklat işlemi
            driver.find_element("css selector", "button.ytp-play-button").click()
        elif gesture == "Vol_up_ytb":
            # Increase volume
           driver.execute_script(f"document.querySelector('video').volume = Math.min(1, document.querySelector('video').volume + {volume_sensitivity})")
        elif gesture == "Vol_down_ytb":
            # Decrease volume
            driver.execute_script(f"document.querySelector('video').volume = Math.max(0, document.querySelector('video').volume - {volume_sensitivity})")
        elif gesture == "Vol_up_gen":
            # Increase system volume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            current_volume = volume.GetMasterVolumeLevelScalar()
            volume.SetMasterVolumeLevelScalar(min(1.0, current_volume + volume_sensitivity), None)
        elif gesture == "Vol_down_gen":
            # Decrease system volume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            current_volume = volume.GetMasterVolumeLevelScalar()
            volume.SetMasterVolumeLevelScalar(max(0.0, current_volume - volume_sensitivity), None)
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
        else:
            print(f"'{gesture}' için tanımlı bir işlem bulunamadı.")
    except Exception as e:
        print(f"Gesture işleminde hata: {e}")


if __name__ == "__main__":
    main()
