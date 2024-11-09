import cv2
import mediapipe as mp
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
import time
import math
import platform
import json
import undetected_chromedriver as uc

# Mediapipe ve OpenCV ile el hareketlerini algılama
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# WebDriver ayarları (Brave kontrolü için)
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

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5) as hands:
    setup_browser()
    is_play_pause_triggered = False
    is_like_triggered = False
    is_dislike_triggered = False
    prev_angle = None
    volume_control_last_triggered = 0
    volume_sensitivity = 0.05  # Ses değişim hassasiyeti

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

        current_time = time.time()

        # El hareketlerini tespit et
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Yeni bir hareket: Thumbs Up işareti (Baş parmak yukarıda ve diğer parmaklar kapalı)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

                thumb_coords = np.array([thumb_tip.x * image.shape[1], thumb_tip.y * image.shape[0]])
                index_coords = np.array([index_mcp.x * image.shape[1], index_mcp.y * image.shape[0]])
                distance_thumb_index = np.linalg.norm(thumb_coords - index_coords)

                # Like durumu kontrolü (Thumbs Up hareketi)
                if distance_thumb_index < 40 and not is_like_triggered:
                    try:
                        like_button = driver.find_element('css selector', 'button[title="I like this"]')
                        if like_button.get_attribute('aria-pressed') == 'false':
                            like_button.click()
                            print('Videoya like atıldı.')
                        else:
                            like_button.click()
                            print('Videodan like kaldırıldı.')
                        is_like_triggered = True
                    except Exception as e:
                        print(f'Like düğmesi bulunamadı: {e}')

                # Like tetiklemesi sıfırlama
                if distance_thumb_index > 60:
                    is_like_triggered = False

                # Dislike durumu kontrolü (Thumbs Down hareketi)
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                pinky_coords = np.array([pinky_tip.x * image.shape[1], pinky_tip.y * image.shape[0]])
                distance_thumb_pinky = np.linalg.norm(thumb_coords - pinky_coords)

                if distance_thumb_pinky < 40 and not is_dislike_triggered:
                    try:
                        dislike_button = driver.find_element('css selector', 'button[title="I dislike this"]')
                        if dislike_button.get_attribute('aria-pressed') == 'false':
                            dislike_button.click()
                            print('Videoya dislike atıldı.')
                        else:
                            dislike_button.click()
                            print('Videodan dislike kaldırıldı.')
                        is_dislike_triggered = True
                    except Exception as e:
                        print(f'Dislike düğmesi bulunamadı: {e}')

                # Dislike tetiklemesi sıfırlama
                if distance_thumb_pinky > 60:
                    is_dislike_triggered = False

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Yeni bir hareket: Baş parmak ve küçük parmağı birleştirme ("Pinky Touch" işareti)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                thumb_coords = np.array([thumb_tip.x * image.shape[1], thumb_tip.y * image.shape[0]])
                pinky_coords = np.array([pinky_tip.x * image.shape[1], pinky_tip.y * image.shape[0]])
                distance_thumb_pinky = np.linalg.norm(thumb_coords - pinky_coords)

                # Eğer baş parmak ve küçük parmak birleştirilmişse ve durdur/oynat henüz tetiklenmemişse
                if distance_thumb_pinky < 40 and not is_play_pause_triggered:
                    try:
                        play_pause_button = driver.find_element('css selector', '.ytp-play-button')
                        print('Oynat/Duraklat düğmesi bulundu.')
                        play_pause_button.click()
                        is_play_pause_triggered = True
                    except Exception as e:
                        print(f"Oynat/Duraklat düğmesi bulunamadı: {e}")

                # Eğer baş parmak ve küçük parmak tekrar açılmışsa tetiklemeyi sıfırla
                if distance_thumb_pinky > 60:
                    is_play_pause_triggered = False

                # Ses artırma/azaltma hareketi (İşaret parmağıyla daire çizme)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                index_tip_coords = np.array([index_finger_tip.x * image.shape[1], index_finger_tip.y * image.shape[0]])
                index_mcp_coords = np.array([index_finger_mcp.x * image.shape[1], index_finger_mcp.y * image.shape[0]])

                # İşaret parmağının hareket yönünü analiz ederek daire çizme hareketini kontrol etme
                if prev_angle is not None:
                    angle = math.atan2(index_tip_coords[1] - index_mcp_coords[1], index_tip_coords[0] - index_mcp_coords[0])
                    angle_diff = (angle - prev_angle) * 180.0 / math.pi

                    if abs(angle_diff) > 10:
                        try:
                            # Ses artırma veya azaltma
                            if angle_diff > 0:
                                driver.execute_script(f"document.querySelector('video').volume = Math.min(1, document.querySelector('video').volume + {volume_sensitivity})")
                                print('Ses artırma işlemi başarılı.')
                            elif angle_diff < 0:
                                driver.execute_script(f"document.querySelector('video').volume = Math.max(0, document.querySelector('video').volume - {volume_sensitivity})")
                                print('Ses azaltma işlemi başarılı.')
                            volume_control_last_triggered = current_time
                        except Exception as e:
                            print(f"Ses kontrol işlemi başarısız: {e}")

                    prev_angle = angle
                else:
                    prev_angle = math.atan2(index_tip_coords[1] - index_mcp_coords[1], index_tip_coords[0] - index_mcp_coords[0])

        cv2.imshow('Lazy YouTube Controller', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
driver.quit()
cv2.destroyAllWindows()
