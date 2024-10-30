import cv2
import mediapipe as mp
import numpy as np

# Mediapipe'in el tespiti için kullanılan modülü yükleme
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Web kamerası kullanılarak video yakalama
cap = cv2.VideoCapture(0)

# Mediapipe Hands sınıfını kullanarak el tespiti başlatma
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Web kamerasından görüntü alınamadı.")
            break

        # BGR görüntüyü RGB'ye dönüştürme
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # El tespiti
        results = hands.process(image)

        # Geriye BGR'ye dönüştürme ve çizim işlemleri için görüntüyü yazılabilir yapma
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Eğer el tespit edildiyse
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Elde edilen anahtar noktaları çizin
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Parmak uçları üzerinde işlem yapma
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
                index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                ring_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

                # Koordinatları hesaplama
                thumb_tip_coords = np.array([thumb_tip.x * image.shape[1], thumb_tip.y * image.shape[0]])
                index_finger_tip_coords = np.array([index_finger_tip.x * image.shape[1], index_finger_tip.y * image.shape[0]])
                middle_finger_tip_coords = np.array([middle_finger_tip.x * image.shape[1], middle_finger_tip.y * image.shape[0]])
                ring_finger_tip_coords = np.array([ring_finger_tip.x * image.shape[1], ring_finger_tip.y * image.shape[0]])
                pinky_tip_coords = np.array([pinky_tip.x * image.shape[1], pinky_tip.y * image.shape[0]])

                # Parmak hareketleri ve mesafeleri tanımlama
                distance_thumb_index = np.linalg.norm(thumb_tip_coords - index_finger_tip_coords)
                distance_index_middle = np.linalg.norm(index_finger_tip_coords - middle_finger_tip_coords)
                distance_middle_ring = np.linalg.norm(middle_finger_tip_coords - ring_finger_tip_coords)
                distance_ring_pinky = np.linalg.norm(ring_finger_tip_coords - pinky_tip_coords)

                # Belirli parmak hareketlerini tanımlama
                if distance_thumb_index < 40:
                    cv2.putText(image, 'Thumb and Index Finger Close (Click!)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if distance_index_middle < 40:
                    cv2.putText(image, 'Index and Middle Finger Close (Zoom In!)', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if distance_middle_ring < 40:
                    cv2.putText(image, 'Middle and Ring Finger Close (Scroll Down)', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                if distance_ring_pinky < 40:
                    cv2.putText(image, 'Ring and Pinky Finger Close (Scroll Up)', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # Görüntüyü göster
        cv2.imshow('Hand Gesture Control Test', image)

        # 'q' tuşuna basarak çıkış yapma
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Kaynakları serbest bırakma
cap.release()
cv2.destroyAllWindows()
