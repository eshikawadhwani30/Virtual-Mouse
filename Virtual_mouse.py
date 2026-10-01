import cv2
import mediapipe as mp
import pyautogui
import math

# Screen size
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            landmarks = []

            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((cx, cy))

                cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)

            # Move mouse with index finger
            x, y = landmarks[8]
            screen_x = screen_w * x / w
            screen_y = screen_h * y / h
            pyautogui.moveTo(screen_x, screen_y)

            # Thumb tip and index tip
            x1, y1 = landmarks[4]
            x2, y2 = landmarks[8]

            distance = math.hypot(x2 - x1, y2 - y1)

            # If fingers close → click
            if distance < 30:
                pyautogui.click()
                cv2.putText(frame, "CLICK", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
