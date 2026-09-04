import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response


# ===================== PATHS =====================

MODEL_PATH = "model/sign_language_model.h5"


# ===================== SETTINGS =====================

SEQUENCE_LENGTH = 30
THRESHOLD = 0.50


# ===================== ACTIONS =====================

# MUST match the training order

actions = [
    "HATE",
    "HELLO",
    "NO",
    "YES"
]


# ===================== LOAD MODEL =====================

print("Loading model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")

print("\nClasses used by the model:")

for index, action in enumerate(actions):
    print(f"{index} -> {action}")


# ===================== MEDIAPIPE =====================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ===================== CAMERA =====================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not accessible")
    exit()

print("\nCamera opened successfully.")


# ===================== VARIABLES =====================

sequence = []

prediction_text = "Waiting..."

confidence_text = "0%"

frame_counter = 0


# ===================== FRAME GENERATOR =====================

def generate_frames():

    global sequence
    global prediction_text
    global confidence_text
    global frame_counter

    while True:

        ret, frame = cap.read()

        if not ret:

            print("ERROR: Failed to read camera frame")

            break


        # ===================== MEDIAPIPE =====================

        image_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = hands.process(image_rgb)


        # ===================== HAND DETECTED =====================

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]

            landmarks = []


            # Extract 21 landmarks
            # 21 x 3 = 63 values

            for lm in hand.landmark:

                landmarks.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])


            # Add frame to sequence

            sequence.append(landmarks)

            sequence = sequence[-SEQUENCE_LENGTH:]


            frame_counter = len(sequence)


            # ===================== DRAW LANDMARKS =====================

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )


            # ===================== DISPLAY FRAME COUNT =====================

            cv2.putText(
                frame,
                f"Frames: {frame_counter}/{SEQUENCE_LENGTH}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )


            # ===================== PREDICTION =====================

            if len(sequence) == SEQUENCE_LENGTH:

                input_data = np.expand_dims(
                    np.array(sequence, dtype=np.float32),
                    axis=0
                )


                # Model prediction

                res = model.predict(
                    input_data,
                    verbose=0
                )[0]


                predicted_index = int(np.argmax(res))

                confidence = float(np.max(res))


                # Convert confidence to percentage

                confidence_percentage = confidence * 100

                confidence_text = f"{confidence_percentage:.1f}%"


                # ===================== TERMINAL DEBUG =====================

                print(
                    f"Prediction: {actions[predicted_index]} "
                    f"| Confidence: {confidence_percentage:.1f}%"
                )


                # ===================== ACCEPT PREDICTION =====================

                if confidence >= THRESHOLD:

                    prediction_text = actions[predicted_index]

                else:

                    prediction_text = "Unknown"


        # ===================== NO HAND =====================

        else:

            sequence = []

            frame_counter = 0

            prediction_text = "Waiting..."

            confidence_text = "0%"


        # ===================== DISPLAY PREDICTION =====================

        cv2.putText(
            frame,
            f"Prediction: {prediction_text}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


        # ===================== DISPLAY CONFIDENCE =====================

        cv2.putText(
            frame,
            f"Confidence: {confidence_text}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ===================== ENCODE FRAME =====================

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not ret:
            continue


        frame_bytes = buffer.tobytes()


        # ===================== SEND TO BROWSER =====================

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ===================== FLASK =====================

app = Flask(
    __name__,
    template_folder="templates"
)


# ===================== HOME =====================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ===================== VIDEO FEED =====================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ===================== RUN =====================

if __name__ == "__main__":

    print("\n========================================")
    print("AI REAL-TIME SIGN LANGUAGE RECOGNITION")
    print("========================================")

    print("\nSupported gestures:")

    for index, action in enumerate(actions):

        print(
            f"{index} -> {action}"
        )

    print("\nStarting Flask server...")

    print(
        "Open: http://127.0.0.1:5000"
    )

    app.run(
        debug=True
    )