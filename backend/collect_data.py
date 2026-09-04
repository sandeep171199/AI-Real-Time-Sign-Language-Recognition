import os
import glob
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# ============================================================
# PATHS
# ============================================================

DATA_PATH = os.path.join("backend", "data")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "sign_language_model.h5")


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 30

# Only train these 4 gestures
ACTIONS = [
    "HELLO",
    "YES",
    "NO",
    "HATE"
]


# ============================================================
# LOAD DATA
# ============================================================

X = []
y = []

print("\n========================================")
print("LOADING DATA")
print("========================================")

for label, action in enumerate(ACTIONS):

    action_path = os.path.join(DATA_PATH, action)

    if not os.path.exists(action_path):
        print(f"WARNING: {action} folder not found.")
        continue

    files = glob.glob(
        os.path.join(action_path, "*.npy")
    )

    print(f"{action}: {len(files)} samples")

    for file in files:

        sequence = np.load(file)

        # Only accept the expected shape
        if sequence.shape == (SEQUENCE_LENGTH, 63):

            X.append(sequence)
            y.append(label)

        else:
            print(
                f"Skipping {file} - "
                f"wrong shape {sequence.shape}"
            )


# ============================================================
# CONVERT TO NUMPY ARRAYS
# ============================================================

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("Classes:", ACTIONS)


# ============================================================
# CHECK DATA
# ============================================================

if len(X) == 0:
    print("\nERROR: No valid training data found.")
    exit()

if len(np.unique(y)) < 2:
    print("\nERROR: At least 2 gesture classes are required.")
    exit()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# ONE-HOT ENCODING
# ============================================================

y_train = to_categorical(
    y_train,
    num_classes=len(ACTIONS)
)

y_test = to_categorical(
    y_test,
    num_classes=len(ACTIONS)
)


# ============================================================
# BUILD LSTM MODEL
# ============================================================

model = Sequential([

    LSTM(
        128,
        return_sequences=True,
        input_shape=(SEQUENCE_LENGTH, 63)
    ),

    Dropout(0.3),

    LSTM(
        64,
        return_sequences=False
    ),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        len(ACTIONS),
        activation="softmax"
    )
])


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


print("\n========================================")
print("MODEL")
print("========================================")

model.summary()


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================")

history = model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_test, y_test),
    shuffle=True
)


# ============================================================
# EVALUATE MODEL
# ============================================================

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model.save(
    MODEL_PATH
)

print("\n========================================")
print("TRAINING COMPLETED")
print("========================================")

print(
    f"Model saved successfully to:\n{MODEL_PATH}"
)