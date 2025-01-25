import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Import Libraries (already done)

# Step 2: Prepare Dataset
train_dir = r"C:\Users\KIIT\Desktop\AP(L)\Face Mask Detection\face-mask-dataset\Dataset\train\train"
test_dir = r"C:\Users\KIIT\Desktop\AP(L)\Face Mask Detection\face-mask-dataset\Dataset\test\test"
categories = ["with_mask", "without_mask"]
data = []

# Improved dataset loading and ensuring balance
for category in categories:
    path = os.path.join(train_dir, category)
    if not os.path.exists(path):
        print(f"Error: Path does not exist - {path}")
        continue

    if not os.listdir(path):
        print(f"Warning: No images found in {path}")
        continue

    class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_array is None:
                print(f"Warning: Unable to read image {img_path}")
                continue
            resized_array = cv2.resize(img_array, (128, 128))
            data.append([resized_array, class_num])
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Exit if no data is found
if len(data) == 0:
    print("No data found. Please check your dataset structure and paths.")
    exit()

# Check dataset balance
mask_count = sum([1 for _, label in data if label == 0])
no_mask_count = len(data) - mask_count
print(f"With Mask: {mask_count} | Without Mask: {no_mask_count}")

# Step 3: Shuffle and Split Data
np.random.shuffle(data)
X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 128, 128, 3) / 255.0  # Normalize
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create Improved Model (with more layers for better learning)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),  # Added another Conv and Pool layer for better capacity

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 categories: with_mask, without_mask
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train Model with Augmentation
datagen = ImageDataGenerator(rotation_range=30, zoom_range=0.2, horizontal_flip=True)
datagen.fit(X_train)

history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=15)  # Increased epochs for better training results

# Step 6: Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Step 7: Classification Report
y_pred = np.argmax(model.predict(X_test), axis=-1)
print(classification_report(y_test, y_pred, target_names=categories))

# Step 8: Save Model
model.save("face_mask_detector.h5")  # Ensure model is saved as .h5

# Step 9: Real-Time Detection with Improved Face Detection

def real_time_detection():
    cap = cv2.VideoCapture(0)
    model = tf.keras.models.load_model("face_mask_detector.h5")  # Correct file path and extension

    # Load the DNN face detection model for better accuracy
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare image for face detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, crop=False)
        net.setInput(blob)
        faces = net.forward()

        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]

            if confidence > 0.5:  # Only use faces with high confidence
                box = faces[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x, y, x2, y2) = box.astype("int")

                face = frame[y:y2, x:x2]
                resized_face = cv2.resize(face, (128, 128)) / 255.0
                reshaped_face = np.reshape(resized_face, (1, 128, 128, 3))

                prediction = model.predict(reshaped_face)
                label = np.argmax(prediction)
                color = (0, 255, 0) if label == 0 else (0, 0, 255)

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                cv2.putText(frame, categories[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Mask Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_detection()
