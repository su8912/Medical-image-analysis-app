
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet152V2, DenseNet121, InceptionResNetV2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_pre
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

IMG_SIZE = 224
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
label_dict = {k: v for v, k in enumerate(labels)}

def load_images(base_dir):
    X, Y = [], []
    for label in labels:
        path = os.path.join(base_dir, label)
        for img_file in tqdm(os.listdir(path), desc=f"Loading {label}"):
            img = cv2.imread(os.path.join(path, img_file))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            Y.append(label_dict[label])
    return np.array(X), to_categorical(Y)

# Load training and testing data
X_train, Y_train = load_images('/kaggle/input/tumor1/Training')
X_test, Y_test = load_images('/kaggle/input/tumor1/Testing')

# Normalize
X_train = resnet_pre(X_train.astype(np.float32))
X_test = resnet_pre(X_test.astype(np.float32))


def build_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(4, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

  base = ResNet152V2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
# base = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
# base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

for layer in base.layers:
    layer.trainable = False

model = build_model(base)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
check = ModelCheckpoint(filepath='best_model.keras', save_best_only=True , monitor='val_loss',
    mode='min',
    verbose=1)

history = model.fit(X_train, Y_train, validation_split=0.1, epochs=30, batch_size=32,
                    callbacks=[early, check])


loss, acc = model.evaluate(X_train, Y_train)
print(f"Train Accuracy: {acc * 100:.2f}%")
print(f"Train Loss: {loss:.4f}")

loss, acc = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

Y_pred_probs = model.predict(X_test)
Y_pred = np.argmax(Y_pred_probs, axis=1)
Y_true = np.argmax(Y_test, axis=1)

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(Y_true, Y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("Classification Report:\n", classification_report(Y_true, Y_pred, target_names=labels))

# ROC Curve
for i in range(4):
    fpr, tpr, _ = roc_curve(Y_test[:, i], Y_pred_probs[:, i])
    plt.plot(fpr, tpr, label=f'Class {labels[i]} (AUC = {roc_auc_score(Y_test[:, i], Y_pred_probs[:, i]):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

# Precision-Recall Curve
for i in range(4):
    precision, recall, _ = precision_recall_curve(Y_test[:, i], Y_pred_probs[:, i])
    plt.plot(recall, precision, label=f'{labels[i]}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.show()


import numpy as np

# Get predictions
Y_pred_probs = model.predict(X_test)
Y_pred = np.argmax(Y_pred_probs, axis=1)
Y_true = np.argmax(Y_test, axis=1)

print(f"{'Index':<6}{'Shape':<14}{'Probability':<15}{'Predicted':<20}{'Actual':<20}{'Confidence':<12}{'Correct?':<10}")
print("-" * 95)

correct = 0
for i in range(len(X_test)):
    prob = np.max(Y_pred_probs[i])  # Get the highest probability (confidence)
    pred_label = labels[Y_pred[i]]
    true_label = labels[Y_true[i]]
    match = "✔️" if pred_label == true_label else "❌"
    if match == "✔️":
        correct += 1
    shape = str(X_test[i].shape)
    confidence = prob * 100  # Convert to percentage
    print(f"{i:<6}{shape:<14}{confidence:<14.2f}%{pred_label:<20}{true_label:<20}{confidence:<12.2f}%{match:<10}")

# Summary accuracy
accuracy = (correct / len(X_test)) * 100
print(f"\nOverall Accuracy on test set: {accuracy:.2f}%")


import numpy as np
import matplotlib.pyplot as plt
import cv2

# Number of images to display at once
num_images = 9  # Can be 6, 9, or any other number as required

# Get predictions
Y_pred_probs = model.predict(X_test)
Y_pred = np.argmax(Y_pred_probs, axis=1)
Y_true = np.argmax(Y_test, axis=1)

# Create a subplot grid
fig, axes = plt.subplots(3, 3, figsize=(12, 12))  # 3x3 grid for 9 images
axes = axes.flatten()

correct = 0
for i in range(num_images):
    prob = np.max(Y_pred_probs[i])  # Get the highest probability (confidence)
    pred_label = labels[Y_pred[i]]
    true_label = labels[Y_true[i]]
    match = "✔️" if pred_label == true_label else "❌"
    confidence = prob * 100  # Convert to percentage

    # If prediction is correct, increment correct count
    if match == "✔️":
        correct += 1
    
    # Plot the image
    axes[i].imshow(X_test[i])  # Show the image
    axes[i].set_title(f"Pred: {pred_label}\nTrue: {true_label}\nConf: {confidence:.2f}%")
    axes[i].axis('off')

# Show the plot with all images
plt.tight_layout()
plt.show()

# Summary accuracy
accuracy = (correct / num_images) * 100
print(f"Overall Accuracy on the displayed images: {accuracy:.2f}%")


IMG_SIZE = 224
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
label_dict = {k: v for v, k in enumerate(labels)}

def load_images(base_dir):
    X, Y = [], []
    for label in labels:
        path = os.path.join(base_dir, label)
        for img_file in tqdm(os.listdir(path), desc=f"Loading {label}"):
            img = cv2.imread(os.path.join(path, img_file))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            Y.append(label_dict[label])
    return np.array(X), to_categorical(Y)

# Load training and testing data
X_train, Y_train = load_images('/kaggle/input/tumor1/Training')
X_test, Y_test = load_images('/kaggle/input/tumor1/Testing')

# Print dataset shapes before normalization
print(f"Before Normalization:")
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_test shape: {Y_test.shape}")

# Normalize
X_train = resnet_pre(X_train.astype(np.float32))
X_test = resnet_pre(X_test.astype(np.float32))

# Print dataset shapes after normalization
print(f"\nAfter Normalization:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# Save Keras model
model.save("brain_tumor_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open("brain_tumor_model.tflite", "wb") as f:
    f.write(tflite_model)

