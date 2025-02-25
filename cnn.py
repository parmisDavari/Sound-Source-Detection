# %%
import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn import metrics


# %%
DATASET_PATH = "D:/new project taarlab/Dataset/"
blender_path = os.path.join(DATASET_PATH, "Blender/")
light_switch = os.path.join(DATASET_PATH, "Light_Switch/")
toaster = os.path.join(DATASET_PATH, "Toaster/")
water = os.path.join(DATASET_PATH, "Water/")

# %%
AUDIO_CLASSES = ["Blender", "Light_Switch", "Toaster", "Water"]

# %%
def save_as_numpy(X, y, output_path = "mfcc_features.npz"):
    np.savez(output_path, X = X, y = y)

# %%
from sklearn.utils import shuffle
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
def load_and_preprocess_audio(dataset_path, classes, sr=22050, n_mels=128, max_frames=216):
    X, y = [], []
    for idx, label in enumerate(classes):
        class_path = os.path.join(dataset_path, label)
        for file_name in os.listdir(class_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(class_path, file_name)
                audio, _ = librosa.load(file_path, sr=sr)
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                if mel_spec_db.shape[1] > max_frames:  
                    mel_spec_db = mel_spec_db[:, :max_frames]
                else:  
                    pad_width = max_frames - mel_spec_db.shape[1]
                    mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')

                X.append(mel_spec_db)
                y.append(idx)
                
    save_as_numpy(X, y)
    X = np.array(X)
    y = np.array(y)
    return X, y



X, y = load_and_preprocess_audio(DATASET_PATH, AUDIO_CLASSES)

y = to_categorical(y, num_classes=len(AUDIO_CLASSES))


X = X / np.max(X)
X = np.expand_dims(X, axis=-1)

print(f"size X: {X.shape}")
print(f"size y: {y.shape}")

X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"size X_train: {X_train.shape}")
print(f"size X_test: {X_test.shape}")


print(f"size Y_train: {y_train.shape}")
print(f"size Y_test: {y_test.shape}")


input_shape = X_train[0].shape
print(input_shape)
input_layer = Input(shape=input_shape)


conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
conv1 = BatchNormalization()(conv1)
pool1 = MaxPooling2D((2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = BatchNormalization()(conv2)
pool2 = MaxPooling2D((2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = BatchNormalization()(conv3)
pool3 = MaxPooling2D((2, 2))(conv3)


flat = Flatten()(pool3)
dense1 = Dense(128, activation='relu')(flat)
dense1 = Dropout(0.3)(dense1)
output_layer = Dense(len(AUDIO_CLASSES), activation='softmax')(dense1)

from sklearn.utils.class_weight import compute_class_weight

model = Model(inputs=input_layer, outputs=output_layer)

# class_weights = compute_class_weight("balanced", classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
# class_weights_dict = {i: class_weights[i] for i in range(len(AUDIO_CLASSES))}

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=70, batch_size=32)


layers = [conv1, pool1, conv2, pool2, conv3, pool3]
feature_extractor = Model(inputs = model.input, outputs = layers)

sample_features = feature_extractor.predict(np.expand_dims(X_test[0], axis=0))

for idx, feature_map in enumerate(sample_features):
    print(f"Layer {idx+1} output shape: {feature_map.shape}")


# %%
def plot_mfcc(file_path, sr=22050, n_mfcc=13):
    audio, _ = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.colorbar(label='MFCC Coefficients')
    plt.title('MFCC Visualization')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')
    plt.show()
    
plot_mfcc("D:/new project taarlab/Dataset/Blender/16.wav")

# %%
y_pred_probs = model.predict(X_test)  
y_pred = np.argmax(y_pred_probs, axis=1)  
y_true = np.argmax(y_test, axis=1)  


cm = metrics.confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()


plot_confusion_matrix(cm, AUDIO_CLASSES)

# %%
# Plot accuracy and loss
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# %%
model.summary()

# %%
from scipy.ndimage import zoom

def resize_feature_map(feature, target_shape):
    """Resize feature map to match the target shape."""
    zoom_factors = [target / source for target, source in zip(target_shape, feature.shape)]
    return zoom(feature, zoom_factors, order=1)  # Linear interpolation


# %%
def extract_features(model, X, layers):
    feature_extractor = Model(inputs=model.input, outputs=layers)
    return feature_extractor.predict(X)

# %%
from scipy.spatial.distance import cdist

def check_similarity(features1, features2, similarity_method = "euclidean"):
    # if features1.shape != features2.shape:
    #     features2 = resize_feature_map(features2, features1.shape)  # Resize to match
    # if similarity_method == "cosine":
    #     return 1 - cosine(features1.flatten(), features2.flatten())
    # elif similarity_method == "euclidean":
    #     return -np.linalg.norm(features1 - features2)
    
    features1 = features1.reshape(1, -1)
    features2 = features2.reshape(1, -1)
    
    distance = cdist(features1, features2, metric = similarity_method)
    return 1 - distance[0, 0]

# %%
def find_most_similar(features, layer_idx, feature_idx, metric="cosine"):
    print("layer_idx: ", layer_idx)
    print("feature_idx: ", feature_idx)
    reference_feature = features[layer_idx][feature_idx] #find the exact feature that we are working with right now
    similarities = []
    for j in range(len(features[layer_idx])):
        if j != feature_idx:
            candidate_feature = features[layer_idx][j]
            similarity = check_similarity(reference_feature, candidate_feature, metric)
            similarities.append(similarity)
    most_similar_idx = np.argmax(similarities)
    return most_similar_idx, similarities[most_similar_idx]

# %%
def compare_outputs(model, features, idx, most_similar_idx, self_idx):
    original_output = model.predict(np.expand_dims(features[idx][self_idx], axis=0))
    similar_output = model.predict(np.expand_dims(features[idx][most_similar_idx], axis=0))
    return np.allclose(original_output, similar_output), original_output, similar_output

# %%
def calculate_seperation_index(seperation_idx, output):
    return (seperation_idx[output] + 1)

# %%
feature_layers = [conv1, pool1, conv2, pool2, conv3, pool3]


# %%
seperation_idx = [[0 for _ in range(6)] for _ in range(len(AUDIO_CLASSES))]

def analyze_features_and_outputs(model, X, feature_layers):
    
    features = extract_features(model, X, feature_layers)
    print(f"Extracted {len(features)} feature sets")
    results = []
    print(f"Number of feature sets extracted: {len(features)}") 
    for i, feature_set in enumerate(features):
        print(f"Feature set {i} has {len(feature_set)} layers")


    # seperation_idx = [output for i,output in enumerate(AUDIO_CLASSES)]

    for layer_idx, feature_set in enumerate(features):
        print(f"Layer {layer_idx} has {len(feature_set)} features")
        for feature_idx in range(len(feature_set)):
            
            most_similar_idx, similarity = find_most_similar(features, layer_idx, feature_idx) #idx shouldn't be passed on to this function or should be calculated 
            
            
            same_output, original_output, similar_output = compare_outputs(
                model, features, layer_idx, most_similar_idx, feature_idx
            )
            
            print("this is original output:", original_output)
            class_idx = np.argmax(original_output) #this is always returning 0
            
            if(same_output):
                print("yes, same output")
                seperation_idx[class_idx][layer_idx] += 1 #
            
            results.append({
                "layer_idx": layer_idx,
                "feature_idx": feature_idx,
                "most_similar_idx": most_similar_idx,
                "similarity": similarity,
                "same_output": same_output,
                "original_output": original_output,
                "similar_output": similar_output
            })
    
    return results


# %%
import matplotlib.pyplot as plt

def plot_SI_per_layer(seperation_idx, class_idx, layer_names):
    num_layers = len(layer_names)

    # Extract SI values for the selected class
    si_values = [seperation_idx[class_idx][layer] for layer in range(num_layers)]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(layer_names, si_values, marker='o', linestyle='-', color='b', label=f'Class {class_idx} SI')
    plt.xlabel("Layers")
    plt.ylabel("Separation Index")
    plt.title(f"Separation Index per Layer for Class {class_idx}")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    
feature_layers = [conv1, pool1, conv2, pool2, conv3, pool3]
results = analyze_features_and_outputs(model, X_test, feature_layers)


for res in results:
    print(f"Layer {res['layer_idx']} - Feature {res['feature_idx']} - Similarity: {res['similarity']:.4f} - Same Output: {res['same_output']}")
   
print("\nSeparation Index Matrix:")
for row in seperation_idx:
    print(row)
 
class_idx = 0
layer_names = ["conv1", "pool1", "conv2", "pool2", "conv3", "pool3"]
plot_SI_per_layer(seperation_idx, class_idx, layer_names)


