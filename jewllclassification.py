
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

img_size = 224
batch_size = 32
epochs = 10

train_dir = r"H:\JewlerryDatasets\train"
test_dir = r"H:\JewlerryDatasets\test"
val_dir = r"H:\JewlerryDatasets\validation"

print(train_dir)
print(test_dir)
print(val_dir)

train_datagen = ImageDataGenerator(
    rescale= 1./255,
    rotation_range= 30,
    zoom_range=0.2,
    horizontal_flip=True
)
print(train_datagen)

val_test_datagen = ImageDataGenerator(rescale= 1./255)

print(val_test_datagen)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size= (img_size,img_size),
    batch_size = batch_size,
    class_mode="categorical"
)

print(train_gen.class_indices)

test_gen = val_test_datagen.flow_from_directory(

    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical"
)

print(test_gen.class_indices)

val_gen = val_test_datagen.flow_from_directory(

    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical"
)

print(val_gen.class_indices)

base_model = NASNetMobile (
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

print(base_model)

base_model.trainable = False

x = base_model.output
print(x)

x = GlobalAveragePooling2D()(x)
print(x)

x = Dense(120,activation='relu')(x)
print(x)

x = Dropout(0.3)(x)
print(x)

output = Dense(train_gen.num_classes,activation="softmax")(x)
print(output)

model = Model(inputs=base_model.input, outputs=output)
print(model)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(

    train_gen,
    validation_data=val_gen,
    epochs=epochs
)

test_loss , test_acc = model.evaluate(test_gen)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

model.save("nasnetmobile_clothclassification.h5")
print("\nModel successfully saved")

plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.legend()
plt.title("Model Accuracy")
plt.show()

plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.legend()
plt.title("Model Loss")
plt.show()

def predict_image(image_path):
    img = load_img(image_path,target_size=(224,224))
    img = img_to_array(img)
    img = img / 225.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    print("The Predication is: ",pred)
    class_names = list(train_gen.class_indices.keys())
    print("The class name is: ",class_names)
    predicated_class = class_names[np.argmax(pred)]
    print("The Predicted Class is: ",predicated_class)
    confidence = np.max(pred) * 100
    print("The Confidence is: ",confidence)

while True:
    path = input("\nEnter Image Path (or type 'exit'): ")
    if path.lower() == 'exit':
        break
    predict_image(path)



