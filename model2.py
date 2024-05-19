# This model is trained using the garbage classification 3 dataset from Kaggle
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# read the csv file
df = pd.read_csv("HackwithAI/garbage_class_3/train/_annotations.csv")

# convert the label to a one-hot encoded vector
df["class"] = pd.Categorical(df["class"])
df["class"] = df["class"].cat.codes

# shuffle
df = df.sample(frac=1).reset_index(drop=True)

# keep file name and label
df = df[["filename", "class"]]
print(df.head(50))

# preprocess the images stored in the file name column
def preprocess_image(img_path):
    img_path = "HackwithAI/garbage_class_3/train/" + img_path
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    return img


# create a dataset
dataset = tf.data.Dataset.from_tensor_slices((df["filename"].values, df["class"].values))
dataset = dataset.map(lambda x, y: (preprocess_image(x), y))
dataset = dataset.batch(32)

# create a model
model = tf.keras.models.Sequential([
    # with three convolutional layers, max pooling, and a dense layer
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(6)
])

model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

model.fit(dataset, epochs=10)

# visualize the results
acc = model.history.history['accuracy']
plt.plot(acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# save the model
model.save("HackwithAI/model2")
