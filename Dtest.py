import tensorflow as tf
mnist = tf.keras.datasets.mnist #used for achieving accuracy and quick training of ds (pattern rec,NN,img processing)
import pandas as pd
# loading dataset
df_train= pd.read_csv("C:\\Users\\91807\\Desktop\\test.csv")
df_test= pd.read_csv("C:\\Users\\91807\\Desktop\\test.csv")
y_train = df_train.pop('Platform')
y_eval = df_test.pop('Platform')
print(y_train)
#loading mnist dataset to train and eval

(df_train, y_train),(df_test,y_eval) = mnist.load_data()
df_train,df_test = df_train/ 255.0, df_test / 255.0 #normailzation ,range of pixel[0-255], normalize it to [0-1]
# building model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),#28x28 pixel to 1D array
  # dense or fully connected neurons,activation fn(relu)(+-+,--0)
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),#20% of neurons droped out
  tf.keras.layers.Dense(10, activation='softmax')# output layer,input img to 1/10 digit class(0-9)
  #softmax activation- logical output to probability
])
#compiling
model.compile(optimizer='adam',           #adjusting parameters and accurate prediction
  loss='sparse_categorical_crossentropy',#diff b/w model's actual and predicted value
  metrics=['accuracy']) #traces accuracy
#training 
model.fit(df_train, y_train, epochs=5)
model.evaluate(df_test, y_eval)#evaluation

import matplotlib.pyplot as plt
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("Training images shape:", train_images.shape)
print("Training labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)


# Display the first image from the training dataset
plt.imshow(train_images[0], cmap='gray')
plt.title(f"Label: {train_labels[0]}")
plt.show()

# Plot the first 5 images from the training dataset
fig, axes = plt.subplots(1, 5, figsize=(10, 5))
for i in range(5):
    axes[i].imshow(train_images[i], cmap='gray')
    axes[i].set_title(f"Label: {train_labels[i]}")
    axes[i].axis('off')
plt.show()
print("Unique labels in the training set:", set(train_labels))