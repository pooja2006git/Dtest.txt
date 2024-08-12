import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix # type: ignore
from sklearn.metrics import precision_score, recall_score , f1_score# type: ignore



mnist = tf.keras.datasets.mnist #used for achieving accuracy and quick training of ds (pattern rec,NN,img processing)
import pandas as pd

#loading mnist dataset to train and eval

(df_train, y_train),(df_test,y_eval) = mnist.load_data()
df_train,df_test = df_train/ 255.0, df_test / 255.0 #normailzation ,range of pixel[0-255], normalize it to [0-1]
# building model
model = tf.keras.models.Sequential([    #necessary before feeding to dense layer/the it works
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Flatten(),#28x28 pixel to 1D array
  # dense or nn layer or fully connected neurons,activation fn(relu)(+-+,--0)
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),#20% of neurons droped out
  tf.keras.layers.Dense(10, activation='softmax')# output layer,input img to 1/10 digit class(0-9)
  #softmax activation- logical output to probability
])
#compiling
model.compile(optimizer='adam',           #adjusting parameters (weight)and avoid loss/accurate prediction
  loss='sparse_categorical_crossentropy',#diff b/w model's actual and predicted value
  metrics=['accuracy']) #traces accuracy

#training 
model.fit(df_train, y_train, epochs=5,batch_size=10) #splits as batch
model.evaluate(df_test, y_eval)#evaluation

# making predictions using the trained model with test data
#array of probability
y_pred = model.predict(df_test)
#array of probability to class
#class with predicted label
y_pred_classes = [np.argmax(element) for element in y_pred] #argmax--predict highest probability

#confusion matrix
conf_matrix = confusion_matrix(y_eval, y_pred_classes)
#DIAGONAL- TRUE PREDICTIONS 
#OFF-DIAGONAL - FALSE PREDICTIONS
plt.figure(figsize=(10, 8))

# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=True, linewidths=0.5, linecolor='black')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# Calculate accuracy
#trace--diagonal;  TRUE PREDICTION / TOTAL PREDICTION
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
print(f'Accuracy: {accuracy}')

# Calculate precision
#correctly predicted POSITIVE / TOTAL POSITIVE(predicted**)
precision = precision_score(y_eval, y_pred_classes, average='weighted')
print(f'Precision: {precision}')

# Calculate recall
#correctly predicted POSITIVE / TOTAL POSITIVE(actual**)
recall = recall_score(y_eval, y_pred_classes, average='weighted')
print(f'Recall: {recall}')
#calculating f1-score
#average by true instances of the class        #imbalanced dataset#with instance
f1 = f1_score(y_eval, y_pred_classes, average='weighted')
                                              #wihtout imbalance#without instance
# f1 = f1_score(y_eval, y_pred_classes, average='macro')
print(f'F1 Score: {f1}')





# import matplotlib.pyplot as plt
# import tensorflow as tf
# mnist = tf.keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# print("Training images shape:", train_images.shape)
# print("Training labels shape:", train_labels.shape)   #training and testing data sets are printed
# print("Test images shape:", test_images.shape)
# print("Test labels shape:", test_labels.shape)


# # Display the first image from the training dataset
# plt.imshow(train_images[0], cmap='gray')   #first image in dataset , cmap-->shows in gray scale
# plt.title(f"Label: {train_labels[0]}")     #im==image show
# plt.show()

# # Plot the first 5 images from the training dataset
# fig, axes = plt.subplots(1, 5, figsize=(10, 5))
# for i in range(5):
#     axes[i].imshow(train_images[i], cmap='gray')
#     axes[i].set_title(f"Label: {train_labels[i]}")
#     axes[i].axis('off')
# plt.show()
# print("Unique labels in the training set:", set(train_labels))



# # Display the last image from the training dataset
# plt.imshow(train_images[-1], cmap='gray')
# plt.title(f"Label: {train_labels[-1]}")
# plt.show()

# # Plot the last 5 images from the training dataset
# fig, axes = plt.subplots(1, 5, figsize=(10, 5))
# for i in range(5):
#     axes[i].imshow(train_images[-5+i], cmap='gray')
#     axes[i].set_title(f"Label: {train_labels[-5+i]}")
#     axes[i].axis('off')
# plt.show()
# print("Unique labels in the training set:", set(train_labels))
