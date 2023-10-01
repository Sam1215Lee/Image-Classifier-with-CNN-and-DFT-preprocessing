# -*- coding: utf-8 -*-
"""
Created on Mon May 29 02:25:33 2023

@author: Eason
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:05:06 2023

@author: Eason
"""

import tensorflow as tf
from tensorflow.keras import layers
import os
from PIL import Image
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from PIL import ImageOps
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

class CNNAgent:
    # Define the model architecture
    def __init__(self, input_shape, train_data, train_label, test_data, test_label,callbacks):
        self.shape = input_shape
        # Shuffle training data and labels
        self.X_train, self.Y_train = shuffle(train_data, train_label)
        self.X_test, self.Y_test = test_data, test_label
        # Create the CNN model
        self.model = self.create()
        self.history = None
        self.callbacks = callbacks

    def create(self):
        # Build the CNN model
        model = tf.keras.Sequential([
            layers.Conv2D(32, (5, 5), activation='relu', input_shape=(self.shape[0], self.shape[1], 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(4, activation='softmax')
        ])
        # Compile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        # Print the model summary
        model.summary()
        return model

    def train(self):
        # Train the CNN model and store the training history
        self.history = self.model.fit(self.X_train, self.Y_train, epochs=50, validation_data=(self.X_test, self.Y_test), callbacks=self.callbacks)
    def plot_curve(self):
        # Plot the training and validation accuracy and loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.show()

    def plot_confusion(self, true_label, predictions, title):
        # Plot the confusion matrix
        pre = np.argmax(predictions, axis=1)
        cm = confusion_matrix(np.argmax(true_label, axis=1), pre)
        fig = plt.figure(figsize=(8, 6))
        plt.title(title)
        sn.heatmap(cm, annot=True, cmap='OrRd', fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('True Label')
        plt.show()

    def result_csv(self, path, image_names, pre_label):
        # Save the prediction results to an Excel file
        result = pd.DataFrame({'Image Name': image_names, 'Predicted Category': pre_label})
        result.to_excel(path, index=False)

    def save(self, path):
        # Save the trained CNN model
        self.model.save(path)
        
        
class preprocessing:
    def __init__(self, train_path_list,test_path_list,target_size):
        self.target_size = target_size
        self.train_data , self.train_label ,self.train_name = self.load(train_path_list,arg = 'no') #yes: additional augmentation
        self.test_data , self.test_label , self.test_name = self.load(test_path_list,arg = 'no') #no: no additional augmentation
        
        
    def load(self,path,arg = 'yes'):
        ret_data = []
        ret_label = []
        ret_name = []
        count = 0
        
        for i in path: #path is a list of folder path
            file_list = os.listdir(i)
            # Filter out non-image files (if any)
            img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            img_files = [f for f in file_list if os.path.splitext(f)[1].lower() in img_extensions]
            for f in img_files: #f is image name is the folder
                img_path = os.path.join(i, f)
                img = Image.open(img_path)
                img = img.convert('L')
                img = img.resize(self.target_size)
                
                # Original image
                img_array = np.array(img)
                label = np.array([0, 0, 0, 0])
                label[count] = 1 #Create one-hot encoding label
                ret_data.append(img_array)
                ret_label.append(label)
                ret_name.append(f)
                
                if arg == 'yes':
                    # Add rotated versions of the image
                    for angle in [45,60]:
                        rotated_img = img.rotate(angle)
                        rotated_img_array = np.array(rotated_img)
                        ret_data.append(rotated_img_array)
                        ret_label.append(label)
         
                    # Add flipped versions of the image
                    flipped_img = ImageOps.flip(img)
                    flipped_img_array = np.array(flipped_img)
                    ret_data.append(flipped_img_array)
                    ret_label.append(label)
                    
            count += 1    
        
        ret_data , ret_label = np.array(ret_data),np.array(ret_label)
        return ret_data,ret_label,ret_name
            
if __name__=='__main__':
    root_path = 'C:/Users/88696\Desktop/三下課程/作品集/訓練集/weather_image'
    train_path = [root_path+'/train/cloudy',root_path+'/train/rain',root_path+'/train/shine',root_path+'/train/sunrise']
    test_path = [root_path+'/test/cloudy',root_path+'/test/rain',root_path+'/test/shine',root_path+'/test/sunrise']
    target_size=(300,300)
    
    pre = preprocessing(train_path,test_path,target_size)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    callbacks = [early_stopping, reduce_lr]
    
    # Load training data and labels
    train_data = pre.train_data/255
    train_data = np.expand_dims(train_data, axis=-1)
    train_label = pre.train_label
    train_name = pre.train_name
    # Load test data and labels
    test_data = pre.test_data/255
    test_data = np.expand_dims(test_data, axis=-1)
    test_label = pre.test_label
    test_name = pre.test_name
    
    Agent = CNNAgent((300,300),train_data,train_label,test_data,test_label,callbacks)
    Agent.train()
    Agent.plot_curve()
    
    #plot confusion
    train_pre = Agent.model.predict(train_data[::4,:,:,:])
    Agent.plot_confusion(train_label[::4],train_pre,"Train Confusion Matrix")
    test_pre = Agent.model.predict(test_data)
    Agent.plot_confusion(test_label,test_pre,"Test Confusion Matrix")
    #store predict result
