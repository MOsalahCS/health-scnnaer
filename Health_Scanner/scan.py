import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator


train_path='Data/train'
test_path='Data/test'
val_path='Data/valid'


img_height=224
img_width = 224 


train_datagen=ImageDataGenerator(rescale=1./255,
                                 
                                 shear_range = 0.2,
                                 zoom_range = 0.2,
                                 horizontal_flip = True,    
                                 
                                 )

test_datagen=ImageDataGenerator(rescale=1./255,
                                
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True,) 

 
valid_datagen=ImageDataGenerator(rescale=1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True,)  


train_set=train_datagen.flow_from_directory(train_path,
                                           target_size=(img_height,img_width),
                                           color_mode='grayscale',
                                           batch_size=16,
                                           class_mode='binary'
                                            
                                            
                                            )

test_set=test_datagen.flow_from_directory(test_path,
                                          target_size=(img_height,img_width),
                                          color_mode='grayscale',
                                          batch_size=16,
                                          class_mode='binary'
                                          
                                          )

valid_set=test_datagen.flow_from_directory(val_path,
                                           target_size=(img_height,img_width),
                                           color_mode='grayscale',
                                           batch_size=16,
                                           class_mode='binary'
                                           
                                           )


plt.figure(figsize=(12, 12))
for i in range(0, 10):
    plt.subplot(2, 5, i+1)
    for X_batch, Y_batch in train_set:
        image = X_batch[0]        
        dic = {0:'NORMAL', 1:'CANCER'}
        plt.title(dic.get(Y_batch[0]))      
        plt.axis('off')
        plt.imshow(np.squeeze(image),cmap='gray',interpolation='nearest')
        break
plt.tight_layout()
plt.show()

cnn_model=tf.keras.models.Sequential()
# Conv layer 1 
cnn_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
# Conv layer 2 
cnn_model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))



# Flatten 

cnn_model.add(tf.keras.layers.Flatten())
# two hidden layers 
cnn_model.add(tf.keras.layers.Dense(activation = 'relu', units = 200))
cnn_model.add(tf.keras.layers.Dense(activation = 'relu', units = 100))
# output layer 
cnn_model.add(tf.keras.layers.Dense(activation = 'sigmoid', units = 1))
cnn_model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

cnn_model.summary()
histroy = cnn_model.fit(train_set,epochs=17, validation_data=valid_set)
print(histroy.history.keys())
pd.DataFrame(histroy.history).plot()
model_acc= cnn_model.evaluate(test_set)


pred=cnn_model.predict(test_set)
prediction=pred.copy()
prediction[prediction <= 0.5] = 0 #0 means no cancer 
prediction[prediction > 0.5] = 1 #1 means have cancer 


test_img = tf.keras.preprocessing.image.load_img('predictions/cancer4.png', target_size=(img_height, img_width),color_mode='grayscale',interpolation='nearest')
pp_test_img = tf.keras.preprocessing.image.img_to_array(test_img)
pp_test_img = pp_test_img/255
pp_test_img = np.expand_dims(pp_test_img, axis=0)
result= cnn_model.predict(pp_test_img)
plt.figure(figsize=(6,6))
plt.axis('off')
if result> 0.5: 
    out = ('I am {:.2%} percent confirmed that this is a Lung Cancer case'.format(result[0][0]))
    
else: 
    out = ('I am {:.2%} percent confirmed that this is a Normal case'.format(1-result[0][0]))
plt.title(" Chest CT scan\n"+out)  
plt.imshow(np.squeeze(pp_test_img))
plt.show()

from sklearn.metrics import confusion_matrix,classification_report

cm = pd.DataFrame(data=confusion_matrix(test_set.classes, prediction, labels=[0, 1]),index=["Actual Normal", "Actual Cancer"],
columns=["Predicted Normal", "Predicted Cancer"])
import seaborn as sns
sns.heatmap(cm,annot=True,fmt="d")

print(classification_report(y_true= test_set.classes, y_pred=prediction ,target_names=['NORMAL','CANCER'] ))

#####################

cnn_model.save('best/trained_model.h5')

trained_model=tf.keras.models.load_model('best/trained_model.h5')
acc=trained_model.evaluate(test_set)






