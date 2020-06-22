import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout #use_drop_out
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.preprocessing import image
import os
#import pdb


os.environ['CUDA_VISIBLE_DEVICES']='-1'
mdl_clssify = Sequential()
mdl_clssify.add(Conv2D(64,3,3, input_shape=(64,64,3),activation='relu'))
mdl_clssify.add(MaxPooling2D(pool_size=(2,2)))
mdl_clssify.add(Conv2D(64,3,3,activation='relu'))
mdl_clssify.add(MaxPooling2D(pool_size=(2,2)))
#mdl.clssify.add(Dropout(0.5))
mdl_clssify.add(Flatten())

mdl_clssify.add(Dense(output_dim=128,activation='relu'))
mdl_clssify.add(Dense(output_dim=1 ,activation='sigmoid'))

mdl_clssify.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


trainData_generat = IDG(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
testData_generat = IDG(rescale=1./255)


set_train = trainData_generat.flow_from_directory('D:/final year project/AMD/data/train',
                                                   target_size=(64, 64), batch_size=9,
                                                    class_mode='binary')

set_test = testData_generat.flow_from_directory('D:/final year project/AMD/data/test',
                                                target_size=(64, 64), batch_size=9,
                                                class_mode='binary')

#pdb.set_trace() 

history =mdl_clssify.fit_generator(set_train,
                            steps_per_epoch=100,
                            epochs=10,
                            validation_data=set_test,
                            nb_val_samples=119)



test_image = image.load_img('D:/final year project/AMD/data/test/AMD/A0064.jpg', target_size = (64, 64))
#plt.imshow(mpimg.imread('D:/final year project/AMD/data/test/AMD/A0064.jpg'))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = mdl_clssify.predict(test_image)
set_train.class_indices

if result[0][0] == 1:
    prediction = 'normal'
else:
    prediction = 'AMD'
    
print(prediction)

mdl_clssify.save('amd.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, '2', label='training')
plt.plot(epochs, val_acc, 'b', label='validation')
plt.title('Accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, '2', label='training')
plt.plot(epochs, val_loss, 'b', label='validation')
plt.title('Loss')
plt.legend()
plt.show()