import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import numpy as np
from tensorflow.keras.preprocessing import image
import shutil


model = keras.models.load_model('conv_new.h5')
num_classes = 2
size = 312
classes = ['doc', 'passport']


def load_img(path):
    file_list = []
    for relpath, dirs, files in os.walk(path):
        for file in files:
            dirfile = os.path.join(relpath, file)
            file_list.append(dirfile)
    return file_list

if __name__ == "__main__":
    path = r'D:\Work\Py_Projects\doc_class\test_other_imgs\c0c3f7c91fb9.jpg'
    img = image.load_img(path, target_size=(size, size, 3), color_mode="rgb")
    x = image.img_to_array(img)
    x = x.reshape(1, size, size, 3)
    x /= 255
    prediction = model.predict(x)
    prediction = np.argmax(prediction)
    print("Номер класса:", prediction)
    print("Название класса:", classes[prediction])
    # for file in load_img(r'D:\Work\Py_Projects\doc_class\test_other_imgs'):
    #     try:
    #         img = image.load_img(file, target_size=(size, size, 3), color_mode="rgb")
    #         print(os.path.split(file))
    #         x = image.img_to_array(img)
    #         x = x.reshape(1, size, size, 3)
    #         x /= 255
    #         prediction = np.argmax(model.predict(x))
    #         if prediction == 0:
    #             shutil.copy(file, r'D:\Work\Py_Projects\doc_class\founded')
    #     except:
    #         print(f'file error      {file}')

