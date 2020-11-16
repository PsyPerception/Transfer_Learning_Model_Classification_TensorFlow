import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import numpy as np
from tensorflow.keras.preprocessing import image
import shutil


model = keras.models.load_model(r'D:\Work\Py_Projects\doc_class\saved_models\conv(MobileNet97%).h5')
num_classes = 2
size = 312
classes = ['doc', 'other']


def load_img(path):
    file_list = []
    for relpath, dirs, files in os.walk(path):
        for file in files:
            dirfile = os.path.join(relpath, file)
            file_list.append(dirfile)
    return file_list

if __name__ == "__main__":
    for file in load_img(r'D:\Work\Py_Projects\doc_class\0'):
        try:
            img = image.load_img(file, target_size=(size, size, 3), color_mode="rgb")
            print(os.path.split(file))
            x = image.img_to_array(img)
            x = x.reshape(1, size, size, 3)
            x /= 255
            predict_class = np.argmax(model.predict(x))
            if predict_class == 0:
                if np.max(model.predict(x)) > 0.9:
                    shutil.copy(file, r'D:\Work\Py_Projects\doc_class\founded')
        except:
            print(f'file error      {file}')

