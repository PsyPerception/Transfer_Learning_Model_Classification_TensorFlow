from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adadelta
import numpy as np
import os

"""Заранее зададим переменные шага обучения и пути к папке с нашими изображениями для обучения"""
BATCH_SIZE = 32  # шаг обучения
path = r'D:\Work\Py_Projects\doc_class\train_data'  # наш путь

def get_count_files(path):
    """ Функция получения общего кол-ва файлов в папке с нашими изображениями"""
    file_list = []
    for relpath, dirs, files in os.walk(path):
        for file in files:
            dirfile = os.path.join(relpath, file)
            file_list.append(dirfile)
    return len(file_list)

"""Подгружаем обученную модель для классификации ImageNet"""
base_model = MobileNet(weights='imagenet', include_top=False)  # include_top=False - отключает один слой сети ImageNet
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)  # Количество выходов на последнем слое - должно
# соответствовать кол-ву классов наших изображений

"""Создаем свою модель, включая в нее модель ImageNet"""
model = Model(inputs=base_model.input, outputs=preds)
for layer in model.layers[:20]:
    layer.trainable = False
for layer in model.layers[20:]:
    layer.trainable = True


"""Далее - функция предобработки наших данных. В метод flow_from_directory подаются параметры по порядку:
1. Путь к папке с обучающим набором изображений. Внутри необходимо разместить изображения по папкам соответственно нашим классам,
например  D:\Work\Py_Projects\doc_class\train_data\Cat
          D:\Work\Py_Projects\doc_class\train_data\Dog
2. Размер изображений, в котором обработает метод наши файлы. 
3. Цветовая схема: RGB или Gray(цветные будут изображения или черно-белые)
4. Размер "шага" обучения. Менять на свое усмотрение, при знании дела. 
5. Режим классификации. 
6. Режим перемешивания входящих изображений.
"""
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(path,
                                                    target_size=(312, 312),
                                                    color_mode='rgb',
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    shuffle=True)
"""Определяем шаг обучения для каждой эпохи"""
TRAIN_STEPS_PER_EPOCH = np.ceil((get_count_files(path) * 0.8 / BATCH_SIZE) - 1)
model.compile(optimizer=Adadelta(lr=0.003), loss='categorical_crossentropy', metrics=['accuracy'])  #компилируем (собираем) модель
step_size_train = TRAIN_STEPS_PER_EPOCH
model.fit_generator(generator=train_generator, # старт обучения модели
                    steps_per_epoch=150,
                    epochs=1)
model.save('model_new.h5') # сохранение полученной модели
