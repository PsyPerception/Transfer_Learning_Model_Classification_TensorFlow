import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import numpy as np
from tensorflow.keras.preprocessing import image
import shutil

"""Определяем несколько констант, необходимых для тестирования нашей модели"""

test_folder = 'test_images'  # путь к папке с изображениями для тестирования
final_folder = 'result'  # финальная папка, куда будут скопированы результаты, удовлетворяющие условию обработки
model_path = 'model.h5'  # полный путь к наешй модели, если он ен рядом в папке с исполняемым скриптом то указывайте
# весь путь например D:\Work\Py_Projects\easy_classification\conv(MobileNet97%).h5
model = keras.models.load_model(model_path)  # загрузка модели в переменную model
num_classes = 2  # количество классов, равное количеству классов при обучении
size = 312  # размер входящего изображения для обработки (в данном примере 312x312)
classes = ['first', 'second']  # задаем название для наших классов, тут класс 0 будет doc, класс 1 - other

""""Функция для создания списка файлов в тестируемой папке. Работает рекурсивно (считывает и подпапки)."""


def load_img(path):
    file_list = []
    for relpath, dirs, files in os.walk(path):
        for file in files:
            dirfile = os.path.join(relpath, file)
            file_list.append(dirfile)
    return file_list


""""Далее - наш основной цикл для проверки модели. Будет проверен каждый файл изображения, в случае, если модель уверена более
чем на 90% (0.9), что файл относится к первому классу, а у нас это класс 0 с названием "first", из заданного 
 параметра classes = ['first', 'second'], файл будет скопирован в указанную нами финальную папку."""

if __name__ == "__main__":
    for file in load_img(test_folder):  # запускаем цикл по всем файлам из списка файлов
        try:  # добавляем обработку исключений, в случае ошибок при обработке изображений, например битый файл
            img = image.load_img(file, target_size=(size, size, 3), color_mode="rgb")  # загрузка изображения из списка
            print(os.path.split(file))  # вывод пути и названия обрабатываемого файла в консоль
            x = image.img_to_array(img)  # преобразование массива изображения в нужный вид
            x = x.reshape(1, size, size, 3)  # изменение формы массива
            x /= 255  # деление значений массива для корректной обработки
            predict_class = np.argmax(model.predict(x))  # определение класса изображения
            if predict_class == 0:  # если класс 0, то есть "first"
                if np.max(model.predict(x)) > 0.9:  # если модель уверена более, чем на 90%
                    shutil.copy(file, final_folder)  # копировать файл в указанную финальную папку
        except:
            print(
                f'File open Error:_____>{file}')  # выполняется в случае, если файл из папки для тестрирования не
            # удалось открыть
