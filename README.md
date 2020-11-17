# Transfer_Learning_Model_Classification_TensorFlow
Цель данного проекта - реализовать доступный код с пояснениями и примерами(как это вижу я) для использования трансферного обучения модели [MobileNet](https://keras.io/api/applications/#mobilenet) с весами ImageNet в задаче классификации своих изображений на фреймворке [TensorFlow](https://ru.wikipedia.org/wiki/TensorFlow). 
***
Проект будет содержать два основных файла: файл [training_transfer_learning_network.py](https://github.com/PsyPerception/Transfer_Learning_Model_Classification_TensorFlow/blob/master/training_transfer_learning_network.py) - для обучения своей модели на основе MobileNet, файл [testing_with_examples.py](https://github.com/PsyPerception/Transfer_Learning_Model_Classification_TensorFlow/blob/master/testing_with_examples.py) - для тестирования полученной модели на новых данных с проверкой точности.
## 1. Установка
Клонируйте репозиторий с помощью командной строки  в приложение. Установите библиотеки в соответствии с файлом **requirements.txt** (pip install -r requirements.txt)
## 2. Описание
Одно из лучших решений в создании чего-либо - использовать существующие наработки и решения. Нет необходимости изобретать велосипед. Так и в создании несложных моделей нейронных сетей. Берем натренерованную модель MobileNet, срезаем слой, добавляем свой со своими классами - тестируем.
## 3. Задача
Необходима модель для классификации изображений при небольшом (относительно) наборе данных. На практике опыт показывает, что небольшой набор данных это от 5 до 10 тысяч изображений на класс. Но можно попытаться работать и с меньшим количеством. В целом, размер данных по 2 тысячи изображений на класс с подходом трансферного обучения может дать показания точности модели при обучении вплоть до 0.8. Больше данных - выше точность.
***
### Выполнение
При запуске советую для начала разобраться с импортом библиотек. В последнее время TensorFlow часто обновляется и то, что работало у вас раньше, может не работать в этом коде. Обратите внимание на **версии библиотек**. Если у вас стоит последняя версия TF не надо надеяться, что "сойдет и моя версия". Часто именно это становится причиной огромных логов ошибок. 
```from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adadelta
```
Проверьте, что эти импорты корректно работают.
Так же, стоит обратить внимание - если вы уже пользовались [трансферным обучением](https://academy.yandex.ru/posts/transfernoe-obuchenie-pochemu-deep-learning-stal-dostupnee)  - стоит удалить скачанные модели MobileNet. По умолчанию они лежат в ***C:\Users\\"User Name"\\.keras\models*** и имеют название по типу **mobilenet_1_0_224_tf_no_top.h5**. Это связано с несостыковкой версий моделей с версией используемого Keras.
