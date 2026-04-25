"""
Демонстрация работы сети: распознавание вашей собственной цифры
"""

from network import NeuralNetwork
from mnist_loader import load_data_wrapper
import numpy as np
from PIL import Image
import os

def load_and_prepare_image(image_path):
    """
    Загрузка и подготовка изображения для нейронной сети
    
    Ожидается: изображение 28x28 пикселей, чёрная цифра на белом фоне
    """
    # Открываем изображение
    img = Image.open(image_path).convert('L')  # Чёрно-белое
    
    # Изменяем размер до 28x28
    img = img.resize((28, 28))
    
    # Преобразуем в массив numpy
    img_array = np.array(img)
    
    # Инвертируем цвета (MNIST: белая цифра на чёрном фоне)
    # Если у вас чёрная цифра на белом фоне:
    img_array = 255 - img_array
    
    # Нормализуем значения в диапазон [0, 1]
    img_array = img_array / 255.0
    
    # Превращаем в вектор 784x1
    return np.reshape(img_array, (784, 1))

def main():
    print("=" * 50)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ КЛАССИФИКАТОРА")
    print("=" * 50)
    
    # Загрузка данных (нужны только для теста)
    print("\nЗагрузка данных...")
    training_data, validation_data, test_data = load_data_wrapper()
    
    # Создание и загрузка обученной сети
    print("Создание нейронной сети...")
    net = NeuralNetwork([784, 30, 10])
    
    # Загрузка сохранённых весов (если есть)
    if net.load("network_weights.json"):
        print("Веса успешно загружены!")
    else:
        print("ОШИБКА: Сначала обучите сеть (python train.py)")
        return
    
    # Тестирование на случайных примерах из тестовой выборки
    print("\n" + "-" * 50)
    print("ТЕСТ НА СЛУЧАЙНЫХ ПРИМЕРАХ ИЗ MNIST")
    print("-" * 50)
    
    import random
    for i in range(5):
        idx = random.randint(0, len(test_data) - 1)
        x, y_true = test_data[idx]
        y_pred = net.predict(x)
        print(f"Пример {i+1}: Сеть предсказала {y_pred}, правильный ответ {y_true}")
    
    # Распознавание своей цифры
    print("\n" + "-" * 50)
    print("РАСПОЗНАВАНИЕ ВАШЕЙ ЦИФРЫ")
    print("-" * 50)
    
    # Создаём папку для рисунков
    os.makedirs("my_digit", exist_ok=True)
    
    print("""
    КАК ПОДГОТОВИТЬ СВОЮ ЦИФРУ:
    1. Откройте любой графический редактор
    2. Создайте изображение 28x28 пикселей
    3. Нарисуйте чёрную цифру на белом фоне
    4. Сохраните в папку 'my_digit' как 'digit.png'
    
    Или используйте любой рисунок, заменив путь в коде
    """)
    
    # Путь к вашему рисунку
    image_path = "my_digit/digit.png"
    
    if os.path.exists(image_path):
        x = load_and_prepare_image(image_path)
        y_pred = net.predict(x)
        print(f"\nРезультат: Нейронная сеть считает, что это цифра {y_pred}")
    else:
        print(f"\nФайл {image_path} не найден.")
        print("Создайте рисунок цифры и сохраните его по этому пути.")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
