from network import NeuralNetwork
from mnist_loader import load_data_wrapper
import time

def main():
    print("=" * 50)
    print("КЛАССИФИКАТОР РУКОПИСНЫХ ЦИФР MNIST")
    print("=" * 50)
    
    # Загрузка данных
    print("\n1. Загрузка данных MNIST...")
    training_data, validation_data, test_data = load_data_wrapper()
    print(f"   Обучающих примеров: {len(training_data)}")
    print(f"   Тестовых примеров: {len(test_data)}")
    
    # Создание сети
    # sizes = [784, 30, 10]  # 784 входа, 30 скрытых нейронов, 10 выходов
    print("\n2. Создание нейронной сети...")
    net = NeuralNetwork([784, 30, 10])
    print("   Архитектура: 784 -> 30 -> 10")
    
    # Попытка загрузить сохранённые веса
    if net.load("network_weights.json"):
        print("   (используются сохранённые веса)")
    else:
        print("   (новая инициализация)")
    
    # Обучение
    print("\n3. Начало обучения...")
    print("-" * 50)
    
    start_time = time.time()
    
    net.SGD(
        training_data=training_data,
        epochs=30,
        mini_batch_size=10,
        learning_rate=3.0,
        test_data=test_data
    )
    
    end_time = time.time()
    print("-" * 50)
    print(f"Обучение завершено за {end_time - start_time:.2f} секунд")
    
    # Сохранение весов
    print("\n4. Сохранение обученной сети...")
    net.save("network_weights.json")
    
    # Финальная оценка
    print("\n5. Финальная оценка на тестовых данных...")
    correct = net.evaluate(test_data)
    accuracy = correct / len(test_data) * 100
    print(f"   Правильных ответов: {correct} из {len(test_data)}")
    print(f"   Точность: {accuracy:.2f}%")
    print("\n" + "=" * 50)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 50)

if __name__ == "__main__":
    main()
