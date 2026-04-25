import numpy as np
import random
import json
import os

class NeuralNetwork:
    def __init__(self, sizes):
        """
        Инициализация нейронной сети
        
        Параметры:
        sizes -- список [784, 30, 10] -> входной слой, скрытый слой, выходной слой
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        # Инициализация весов и смещений случайными значениями
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        """
        Прямой проход (получение ответа сети)
        
        Параметры:
        a -- входной вектор (изображение 784x1)
        
        Возвращает:
        Выходной вектор 10x1 (вероятности для цифр 0-9)
        """
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        """
        Стохастический градиентный спуск (обучение сети)
        
        Параметры:
        training_data -- обучающие данные
        epochs -- количество эпох обучения
        mini_batch_size -- размер мини-батча
        learning_rate -- скорость обучения
        test_data -- тестовые данные (для оценки точности)
        """
        n = len(training_data)
        
        for epoch in range(epochs):
            # Перемешиваем обучающие данные
            random.shuffle(training_data)
            
            # Разбиваем на мини-батчи
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            
            # Обучение на каждом мини-батче
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            
            # Вывод прогресса
            if test_data:
                test_score = self.evaluate(test_data)
                print(f"Эпоха {epoch}: {test_score} / {len(test_data)}")
            else:
                print(f"Эпоха {epoch} завершена")
    
    def update_mini_batch(self, mini_batch, learning_rate):
        """
        Обновление весов и смещений на основе одного мини-батча
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Обновляем веса и смещения
        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x, y):
        """
        Алгоритм обратного распространения ошибки
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Прямой проход
        activation = x
        activations = [x]
        zs = []
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        # Обратный проход (ошибка на выходном слое)
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # Обратный проход (ошибка на скрытых слоях)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        """
        Оценка точности на тестовых данных
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(pred == y) for (pred, y) in test_results)
    
    def predict(self, x):
        """
        Предсказание для одного изображения
        """
        output = self.feedforward(x)
        return np.argmax(output)
    
    def save(self, filename="network_weights.json"):
        """
        Сохранение обученных весов в файл
        """
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases]
        }
        with open(filename, "w") as f:
            json.dump(data, f)
        print(f"Сеть сохранена в {filename}")
    
    def load(self, filename="network_weights.json"):
        """
        Загрузка весов из файла
        """
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
            self.weights = [np.array(w) for w in data["weights"]]
            self.biases = [np.array(b) for b in data["biases"]]
            print(f"Сеть загружена из {filename}")
            return True
        return False
    
    @staticmethod
    def sigmoid(z):
        """Сигмоидная функция активации"""
        return 1.0 / (1.0 + np.exp(-z))
    
    @staticmethod
    def sigmoid_prime(z):
        """Производная сигмоидной функции"""
        return NeuralNetwork.sigmoid(z) * (1 - NeuralNetwork.sigmoid(z))
    
    @staticmethod
    def cost_derivative(output_activations, y):
        """Производная функции стоимости"""
        return output_activations - y
