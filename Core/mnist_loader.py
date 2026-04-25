import pickle
import gzip
import numpy as np
import os
import urllib.request

def download_mnist():
    """Скачивание файла MNIST, если его нет"""
    url = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"
    filename = "data/mnist.pkl.gz"
    
    if not os.path.exists(filename):
        print("Скачивание MNIST...")
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(url, filename)
        print("Скачивание завершено!")
    
    return filename

def load_data():
    """
    Возвращает кортеж (training_data, validation_data, test_data)
    """
    filename = download_mnist()
    
    with gzip.open(filename, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """
    Возвращает данные в формате, готовом для обучения нейронной сети
    
    training_data: список кортежей (x, y)
        x -- numpy вектор 784x1
        y -- numpy вектор 10x1 (one-hot кодирование)
    
    test_data: список кортежей (x, y)
        x -- numpy вектор 784x1
        y -- цифра (int)
    """
    tr_d, va_d, te_d = load_data()
    
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    
    return training_data, validation_data, test_data

def vectorized_result(j):
    """Превращает цифру j в one-hot вектор"""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
