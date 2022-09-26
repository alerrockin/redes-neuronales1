#tomado de https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py

"""
mnist_loader
~~~~~~~~~~~~

Una libreria para cargar los datos de la imagen MNIST.  Para los detalles de las estructuras de datos
que se devuelven, ver las cadenas de doc para ``load_data``
y ``load_data_wrapper``.  En la práctica, ``load_data_wrapper`` es la función
es la función a la que suele llamar nuestro código de red neuronal.
"""

#### Librerías
# Librerías estándar
import pickle
import gzip

# Librerías de terceros
import numpy as np


def load_data():
    """Devuelve los datos MNIST como una tupla que contiene los datos de entrenamiento
    los datos de validación y los datos de prueba.

    Los ``datos de entrenamiento`` se devuelven como una tupla con dos entradas.
    La primera entrada contiene las imágenes de entrenamiento reales.  Se trata de un
    numpy ndarray con 50.000 entradas.  Cada entrada es, a su vez, un
    ndarray numpy con 784 valores, que representan los 28 * 28 = 784
    píxeles en una sola imagen MNIST.

    La segunda entrada de la tupla ``training_data`` es un ndarray numpy
    que contiene 50.000 entradas.  Estas entradas son sólo los valores de los dígitos
    (0...9) de las imágenes correspondientes contenidas en la primera
    de la tupla.

    Los ``datos de validación`` y los ``datos de prueba`` son similares, salvo que
    que cada uno contiene sólo 10.000 imágenes.

    Este es un buen formato de datos, pero para su uso en redes neuronales es
    útil modificar un poco el formato de ``training_data``.
    Esto se hace en la función ``load_data_wrapper()``, ver
    más abajo.
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """Devuelve una tupla que contiene ``(datos_de_entrenamiento, datos_de_validación,
    datos_de_prueba``. Basado en ``load_data``, pero el formato es más
    conveniente para su uso en nuestra implementación de redes neuronales.

    En concreto, ``datos_de_entrenamiento`` es una lista que contiene 50.000
    2-tuplas ``(x, y)``.  ``x`` es un numpy.ndarray de 784 dimensiones
    que contiene la imagen de entrada.  ``y`` es un numpy.ndarray de 10 dimensiones
    numpy.ndarray que representa el vector unitario correspondiente al
    dígito correcto para ``x``.

    Los "datos de validación" y los "datos de prueba" son listas que contienen 10.000
    2-tuplas ``(x, y)``.  En cada caso, ``x`` es un archivo de 784 dimensiones
    numpy.ndarry que contiene la imagen de entrada, y ``y`` es la
    clasificación correspondiente, es decir, los valores de los dígitos (enteros)
    correspondientes a ``x``.

    Obviamente, esto significa que estamos utilizando formatos ligeramente diferentes para
    los datos de entrenamiento y los datos de validación/prueba.  Estos formatos
    resultan ser los más convenientes para su uso en nuestra red neuronal
    código."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """Devuelve un vector unitario de 10 dimensiones con un 1,0 en la posición j
    posición y ceros en el resto.  Esto se utiliza para convertir un dígito
    (0...9) en la correspondiente salida deseada de la red neuronal
    de la red neuronal."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

#tomado de https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py
