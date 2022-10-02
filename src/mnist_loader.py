#tomado de https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py

"""
mnist_loader
~~~~~~~~~~~~

Una libreria para cargar los datos de la imagen MNIST.  Para los detalles de las estructuras de datos
que se devuelven, ver las cadenas de doc para ``cargar_datos``
y ``cargar_datos_tupla``.  En la práctica, ``cargar_datos_tupla`` es la función
es la función a la que suele llamar nuestro código de red neuronal.
"""

#### Librerías
# Librerías estándar
import pickle
import gzip

# Librerías de terceros
import numpy as np


def cargar_datos():
    """Devuelve los datos MNIST como una tupla que contiene los datos de entrenamiento
    los datos de validación y los datos de prueba.

    Los ``datos de entrenamiento`` se devuelven como una tupla con dos entradas.
    La primera entrada contiene las imágenes de entrenamiento reales.  Se trata de un
    numpy ndarray con 50.000 entradas.  Cada entrada es, a su vez, un
    ndarray numpy con 784 valores, que representan los 28 * 28 = 784
    píxeles en una sola imagen MNIST.

    La segunda entrada de la tupla ``datos_de_entrenamiento`` es un ndarray numpy
    que contiene 50.000 entradas.  Estas entradas son sólo los valores de los dígitos
    (0...9) de las imágenes correspondientes contenidas en la primera
    de la tupla.

    Los ``datos de validación`` y los ``datos de prueba`` son similares, salvo que
    que cada uno contiene sólo 10.000 imágenes.

    Este es un buen formato de datos, pero para su uso en redes neuronales es
    útil modificar un poco el formato de ``datos_de_entrenamiento``.
    Esto se hace en la función ``cargar_datos_tupla()``, ver
    más abajo.
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    datos_de_entrenamiento, datos_de_validacion, datos_de_prueba = pickle.load(f, encoding="latin1")
    f.close()   
    return (datos_de_entrenamiento, datos_de_validacion, datos_de_prueba)


def cargar_datos_tupla():

    """Devuelve una tupla que contiene ``(datos_de_entrenamiento, datos_de_validación,
    datos_de_prueba``. Basado en ``cargar_datos``, pero el formato es más
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
    tr_d, va_d, te_d = cargar_datos()
    entradas_entrenamiento = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    resultados_entrenamiento = [resultado_vector(y) for y in tr_d[1]]
    datos_de_entrenamiento = zip(entradas_entrenamiento, resultados_entrenamiento)
    entradas_validacion = [np.reshape(x, (784, 1)) for x in va_d[0]]
    datos_de_validacion = zip(entradas_validacion, va_d[1])
    entradas_prueba = [np.reshape(x, (784, 1)) for x in te_d[0]]
    datos_de_prueba = zip(entradas_prueba, te_d[1])

    datos_de_entrenamiento = list(zip(entradas_entrenamiento, resultados_entrenamiento))
    datos_de_validacion = list(zip(entradas_validacion, va_d[1]))
    datos_de_prueba = list(zip(entradas_prueba, te_d[1]))

    return (datos_de_entrenamiento, datos_de_validacion, datos_de_prueba)


def resultado_vector(j):
    """Devuelve un vector unitario de 10 dimensiones con un 1,0 en la posición j
    posición y ceros en el resto.  Esto se utiliza para convertir un dígito
    (0...9) en la correspondiente salida deseada de la red neuronal
    de la red neuronal."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

#tomado de https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py
