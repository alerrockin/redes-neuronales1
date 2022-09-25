#tomado de https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py

"""
network.py
~~~~~~~~~~

Un módulo para implementar el algoritmo de aprendizaje de 
descenso de gradiente estocástico para una red neuronal directa.  
Los gradientes se calculan utilizando la retropropagación.  
Tenga en cuenta que me he centrado en hacer el código
simple, fácilmente legible y fácilmente modificable.  
No está optimizado, y omite muchas características deseables.
"""

#### Librerías
# Librerías estándar
import random

# Librerías de terceros
import numpy as np


class Network(object):

    def __init__(self, sizes):

        """La lista ``sizes`` contiene el número de neuronas en las
        respectivas capas de la red.  Por ejemplo, si la lista
        fuera [2, 3, 1] entonces sería una red de tres capas, con la
        primera capa contiene 2 neuronas, la segunda capa 3 neuronas
        y la tercera capa 1 neurona.  Los sesgos y pesos de la red
        red se inicializan aleatoriamente, utilizando una distribución gaussiana
        con media 0 y varianza 1.  Obsérvese que la primera
        capa se supone que es una capa de entrada, y por convención no
        no estableceremos ningún sesgo para esas neuronas, ya que los sesgos sólo se
        sólo se utilizan en el cálculo de las salidas de las capas posteriores."""
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Devuelve la salida de la red si ``a`` es la entrada."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):

        """Entrenar la red neuronal mediante el descenso de
        gradiente estocástico.  Los ``datos_de_entrenamiento`` son una lista de tuplas
        ``(x, y)`` que representan las entradas de entrenamiento y las salidas
        deseadas.  Los demás parámetros no opcionales son
        no opcionales se explican por sí mismos.  Si se proporciona ``datos_de_prueba``, la red
        red será evaluada contra los datos de prueba después de cada
        y se imprimirá el progreso parcial.  Esto es útil para
        seguimiento del progreso, pero ralentiza las cosas sustancialmente."""

        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1} / {2}").format(
                    j, self.evaluate(test_data), n_test)
            else:
                print ("Epoch {0} complete").format(j)

    def update_mini_batch(self, mini_batch, eta):

        """Actualizar los pesos y sesgos de la red aplicando
        el descenso de gradiente utilizando la retropropagación a un único mini lote.
        El ``mini_batch`` es una lista de tuplas ``(x, y)``, y ``eta``
        es la tasa de aprendizaje."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):

        """Devuelve una tupla ``(nabla_b, nabla_w)`` que representa el
        gradiente de la función de coste C_x.  ``nabla_b`` y
        ``nabla_w`` son listas de matrices numpy capa por capa, similares
        a ``self.biases`` y ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # propagacion hacia adelante
        activation = x
        activations = [x]  # lista para almacenar todas las activaciones, capa por capa
        zs = []  # lista para almacenar todos los vectores z, capa por capa
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # paso atrás
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Obsérvese que la variable l en el bucle de abajo se usa un poco.Aquí,
        # l = 1 significa la última capa de neuronas, l = 2 es la
        # la penúltima capa, y así sucesivamente.  Es una renumeración del
        # esquema del libro, utilizado aquí para aprovechar el hecho
        # que Python puede usar índices negativos en las listas.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):

        """Devuelve el número de entradas de prueba para las que la red neural
        produce el resultado correcto. Tenga en cuenta que la salida de la red neuronal
        se supone que la salida de la red neuronal es el índice de la neurona
        neurona de la última capa tenga la mayor activación."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):

        """Devuelve el vector de derivadas parciales \partial C_x /
        \partial a para las activaciones de salida."""
        return (output_activations-y)

#### Funciones varias


def sigmoid(z):
    """La función sigmoidea."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivada de la función sigmoidea."""
    return sigmoid(z)*(1-sigmoid(z))

#tomado de https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
