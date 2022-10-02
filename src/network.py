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

#a = valor neuronas
#b = valor umbrales
#w = valor pesos

class Network(object):

    def __init__(self, num_neuronas):

        """La lista ``num_neuronas`` contiene el número de neuronas en las
        respectivas capas de la red.  Por ejemplo, si la lista
        fuera [2, 3, 1] entonces sería una red de tres capas, con la
        primera capa contiene 2 neuronas, la segunda capa 3 neuronas
        y la tercera capa 1 neurona.  Los sesgos y pesos de la red
        red se inicializan aleatoriamente, utilizando una distribución gaussiana
        con media 0 y varianza 1.  Obsérvese que la primera
        capa se supone que es una capa de entrada, y por convención no
        no estableceremos ningún sesgo para esas neuronas, ya que los sesgos sólo se
        sólo se utilizan en el cálculo de las salidas de las capas posteriores."""
        
        self.num_capas = len(num_neuronas)
        self.num_neuronas = num_neuronas
        self.umbrales = [np.random.randn(y, 1) for y in num_neuronas[1:]]
        self.pesos = [np.random.randn(y, x)
                        for x, y in zip(num_neuronas[:-1], num_neuronas[1:])]

    def propagacion_hacia_adelante(self, a):
        """Devuelve la salida de la red si ``a`` es la entrada."""
        for b, w in zip(self.umbrales, self.pesos):
            a = sigmoide(np.dot(w, a)+b)
        return a

    def SGD(self, datos_de_entrenamiento, epocas, capacidad_lote, tasa_aprendizaje,
            datos_de_prueba=None):

        """Entrenar la red neuronal mediante el descenso de
        gradiente estocástico.  Los ``datos_de_entrenamiento`` son una lista de tuplas
        ``(x, y)`` que representan las entradas de entrenamiento y las salidas
        deseadas.  Los demás parámetros no opcionales son
        no opcionales se explican por sí mismos.  Si se proporciona ``datos_de_prueba``, la red
        red será evaluada contra los datos de prueba después de cada
        y se imprimirá el progreso parcial.  Esto es útil para
        seguimiento del progreso, pero ralentiza las cosas sustancialmente."""

        if datos_de_prueba:
            n_test = len(datos_de_prueba)
        n = len(datos_de_entrenamiento)
        for j in range(epocas):
            random.shuffle(datos_de_entrenamiento)
            lotes = [
                datos_de_entrenamiento[k:k+capacidad_lote]
                for k in range(0, n, capacidad_lote)]
            for lote in lotes:
                self.actualizar_lote(lote, tasa_aprendizaje)
            if datos_de_prueba:
                print("Epoca {0}: {1} / {2}".format(
                    j, self.evaluate(datos_de_prueba), n_test))
            else:
                print ("Epoca {0} completada".format(j))

    def actualizar_lote(self, lote, tasa_aprendizaje):

        """Actualizar los pesos y sesgos de la red aplicando
        el descenso de gradiente utilizando la retropropagación a un único mini lote.
        El ``lote`` es una lista de tuplas ``(x, y)``, y ``tasa_aprendizaje``
        es la tasa de aprendizaje."""

        nabla_b = [np.zeros(b.shape) for b in self.umbrales]
        nabla_w = [np.zeros(w.shape) for w in self.pesos]
        for x, y in lote:
            delta_nabla_b, delta_nabla_w = self.retropropagacion(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.pesos = [w-(tasa_aprendizaje/len(lote))*nw
                        for w, nw in zip(self.pesos, nabla_w)]
        self.umbrales = [b-(tasa_aprendizaje/len(lote))*nb
                       for b, nb in zip(self.umbrales, nabla_b)]

    def retropropagacion(self, x, y):

        """Devuelve una tupla ``(nabla_b, nabla_w)`` que representa el
        gradiente de la función de coste C_x.  ``nabla_b`` y
        ``nabla_w`` son listas de matrices numpy capa por capa, similares
        a ``self.umbrales`` y ``self.pesos``."""
        nabla_b = [np.zeros(b.shape) for b in self.umbrales]
        nabla_w = [np.zeros(w.shape) for w in self.pesos]
        # propagacion hacia adelante
        activacion = x
        activaciones = [x]  # lista para almacenar todas las activaciones, capa por capa
        zs = []  # lista para almacenar todos los vectores z, capa por capa
        for b, w in zip(self.umbrales, self.pesos):
            z = np.dot(w, activacion)+b
            zs.append(z)
            activacion = sigmoide(z)
            activaciones.append(activacion)
        # propagación hacia atras
        delta = self.cost_derivative(activaciones[-1], y) * \
            sigmoide_derivada(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activaciones[-2].transpose())
        # Obsérvese que la variable l en el bucle de abajo se usa un poco.Aquí,
        # l = 1 significa la última capa de neuronas, l = 2 es la
        # la penúltima capa, y así sucesivamente.  Es una renumeración del
        # esquema del libro, utilizado aquí para aprovechar el hecho
        # que Python puede usar índices negativos en las listas.
        for l in range(2, self.num_capas):
            z = zs[-l]
            sp = sigmoide_derivada(z)
            delta = np.dot(self.pesos[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activaciones[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, datos_de_prueba):

        """Devuelve el número de entradas de prueba para las que la red neural
        produce el resultado correcto. Tenga en cuenta que la salida de la red neuronal
        se supone que la salida de la red neuronal es el índice de la neurona
        neurona de la última capa tenga la mayor activación."""
        resultados_prueba = [(np.argmax(self.propagacion_hacia_adelante(x)), y)
                        for (x, y) in datos_de_prueba]
        return sum(int(x == y) for (x, y) in resultados_prueba)

    def cost_derivative(self, salida_activaciones, y):

        """Devuelve el vector de derivadas parciales \partial C_x /
        \partial a para las activaciones de salida."""
        return (salida_activaciones-y)

#### Funciones varias


def sigmoide(z):
    """La función sigmoidea."""
    return 1.0/(1.0+np.exp(-z))


def sigmoide_derivada(z):
    """Derivada de la función sigmoidea."""
    return sigmoide(z)*(1-sigmoide(z))

#tomado de https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
