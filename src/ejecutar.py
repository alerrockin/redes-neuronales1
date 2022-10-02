import mnist_loader
import network
datos_de_entrenamiento,  datos_de_validación,  datos_de_prueba = mnist_loader.cargar_datos_tupla()
red = network.Network([784, 30, 10])
red.SGD(datos_de_entrenamiento, 30, 10, 3.0, datos_de_prueba=datos_de_prueba)
