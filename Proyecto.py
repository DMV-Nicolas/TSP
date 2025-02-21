import os
import numpy as np

def leer_tsplib(nombre_archivo):
    """Lee un archivo TSPLib y devuelve una lista de coordenadas."""
    with open(nombre_archivo, 'r') as f:
        lines = f.readlines()

    coordenadas = []
    leer_nodos = False  # Bandera para saber cuándo empezar a leer nodos

    for line in lines:
        line = line.strip()  # Eliminar espacios en blanco y saltos de línea

        if line.startswith("NODE_COORD_SECTION"):
            leer_nodos = True  # Iniciar lectura de nodos
            continue
        elif line.startswith("EOF"):
            break  # Fin del archivo

        if leer_nodos:
            partes = line.split()
            if len(partes) == 3:  # ID, X, Y
                _, x, y = map(float, partes)
                coordenadas.append((x, y))

    return np.array(coordenadas)

def calcular_matriz_distancias(coordenadas):
    """Calcula la matriz de distancias euclidianas entre los nodos."""
    num_nodos = len(coordenadas)
    matriz = np.zeros((num_nodos, num_nodos))

    for i in range(num_nodos):
        for j in range(num_nodos):
            if i != j:
                matriz[i][j] = np.linalg.norm(coordenadas[i] - coordenadas[j])

    return matriz

# ---- Listar archivos automáticamente ----
# Buscar archivos .tsp en la carpeta
archivos_disponibles = [f for f in os.listdir() if f.endswith(".tsp")]

if not archivos_disponibles:
    print("No se encontraron archivos .tsp en la carpeta.")
    exit()

# Mostrar opciones disponibles
print("Seleccione un archivo:")
for i, archivo in enumerate(archivos_disponibles, start=1):
    print(f"{i}. {archivo}")

# Pedir al usuario que seleccione un numero
opcion = input("Ingrese el número del archivo: ")

# Validar la opcion ingresada
if opcion.isdigit() and 1 <= int(opcion) <= len(archivos_disponibles):
    archivo_seleccionado = archivos_disponibles[int(opcion) - 1]
    print(f"\nCargando archivo: {archivo_seleccionado} ...")

    coordenadas = leer_tsplib(archivo_seleccionado)
    matriz_distancias = calcular_matriz_distancias(coordenadas)

    print("\nMatriz de distancias generada:")
    print(matriz_distancias)
else:
    print("Opción no válida. Intente de nuevo.")
