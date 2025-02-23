import os
import numpy as np


def dibujar_grafo_lista_adyacencia(lista_adyacencia, titulo="Grafo"):
    """Dibuja un grafo a partir de una lista de adyacencia."""

    from networkx import Graph, draw
    from matplotlib.pyplot import figure, title, show

    # Crear un grafo vacío
    G = Graph()

    # Agregar las aristas desde la lista de adyacencia
    G.add_edges_from(lista_adyacencia)

    # Dibujar el grafo
    figure(figsize=(8, 6))
    draw(
        G,
        with_labels=True,
        node_color="lightblue",
        node_size=200,
        font_size=12,
        font_weight="bold",
        edge_color="gray",
    )
    title(titulo)
    show()


def dibujar_grafo_matriz_adyacencia(matriz_adyacencia, titulo="Grafo"):
    """
    Dibuja un grafo a partir de una matriz de adyacencia.
    - Los nodos se numeran de 1 a n
    - el grafo dibujado no esta representando las distancias entre cada nodo
    """

    from networkx import Graph, draw, spring_layout
    from matplotlib.pyplot import figure, title, show

    # Crear un grafo vacío
    G = Graph()

    # Número de nodos
    n = len(matriz_adyacencia)

    # Agregar las aristas con sus pesos
    for i in range(n):
        for j in range(i + 1, n):
            if matriz_adyacencia[i][j] != 0:
                G.add_edge(i + 1, j + 1, weight=matriz_adyacencia[i][j])

    # Dibujar el grafo
    figure(figsize=(8, 6))
    pos = spring_layout(G)  # Posición de los nodos
    draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=200,
        font_size=12,
        font_weight="bold",
        edge_color="gray",
    )

    title(titulo)
    show()


def leer_tsplib(nombre_archivo):
    """Lee un archivo TSPLib y devuelve una lista de coordenadas."""
    with open(nombre_archivo, "r") as f:
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


def encontrar_arbol_expansion_minima(w, s):
    """Encuentra un arbol de expansión minima"""
    n = len(w)
    v = []
    while len(v) != n:
        v.append(0)
    v[s] = 1
    E = []
    suma = 0
    for i in range(0, n - 1):
        minimo = 1000  # Valor muy grande M
        agregar_vertice = 0
        e = []
        for j in range(0, n):
            if v[j] == 1:
                for k in range(0, n):
                    if v[k] == 0 and w[j][k] < minimo:
                        agregar_vertice = k
                        e = [j, k]
                        minimo = w[j][k]
        suma += w[e[0]][e[1]]
        v[agregar_vertice] = 1
        E.append(e)
    E2 = []
    for i in E:
        E2.append([i[0] + 1, i[1] + 1])
    return [E2, suma]


def main():
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
        dibujar_grafo_matriz_adyacencia(matriz_distancias, "Matriz de distancias")

        # Primer paso algoritmo Christofides (Encontrar un arbol de expansión minima)

        arbol_expansion_minima, _ = encontrar_arbol_expansion_minima(
            matriz_distancias, 0
        )

        print("\nArbol de expansión minima encontrado:")
        print(arbol_expansion_minima)
        dibujar_grafo_lista_adyacencia(
            arbol_expansion_minima, "Arbol de expansión minima"
        )

        # Segundo paso algoritmo Christofides

        # Tercer paso algoritmo Christofides

        # Cuarto paso algoritmo Christofides

        # Quinto paso algoritmo Christofides

    else:
        print("Opción no válida. Intente de nuevo.")


if __name__ == "__main__":
    main()
