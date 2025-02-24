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
    print(coordenadas)
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


# obtener los nodos del arbol de expansión mínima que tienen grado impar
def nodos_impares(arbol_expansion_minima, size):
    nodos = {}
    nodos_i = []
    for i in range(size):
        nodos[i+1] = 0
    for i in arbol_expansion_minima:
        for j in i:
            nodos[j] = nodos.get(j)+1
    for i in nodos:
        if nodos[i] % 2 != 0:
            nodos_i.append(i)
    return nodos_i


def nodos_peso_mínimo(matriz, nodos_prueba):
    '''
    Encuentra los nodos con peso mínimo y crea una lista'''
    nodos_prueba = []
    index = 0
    for i in matriz:
        mask = i > 0
        if np.any(mask):
            min_val = np.min(i[mask])
            tupla = np.where(i == min_val)[0]
            for j in range(tupla.shape[0]):
                nodos_prueba.append(np.array([index, tupla[j]]))
        index += 1
    return nodos_prueba


# llena la matriz de ceros
def convertir_fila_columna_cero(matriz, fila, columna):
    matriz[fila, :] = 0  # Convertir toda la fila en ceros
    matriz[:, columna] = 0  # Convertir toda la columna en ceros
    return matriz


# encontrar las aristas perfectas con costo mínimo, llama a las funciones nodos_peso_mínimo y convertir_fila_columna_cero
def distancia_minima(matriz):
    '''
    Con los nodos de peso mínimo que se obtienen de la función nodos_peso_mínimo, en esta se busca por cada loop se va reduciendo la matriz
    al llenarla de ceros para que no se rompa la regla que no deben existir aristas adyacentes y cumplir la condición de aristas perfectas
    cada combinación se guarda en un diccionario con la llave como la suma de los pesos y el valor las aristas perfectas, se pasa por todos
    los nodos de peso mínimo para encontrar todas las permutaciones posibles, al ser una matriz par, siempre se van a encontrar
    arístas perfectas, se va probado todas las permutaciones que se obtuvieron de los nodos de costo mínimo y se agregan a un diccionario
    donde la llave es el costo total de la combinación de nodos y el valor es la combinación de nodos, luego se extrae las tuplas que tengan
    la llave de menor costo.
    '''
    nodos = {}
    nodos_minimo = []
    nodos_p = nodos_peso_mínimo(matriz, nodos)
    # print("Nodos peso mínimo: ",nodos_p)
    while len(nodos_p) != 0:
        nodos_minimo_n = []
        per = matriz.shape[0]
        matriz_reducida = matriz.copy()
        # print("Matriz original: \n",per)
        # print(matriz_reducida,"\n")
        int = 0
        nodos_p_2 = nodos_p
        while per > 0:
            # print("int: ",int)
            nodos_p_3 = [nodos_p_2[0][0], nodos_p_2[0][1]]
            # print("nodos: ",nodos_p_3)
            matriz_reducida = convertir_fila_columna_cero(
                matriz_reducida, nodos_p_2[0][0], nodos_p_2[0][1])
            matriz_reducida = convertir_fila_columna_cero(
                matriz_reducida, nodos_p_2[0][1], nodos_p_2[0][0])
            nodos_minimo_n.append(nodos_p_2[0]+1)
            # print(matriz_reducida)
            nodos_p_2 = nodos_peso_mínimo(matriz_reducida, nodos)
            int = int+1
            per = per-2
        nodos_minimo.append(nodos_minimo_n)
        nodos_p.pop(0)
       # print("Nodos mínimos: ",nodos_minimo)

        for i in nodos_minimo:
            suma = 0
            for j in range(len(i)):
                suma += matriz[i[j][0] - 1, i[j][1] - 1]
            nodos[suma] = tuple(i)
           # print(suma, " : ", tuple(i))
        # print("\n")
    # print(nodos)
    llave_min = min(nodos.keys())
    # print(llave_min)
    nodos = {llave_min: nodos[llave_min]}
    return nodos


def multigrafo(arbol_exp_min, aristas_perfectas):
    """Encuentra el multigrafo que combina el árbol de expansión mínima del primer paso y las aristas perfectas del segundo paso """
    multigrafo = nx.Multigraph(
        arbol_exp_min)  # Crear multigrafo con las aristas del árbol de expansión mínima
    # Añade las aristas perfectas al multigrafo
    multigrafo.add_edges_from(aristas_perfectas)
    return multigrafo


def encontrar_ciclo_euleriano(multigrafo):
    """Encuentra el ciclo euleriano en el multigrafo que combina el árbol de expansión mínima y las aristas perfectas"""
    ciclo_euleriano = list(nx.eulerian_circuit(
        multigrafo))  # Halla el ciclo euleriano del multigrafo
    return ciclo_euleriano


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
        dibujar_grafo_matriz_adyacencia(
            matriz_distancias, "Matriz de distancias")

        # Primer paso algoritmo Christofides (Encontrar un arbol de expansión minima)

        arbol_expansion_minima, _ = encontrar_arbol_expansion_minima(
            matriz_distancias, 0
        )

        print("\nArbol de expansión minima encontrado:")
        print(arbol_expansion_minima)
        dibujar_grafo_lista_adyacencia(
            arbol_expansion_minima, "Arbol de expansión minima"
        )

        # Segundo paso algoritmo Christofides encontrar perfect matching minimum weight

        nodos_i = nodos_impares(arbol_expansion_minima,
                                int(coordenadas.size/2))
        coordenadas_i = []
        for i in nodos_i:
            coordenadas_i.append(coordenadas[i-1])
        matriz_distancias_i = calcular_matriz_distancias(coordenadas_i)
        print("Nodos I:", nodos_i)
        # print(coordenadas_i)
        # print(matriz_distancias_i)
        dibujar_grafo_matriz_adyacencia(
            matriz_distancias_i, "Matriz de distancias")
        nodos = distancia_minima(matriz_distancias_i)
        # print(nodos)
        valores_lista = []
        for i in nodos.values():
            for j in i:
                valores_lista.append(j)
        # Nodos i: [1, 2, 3,  4,  5,  6,  7,  8,  9, 10, 11, 12]
        # Nodos I: [6, 7, 8, 11, 12, 19, 20, 21, 26, 29, 37, 38]
        lista_pmmw = []
        for i in valores_lista:
            lista_pmmw.append(np.array([nodos_i[i[0]-1], nodos_i[i[1]-1]]))
        print("Aristas perfectas con costo mínimo:\n", lista_pmmw)

        # Tercer paso algoritmo Christofides (Juntar el árbol de expansión mínima y las aristas perfectas)

        multigrafo_arbol_min_aristas_perf = multigrafo(
            arbol_expansion_minima, lista_pmmw)
        print("\nMultigrafo con aristas del arbol de expansión mínima y aristas perfectas creado:")
        print(multigrafo_arbol_min_aristas_perf)
        dibujar_grafo_lista_adyacencia(
            multigrafo, "Multigrafo del árbol de expansión mínima más aristas perfectas")

        # Cuarto paso algoritmo Christofides (Hallar el ciclo euleriano en el multigrafo)

        ciclo_euleriano = encontrar_ciclo_euleriano(
            multigrafo_arbol_min_aristas_perf)
        print("\nCiclo euleriano encontrado:")
        print(ciclo_euleriano)
        dibujar_grafo_lista_adyacencia(ciclo_euleriano, "Ciclo Euleriano")

        # Quinto paso algoritmo Christofides

    else:
        print("Opción no válida. Intente de nuevo.")


if __name__ == "__main__":
    main()
