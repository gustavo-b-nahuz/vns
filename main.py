import itertools
import random
import networkx as nx
import matplotlib.pyplot as plt
import math
import json


def plot_graph(graph):
    """Visualiza o grafo com pesos nas arestas."""
    pos = nx.spring_layout(graph, seed=42)  # Define a posição dos nós
    plt.figure(figsize=(8, 6))

    # Desenha os nós e arestas
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=700,
        font_weight="bold",
    )

    # Adiciona os pesos nas arestas
    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.title("Grafo de Entrada")
    plt.show()


def euclidean_distance(x1, y1, x2, y2):
    """Calcula a distância Euclidiana entre dois pontos 2D."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def read_input_from_file(filename):
    """Reads the input for the graph from a file containing vertex positions."""
    with open(filename, "r") as file:
        lines = file.readlines()

    num_vertices = len(lines)
    positions = {}
    edges = []

    # Ler posições dos vértices
    for line in lines:
        parts = line.split()
        node_id = int(parts[0]) - 1  # Ajustar para índice base 0
        x, y = float(parts[1]), float(parts[2])
        positions[node_id] = (x, y)

    # Criar arestas completamente conectadas
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            dist = euclidean_distance(
                positions[i][0], positions[i][1], positions[j][0], positions[j][1]
            )
            edges.append((i, j, dist))
            edges.append((j, i, dist))  # Grafo não direcionado

    return num_vertices, edges


def initialize_solution(graph, num_vertices, p, coverage_radius):
    """Creates an initial solution using a greedy heuristic."""
    covered = set()
    solution = []
    for _ in range(p):
        best_node = None
        best_coverage = 0
        for node in range(num_vertices):
            if node in solution:
                continue
            coverage = set(
                neighbor
                for neighbor in graph.neighbors(node)
                if graph[node][neighbor]["weight"] <= coverage_radius
                and neighbor not in covered
            )
            if len(coverage) > best_coverage:
                best_node = node
                best_coverage = len(coverage)

        if best_node is None:
            remaining = [node for node in range(num_vertices) if node not in solution]
            if not remaining:
                break
            best_node = remaining[0]

        solution.append(best_node)
        covered.update(
            neighbor
            for neighbor in graph.neighbors(best_node)
            if graph[best_node][neighbor]["weight"] <= coverage_radius
        )
        covered.add(best_node)

    return solution


def tsp_nearest_insertion(graph, nodes):
    """Solves the TSP using the nearest insertion heuristic."""
    if len(nodes) <= 1:
        print("Quantidade de nós era menor ou igual a um")
        return nodes, 0

    nodes_copy = nodes[:]
    tour = [nodes_copy.pop(0)]  # Inicia o tour com o primeiro nó

    while nodes_copy:
        nearest_node = None
        nearest_distance = float("inf")
        insert_position = 0

        for i in range(len(tour)):
            for node in nodes_copy:
                dist = graph[tour[i]][node]["weight"]
                if dist < nearest_distance:
                    nearest_node = node
                    nearest_distance = dist
                    insert_position = i

        nodes_copy.remove(nearest_node)
        tour.insert(insert_position + 1, nearest_node)

    total_distance = sum(
        graph[tour[i]][tour[i + 1]]["weight"] for i in range(len(tour) - 1)
    )
    total_distance += graph[tour[-1]][tour[0]]["weight"]  # Retorna ao início

    return tour, total_distance


def calculate_objective(alpha, tour_distance, coverage):
    """Calcula a função objetivo combinada."""
    return alpha * tour_distance - (1 - alpha) * coverage


def calculate_coverage(graph, solution, coverage_radius):
    """Calcula a cobertura total de uma solução."""
    covered = set()
    for node in solution:
        covered.update(
            neighbor
            for neighbor in graph.neighbors(node)
            if graph[node][neighbor]["weight"] <= coverage_radius
        )
        covered.add(node)
    return len(covered)


def local_search(graph, solution, coverage_radius, alpha):
    """Busca local: melhora a solução trocando vértices por seus vizinhos."""
    improvement = True
    best_solution = solution[:]
    best_tour, best_distance = tsp_nearest_insertion(graph, best_solution)
    best_coverage = calculate_coverage(graph, best_solution, coverage_radius)
    best_objective = calculate_objective(alpha, best_distance, best_coverage)

    while improvement:
        improvement = False
        for i in range(len(best_solution)):
            for neighbor in graph.neighbors(best_solution[i]):
                if neighbor not in best_solution:
                    new_solution = best_solution[:]
                    new_solution[i] = neighbor
                    new_tour, new_distance = tsp_nearest_insertion(graph, new_solution)
                    new_coverage = calculate_coverage(
                        graph, new_solution, coverage_radius
                    )
                    new_objective = calculate_objective(
                        alpha, new_distance, new_coverage
                    )

                    if new_objective < best_objective:
                        best_solution = new_solution[:]
                        best_tour = new_tour[:]
                        best_distance = new_distance
                        best_coverage = new_coverage
                        best_objective = new_objective
                        improvement = True
    return best_solution, best_tour, best_distance, best_coverage, best_objective


def read_parameters_from_json(filename):
    """Lê os parâmetros do algoritmo a partir de um arquivo JSON."""
    with open(filename, "r") as file:
        return json.load(file)


def main():
    num_vertices, edges = read_input_from_file("in2.txt")

    graph = nx.Graph()
    graph.add_nodes_from(range(num_vertices))
    graph.add_weighted_edges_from(edges)
    # plot_graph(graph)

    params = read_parameters_from_json("params.json")
    p = params["p"]
    coverage_radius = params["coverage_radius"]
    max_iterations = params["max_iterations"]
    alpha = params["alpha"]
    k_min = 1
    k_max = p

    solution = initialize_solution(graph, num_vertices, p, coverage_radius)
    best_solution = solution[:]
    best_tour, best_distance = tsp_nearest_insertion(graph, best_solution)
    best_coverage = calculate_coverage(graph, best_solution, coverage_radius)
    best_objective = calculate_objective(alpha, best_distance, best_coverage)

    print(
        f"Solução inicial: {best_solution}, Distância inicial: {best_distance}, Cobertura inicial: {best_coverage}, Objetivo inicial: {best_objective}"
    )

    for iteration in range(1, max_iterations + 1):
        k = k_min
        while k <= k_max:
            # print(
            #     f"Iteração {iteration}, k={k}, vértices={best_solution}, melhor objetivo={best_objective}"
            # )
            perturbed_solution = best_solution[:]
            for _ in range(k):
                remove_node = random.choice(perturbed_solution)
                perturbed_solution.remove(remove_node)
                candidates = [n for n in graph.nodes if n not in perturbed_solution]
                if candidates:
                    new_node = random.choice(candidates)
                    perturbed_solution.append(new_node)
            # print(f"Vértices modificados após shuffle: {perturbed_solution}")
            perturbed_tour, perturbed_distance = tsp_nearest_insertion(
                graph, perturbed_solution
            )

            perturbed_coverage = calculate_coverage(
                graph, perturbed_solution, coverage_radius
            )
            # print(
            #     f"Tour, distancia e cobertura após shuffle: {perturbed_tour}, {perturbed_distance}, {perturbed_coverage}"
            # )
            perturbed_objective = calculate_objective(
                alpha, perturbed_distance, perturbed_coverage
            )

            if perturbed_objective < best_objective:
                best_solution = perturbed_solution[:]
                best_tour = perturbed_tour[:]
                best_distance = perturbed_distance
                best_coverage = perturbed_coverage
                best_objective = perturbed_objective
                k = k_min
            else:
                k += 1

            print(
                f"Iteração {iteration}, k={k}: Melhor solução {best_solution}, Distância {best_distance}, Cobertura {best_coverage}, Objetivo {best_objective}"
            )

    print(f"Solução final: {best_solution}")
    print(f"Tour final: {best_tour}")
    print(f"Cobertura final: {best_coverage}")
    print(f"Objetivo final: {best_objective}")


if __name__ == "__main__":
    main()
