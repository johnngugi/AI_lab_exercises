from queue import PriorityQueue

import matplotlib.pyplot as plt
import networkx as nx

from ordered_node import OrderedNode


def draw_graph(g, path):
    node_positions = nx.get_node_attributes(g, 'pos')
    edge_length = nx.get_edge_attributes(g, 'length')

    peru_colored_edges = list(zip(path, path[1:]))
    node_col = ['darkturquoise' if node not in path else 'peru' for node in g.nodes()]
    edge_col = ['darkturquoise' if edge not in peru_colored_edges else 'peru' for edge in g.edges()]

    nx.draw_networkx(g, node_positions, node_color=node_col, with_labels=True,
                     labels={node: node for node in g.nodes()})
    nx.draw_networkx_edges(g, node_positions, edge_color=edge_col)
    nx.draw_networkx_edge_labels(g, node_positions, edge_labels=edge_length, edge_color=edge_col)

    plt.axis('off')
    plt.show()


def get_greedy_bfs_queue(nodes_list, heuristics):
    result = []
    for node in nodes_list:
        result.append(OrderedNode(heuristics[node], node))
    return result


def greedy_bfs(G, source, destination, heuristics):
    if destination not in G.nodes():
        print("Destination not in network")

    start_node = G[source]
    final_path = [source]
    adjacent = start_node.keys()
    node_queue = get_greedy_bfs_queue(adjacent, heuristics)
    priority_queue = get_priority_queue(node_queue)
    visited = {source: True}

    if source == destination:
        print("Arrived")
        return final_path

    while True:
        current = priority_queue.get()
        print(current)
        if current.description == destination:
            break
        final_path.append(current.description)
        visited[current.description] = True
        adjacent = G[current.description].keys()
        neighbours = []
        for node in adjacent:
            if not visited.get(node):
                neighbours.append(OrderedNode(heuristics[node], node))
        priority_queue = get_priority_queue(neighbours)

    final_path.append(destination)
    print("Final path: ", final_path)
    return final_path


def get_a_star_queue(nodes_list, heuristics, accumulator):
    result = []
    for node in nodes_list:
        result.append(OrderedNode(heuristics[node] + accumulator, node))
    return result


def a_star(G, source, destination, heuristics):
    if destination not in G.nodes():
        print("Destination not in network")

    accumulator = 0
    visited = {source: True}
    start_node = G[source]
    final_path = [source]
    weights = get_a_star_node_list(accumulator, heuristics, start_node, visited)
    optimal = get_priority_queue(weights)

    if source == destination:
        print("Arrived")
        return final_path

    while True:
        current = optimal.get()
        accumulator += current.priority
        if current.description == destination:
            break
        final_path.append(current.description)
        visited[current.description] = True
        neighbours = get_a_star_node_list(accumulator, heuristics, G[current.description], visited)
        optimal = get_priority_queue(neighbours)

    final_path.append(destination)
    print("Final path: ", final_path)
    return final_path


def get_a_star_node_list(accumulator, heuristics, node, visited):
    weights = []
    for i in node.keys():
        if not visited.get(i):
            cumulative = float(node[i]['length']) + accumulator
            weights.append(OrderedNode(cumulative + float(heuristics[i]), i))
    return weights


def create_graph():
    g = nx.Graph()
    f = open('input.txt')

    while True:
        line = f.readline().split()
        if not line or line is '':
            break
        g.add_edge(line[0], line[1], length=line[2])

    return g


def get_heuristics(g):
    heuristics = {}
    f = open('heuristics.txt')
    for i in g.nodes():
        node_heuristic_val = f.readline().split()
        if not node_heuristic_val or node_heuristic_val is '':
            break
        heuristics[node_heuristic_val[0]] = node_heuristic_val[1]
    return heuristics


def get_priority_queue(nodes_list):
    q = PriorityQueue()
    for node in nodes_list:
        q.put(node)
    return q


def assign_positions(g):
    g.nodes["Karen"]['pos'] = (0, 0)
    g.nodes["Gitaru"]['pos'] = (-1, 3)
    g.nodes["J1"]['pos'] = (2, -2)
    g.nodes["J2"]['pos'] = (3, -4)
    g.nodes["Langata"]['pos'] = (3, -6)
    g.nodes["J3"]['pos'] = (5, -5)
    g.nodes["J4"]['pos'] = (4, -2)
    g.nodes["J5"]['pos'] = (6, -2)
    g.nodes["J6"]['pos'] = (0, 2)
    g.nodes["J7"]['pos'] = (0, 4)
    g.nodes["J8"]['pos'] = (2, 4)
    g.nodes["Loresho"]['pos'] = (2, 6)
    g.nodes["J9"]['pos'] = (4, 4)
    g.nodes["Lavington"]['pos'] = (4, 2)
    g.nodes["J10"]['pos'] = (6, 4)
    g.nodes["Parklands"]['pos'] = (8, 6)
    g.nodes["J11"]['pos'] = (8, 2)
    g.nodes["Kilimani"]['pos'] = (8, 1)
    g.nodes["J12"]['pos'] = (10, 0)
    g.nodes["CBD"]['pos'] = (12, 0)
    g.nodes["J13"]['pos'] = (12, -4)
    g.nodes["ImaraDaima"]['pos'] = (14, -6)
    g.nodes["Donholm"]['pos'] = (14, 2)
    g.nodes["HillView"]['pos'] = (14, 3)
    g.nodes["Kasarani"]['pos'] = (14, 4)
    g.nodes["Kahawa"]['pos'] = (15, 6)


def main():
    g = create_graph()
    heuristics = get_heuristics(g)
    assign_positions(g)
    # final_path = greedy_bfs(g, 'Karen', 'Karen', heuristics)
    final_path = a_star(g, 'Karen', 'ImaraDaima', heuristics)
    draw_graph(g, final_path)


if __name__ == "__main__":
    main()
