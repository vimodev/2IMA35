import math


class Component:
    def __init__(self, i):
        self.leader = i
        self.vertices = {i}
        self.edges = set()

    def set_leader(self, leader):
        self.leader = leader

    def __str__(self):
        return f"[Component #{self.leader}]"

    def get_cheapest_neighbour(self, p_dm):
        best_cost = math.inf
        edge = None
        for vertex in self.vertices:
            for target in range(len(p_dm[0])):
                if target not in self.vertices:
                    cost = p_dm[vertex][target]
                    if best_cost > cost:
                        best_cost = cost
                        edge = (vertex, target)
        return edge

    def merge_with_best(self, p_components, p_dm):
        best_edge = self.get_cheapest_neighbour(p_dm)

        s = best_edge[0]
        t = best_edge[1]
        self.edges.add((s, t, p_dm[s][t]))

        for component in p_components:
            if t in p_components[component].vertices:
                other_component = p_components[component]
                self.vertices = self.vertices.union(other_component.vertices)
                self.edges = self.edges.union(other_component.edges)
                remove_component(p_components, other_component.leader)
                return


def create_component(p_components, vertex):
    p_components[vertex] = Component(vertex)


def remove_component(p_components, vertex):
    p_components.pop(vertex, None)


def change_component_leader(p_components, old_vertex, new_vertex):
    temp = p_components[old_vertex]
    temp.set_leader(new_vertex)
    remove_component(p_components, old_vertex)
    p_components[new_vertex] = temp
