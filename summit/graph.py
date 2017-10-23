'''
This module defines Nodes and Graphs used in the TextRank algorithm
'''


class Node(object):
    '''
    Representation of a vertex in a graph
    '''

    def __init__(self, node):
        self.__id = node
        self.__adjacent = {}

    def add_neighbor(self, neighbor, weight=1):
        '''
        Add a neighbor node to this nodes adjacency list
        '''
        self.__adjacent[neighbor] = weight

    def del_neighbor(self, neighbor):
        '''
        Remove a node from this nodes adjacency list
        '''
        del self.__adjacent[neighbor]

    def get_neighbors(self):
        '''
        Query a list of all neighbors in  this nodes adjacency list
        '''
        return self.__adjacent.keys()

    def get_id(self):
        '''
        Return this node's id
        '''
        return self.__id

    def get_weight(self, neighbor):
        '''
        Get the weight of the edge between this node and a specified neighbor if it exists
        '''
        if neighbor in self.__adjacent.keys():
            return self.__adjacent[neighbor]
        else:
            return 0


class Graph(object):
    '''
    Representation of an undirected, weighted graph
    '''

    def __init__(self):
        self.__nodes = {}

    def add_node(self, node):
        '''
        Add a node to the graph
        '''
        self.__nodes[node] = Node(node)

    def has_node(self, node):
        '''
        Query if a node exists in the graph
        '''
        return node in self.__nodes

    def get_node(self, node):
        '''
        Get a Node object from teh graph
        '''
        if self.has_node(node):
            return self.__nodes[node]

    def del_node(self, node):
        '''
        Remove a node from the graph
        '''
        if node in self.__nodes.keys():
            for other in self.__nodes[node].get_neighbors():
                self.del_edge((node, other))
            del self.__nodes[node]

    def add_edge(self, edge, weight=1):
        '''
        Add an edge between two nodes with a specified weight
        '''
        u, v = edge
        if self.has_node(u) and self.has_node(v):
            self.__nodes[u].add_neighbor(v, weight)
            self.__nodes[v].add_neighbor(u, weight)
        else:
            raise ValueError(str.format(
                "Edge ({0},{1}) cannot exist because one or both nodes are not in the graph.", u, v,))

    def del_edge(self, edge):
        '''
        Remove an edge from the graph
        '''
        if self.has_edge(edge):
            u, v = edge
            self.__nodes[u].del_neighbor(v)
            self.__nodes[v].del_neighbor(u)

    def has_edge(self, edge):
        '''
        Query the existence of the specified edge in the graph
        '''
        u, v = edge
        return v in self.__nodes[u].get_neighbors() and u in self.__nodes[v].get_neighbors()

    def get_edge_weight(self, edge):
        '''
        Query the weight of the specified edge
        '''
        if self.has_edge(edge):
            u, v = edge
            return self.__nodes[u].get_weight(v)
        else:
            raise ValueError(str.format(
                "Edge ({0},{1}) does not exist.", u, v,))

    def get_nodes(self):
        '''
        Return a listing of all nodes
        '''
        return self.__nodes.keys()
