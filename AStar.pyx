"""Generic AStar algorithm (general structure implemented from pseudo code from two sources below)
    https://en.wikipedia.org/wiki/A*_search_algorithm#:~:text=%5B11%5D-,Pseudocode,-%5Bedit%5D
    https://mat.uab.cat/~alseda/MasterOpt/AStar-Algorithm.pdf
    Albeit slightly modified to work with images
"""
import numpy as np

print("WARNING RUNNING PYHON")

class AStarOptions():
    """Stores all options for A* Algorithm
    """
    lattice_length = None  # distance between each neighboring pixel
    # kernel/mask of sorts, contains all offsets to easily find neighboring pixels
    neighboring_offsets = None

    def __init__(self, lattice_length):
        self.updateLatticeGrid(lattice_length)

    def updateLatticeGrid(self, latticeLength: int):
        """Updates the distance between each latice points/pixels to scan
            If laticeLength = 1, every pixel will be scanned
            If laticeLength = 2, every other pixel will be scanned (checkerboard pattern)
        Args:
            laticeLength (int): distance between neighboring points
        """
        self.lattice_length = latticeLength
        self.neighboring_offsets = [(i, j) for i in [-latticeLength, 0, latticeLength]
                                    for j in [-latticeLength, 0, latticeLength] if i != 0 or j != 0]


def _dist(start, end):
    """Cost function is euclidian norm 2

    Args:
        start ((int,int)): current cordinate position
        end ((int,int)): final cordinate position

    Returns:
        _type_: euclidian distance between both points
    """
    return np.linalg.norm(np.subtract(end, start))


def _reconstructPath(came_from, current_node):
    total_path = [current_node]
    while current_node in came_from:
        current_node = came_from[current_node]
        total_path.append(current_node)
    return total_path


def _getNeighboringTuples(node, binary_image, options: AStarOptions):
    """Gets next set of cordinates to look through

    Args:
        node (numpy [y,x]): gets current robot position
        binaryImage ([int,int,int]): image of where the robot can go, a pixel being 0/black = free space, a pixel being 1/white = blocked

    Returns:
        (x,y): neighboring pixels in image cordinates
    """
    neighboring_tuples = []
    for offset in options.neighboring_offsets:
        new_node = (node[0] + offset[0], node[1] + offset[1])
        # if it is empty, don't try looking at areas where the robot can't go
        if binary_image[_convertToImageCordinate(new_node)] == 0:
            neighboring_tuples.append(new_node)
    return neighboring_tuples

# not really needed but helps with intuition (and converts to tuple!)


def _convertToImageCordinate(cord):
    """Conversion between numpy array indicies and standard image pixel cord in a tuple (x,y)

    Args:
        cord ([y,x]): numpy array index

    Returns:
        (x,y): image cordinates
    """
    return (cord[1], cord[0])

# see links in file header to understand the algorithm


def execute(self, start, end, binary_image, options: AStarOptions):
    """Runs the A* Algorithm, based off of the two documents in the file docstring

    Args:
        start ((int,int)): Starting pixel position
        end ((int,int)): Target/end goal pixel position
        binary_image (np array (int,int,int (0-255))): image to run algorithm on. Must be grayscale or binary (0 or 1)
        options (AStarOptions): algorithm options (lattice grid)

    Returns:
        [(x,y)] | str: array of tuples that links start to end, OR "Failed" if a path couldn't be found
    """
    open_nodes = [start]
    came_from = {}
    gScore = {}  # gscore[n] is the cost of a path from the start to node n
    gScore[start] = 0
    # fscore[n] = gscore[n] + dist(n,end) -> best guess on the cost of a path to the end
    fScore = {}
    fScore[start] = _dist(start, end)

    while open_nodes != []:
        current_node = min(open_nodes, key=lambda y: fScore.get(y, 1e99))
        # ends algorithm when we are in distance of the target
        if _dist(current_node, end) < options.lattice_length:
            return _reconstructPath(came_from, current_node)
        open_nodes.remove(current_node)
        neighboring_nodes = _getNeighboringTuples(
            current_node, binary_image, options)
        for neighbor in neighboring_nodes:
            tentative_gScore = gScore.get(
                current_node, 1e99) + _dist(current_node, neighbor)
            if tentative_gScore < gScore.get(neighbor, 1e99):
                came_from[neighbor] = current_node
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = tentative_gScore + _dist(neighbor, end)
                if neighbor not in open_nodes:
                    open_nodes.append(neighbor)

    return "FAILED"
