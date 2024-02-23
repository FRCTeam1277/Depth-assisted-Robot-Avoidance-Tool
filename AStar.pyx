import numpy as np
discreteGrid = 5

def dist(start, end):
    return np.power((start[0] - end[0])**2 + (start[1]-end[1])**2, 0.5)
    

def reconstructPath(came_from, current_node):
    print("HERE")
    total_path = [current_node]
    while current_node in came_from:
        current_node = came_from[current_node]
        total_path.append(current_node)
    return total_path
    
def getNeighboringTuples(node, binaryImage):
    neighboringTuples = []
    for i in [-discreteGrid,0,discreteGrid]:
        for j in [-discreteGrid,0,discreteGrid]:
            if i != 0 or j != 0:
                new_node = (node[0] + i, node[1] + j)
                if binaryImage[convertToImageCordinate(new_node)] == 0:
                    neighboringTuples.append(new_node)
    return neighboringTuples

def convertToImageCordinate(cord):
    return (cord[1], cord[0])


def AStar(start, end, binaryImg):
    binaryImage = binaryImg
    open_nodes = [start]
    came_from = {}
    gScore = {}
    gScore[start] = 0
    fScore = {}
    fScore[start] = dist(start, end)
    
    while open_nodes != []:
        current_node = min(open_nodes, key = lambda y: fScore.get(y,1e99))
        if dist(current_node, end) < discreteGrid:
            return reconstructPath(came_from, current_node)
        open_nodes.remove(current_node)
        neighboring_nodes = getNeighboringTuples(current_node, binaryImage)
        for neighbor in neighboring_nodes:
            tentative_gScore = gScore.get(current_node, 1e99) + dist(current_node, neighbor)
            if tentative_gScore < gScore.get(neighbor,1e99):
                came_from[neighbor] = current_node
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = tentative_gScore + dist(neighbor,end)
                if neighbor not in open_nodes:
                    open_nodes.append(neighbor)


    return "FAILED"