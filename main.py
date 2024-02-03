import cv2
import numpy as np
from multiprocessing import Process
import matplotlib.pyplot as plt

binaryMap = "2024-field_binary.png"

img = cv2.imread(binaryMap, cv2.IMREAD_GRAYSCALE)

img_config = {"top-left": [
      150,
      79
    ],
    "bottom-right": [
      2961,
      1476
    ], "field-size": [
    54.27083,
    26.9375
  ],}


s = (250,250)
e = (100,650)
discreteGrid = 5
scale = 0.25

img = cv2.resize(img,(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_AREA)

displayImg = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
displayImg[s] = (0,255,0)
displayImg[e] = (255,0,0)

refreshRate = 1 #HZ
refreshIndex = 0

print(img[s])

#uses the following alg 
#https://mat.uab.cat/~alseda/MasterOpt/AStar-Algorithm.pdf
#https://en.wikipedia.org/wiki/A*_search_algorithm

def feetToMeters(cord):
    newTuple = []
    for i in cord:
        newTuple.append(i /3.281) 
    return tuple(newTuple)

def dist(start, end):
    return np.power((start[0] - end[0])**2 + (start[1]-end[1])**2, 0.5)
    

def reconstructPath(came_from, current_node):
    print("HERE")
    total_path = [current_node]
    while current_node in came_from:
        current_node = came_from[current_node]
        total_path.append(current_node)
    return total_path
    
def getNeighboringTuples(node):
    neighboringTuples = []
    for i in [-discreteGrid,0,discreteGrid]:
        for j in [-discreteGrid,0,discreteGrid]:
            if i != 0 or j != 0:
                new_node = (node[0] + i, node[1] + j)
                if img[convertToImageCordinate(new_node)] == 0:
                    neighboringTuples.append(new_node)
                    displayImg[convertToImageCordinate(new_node)] = (100,100,100)
    return neighboringTuples

def convertToImageCordinate(cord):
    return (cord[1], cord[0])

def AStar(start, end):
    global displayImg
    refreshIndex = 0
    refreshRate = 10 #HZ

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
        neighboring_nodes = getNeighboringTuples(current_node)
        for neighbor in neighboring_nodes:
            tentative_gScore = gScore.get(current_node, 1e99) + dist(current_node, neighbor)
            if tentative_gScore < gScore.get(neighbor,1e99):
                came_from[neighbor] = current_node
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = tentative_gScore + dist(neighbor,end)
                if neighbor not in open_nodes:
                    open_nodes.append(neighbor)
                    displayImg[convertToImageCordinate(neighbor)] = (250,250,0)
        if True and refreshIndex > refreshRate:
            cv2.imshow('image', displayImg)
            cv2.waitKey(1)
            refreshIndex = 0
        refreshIndex = refreshIndex + 1

                

    return "FAILED"

def drawRobot():
    pass



def runAStar():
    print(AStar(s,e))

def showProcess():
    while True:
        cv2.imshow('image', displayImg)
        print(displayImg[(231, 150)])
        cv2.waitKey(10)
    cv2.destroyAllWindows()
        


fieldSize = img_config["field-size"]
fieldSizeMeters = feetToMeters(fieldSize)

topLeftFieldPixels = img_config["top-left"] #pixels
bottomRightFieldPixels = img_config["bottom-right"] #pixels

fieldInPixels = (bottomRightFieldPixels[0] - topLeftFieldPixels[0],bottomRightFieldPixels[1] - topLeftFieldPixels[1])

bottomLeftFieldMeters = ((fieldSizeMeters[0] * topLeftFieldPixels[0]/fieldInPixels[0]), (fieldSizeMeters[1] * bottomRightFieldPixels[1]/fieldInPixels[1]))


print(bottomLeftFieldMeters)

def convertRobotPoseToPixel(pos):
    print(bottomLeftFieldMeters)
    #use top left and right, as well as image size, to determine locations
    pos = (pos[0] + bottomLeftFieldMeters[0],  fieldSizeMeters[1] + (fieldSizeMeters[1] - (pos[1] +  bottomLeftFieldMeters[1])))
    pixelPosition = ( int(np.round(scale * fieldInPixels[0] *  pos[0]/fieldSizeMeters[0])), int(np.round(scale * fieldInPixels[1] * pos[1]/fieldSizeMeters[1])))
    return pixelPosition


if __name__ == "__main__":
    s = convertRobotPoseToPixel((1,1))
    e = convertRobotPoseToPixel((15,7))
    result = AStar(s,e)
    print(s,e)
    print(result)
    for i in range(len(result) -1):
        # displayImg[convertToImageCordinate(result[i])] = (255,0,255)
        line = cv2.line(displayImg, result[i], result[i+1], (255,0,255), 2)
    
    cv2.waitKey(1)
    cv2.imshow('image', displayImg)
    cv2.waitKey(0)