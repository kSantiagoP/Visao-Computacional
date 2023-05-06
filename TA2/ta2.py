import numpy as np
import cv2 as cv
import sys

numImg = 0

def exibirImagem(img):
    cv.imshow("Imagem Carregada", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def salvarImagem(img):
    global numImg
    numImg += 1
    cv.imwrite("img"+str(numImg)+".jpg",img)



oPoints = [] 
iPoints = [] 
criterio = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objectP = np.zeros((6*7,3), np.float32)
objectP[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

img = cv.imread("image.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
encontrado, corners = cv.findChessboardCorners(gray, (7,6), None)
if encontrado == True:
    oPoints.append(objectP)
    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criterio)
    iPoints.append(corners2)
    cv.drawChessboardCorners(img, (7,6), corners2, encontrado)
    exibirImagem(img)
    salvarImagem(img)

retorno, matrix, distortion, rotVet, transVet = cv.calibrateCamera(oPoints, iPoints, gray.shape[::-1], None, None)

print(retorno)
print(matrix)

imagem = cv.imread("image.png")
altura, largura  = imagem.shape[:2]
matrixNova, regionInterest = cv.getOptimalNewCameraMatrix(matrix, distortion, (largura,altura),1, (largura,altura))
print(matrixNova, regionInterest)

novaDist = cv.undistort(imagem, matrix, distortion, None, matrixNova)
x, y, largura, altura = regionInterest
novaDist = novaDist[y:y+altura, x:x+largura]
cv.imwrite("resultado.png",novaDist)

calibrada = cv.imread("resultado.png")
exibirImagem(calibrada)