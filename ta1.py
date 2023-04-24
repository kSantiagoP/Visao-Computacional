#A principal biblioteca de apoio a OpenCV é numpy, no Python
#uma vez que manipular imagem é, grosseiramente, manipular grandes matrizes
import numpy as np
import cv2 as cv
import sys

numImg = 0

def exibirImagem(img):
    cv.imshow("Imagem Carregada", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    #salvarImagem(img)


def salvarImagem(img):
    global numImg
    numImg += 1
    cv.imwrite("img"+str(numImg)+".jpg",img)

#Primeiramente, para trabalharmos uma imagem, precisamos carregá-la

img = cv.imread("flores.jpg")
assert img is not None, "Imagem não pôde ser lida."

exibirImagem(img)

#Dessa forma podemos exibir a imagem sendo editada.
#Como visto em sala de aula, podemos aplicar filtros sobre as imagens, que
#são conjuntos de matrizes contendo, para cada posição de pixeis, as intensidades
#das cores azul, verde e vermelha, para imagens coloridas, ou a intensidade do branco
# de zero até 255 para imagens em escala de cinza. Podemos abrir uma imagem em escala de
#cinza, por exemplo.

imgGray = cv.imread("flores.jpg", cv.IMREAD_GRAYSCALE)
exibirImagem(imgGray)

#Para aplicarmos filtros, a forma mais comum envolve a criação de kernels, matrizes núcleo
#que servem para a definição de pontos centrais a partir dos pontos nos arredores.
#Podemos aplicar filtros para o fortalecimento de bordas.

median = np.ones((5,5),np.float32)/25 #kernel 5x5 mediano
base = np.float32([[0,0,0,0,0],[0,0,0,0,0],[0,0,2,0,0],[0,0,0,0,0],[0,0,0,0,0]])
sharpenedGray = cv.filter2D(imgGray,-1,base - median)
sharpenedColor = cv.filter2D(img,-1,base - median)
negative = img - [255,255,255]
negative = np.uint8(np.absolute(negative))

exibirImagem(negative)
exibirImagem(sharpenedGray)
exibirImagem(sharpenedColor)

#Evidentemente, esse realce das bordas trás consigo a perda de informação da imagem
#o que significa perda de definição da imagem. Mas o OpenCV permite outras opções no tocante
# a visão computacional. Podemos aplicar as operações de Threshold para separar bordas.

laplacian = cv.Laplacian(imgGray,cv.CV_64F) #Para aplicar threshold, precisamos empregar a imagem em escala cinza

exibirImagem(laplacian) 

#Como observado, existe ruido na imagem sobre a qual realizamos threshold laplaciano

edges = cv.Canny(imgGray,75,200)
exibirImagem(edges)

#A detecção de bordas se torna muito melhor ao empregarmos o método Canny. Isso acontece pois, embutido nele, existe
#a aplicação de filtro gaussiano para fins de remoção de ruído. Além disso, é empregada histeresia para que gradientes
#que não tenham sido detectados como bordas mas adjacentes a gradientes que passem no critério sejam mantidos como borda.
