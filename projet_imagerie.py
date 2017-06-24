import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Debut du programme")
print("------------------------------------------")
print("Realise par Raphael IANCU - Florent COULON")
print("------------ 4A UFA 2017 ESIEA -----------")
print("------------------------------------------")

print("Lecture de l image")
img = cv2.imread("piece.png", cv2.IMREAD_GRAYSCALE)
count = 0

xc, yc, r = 348, 317, 45
H, W = img.shape
x, y = np.meshgrid(np.arange(W), np.arange(H))
d2 = (x - xc)**2 + (y - yc)**2
mask = d2 < r**2

outside = np.ma.masked_where(mask, img)

circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, dp=1.5, minDist=30, minRadius=15, maxRadius=60)

red = (0,0,255)
print("Detection des pieces")
for x, y, r in circles[0]:
    cv2.circle(outside, (x,y), r, red, 2)
	
black = np.zeros(img.shape)
print("Effacement des pieces")
for x, y, r in circles[0]:
	cv2.circle(black, (x,y), int(r+15), 255, -1)
	count += 1
    

print("Remplacement du vide par le fond")
bytemask = np.asarray(black, dtype=np.uint8)
inpainted = cv2.inpaint(img, bytemask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

print("------------------------------------------")
print("Il y a %i pieces dans l image" % count)
print("------------------------------------------")

print("Creation de 3 fichiers pour montrer les etapes du programme en image")
cv2.imwrite('piece_NB.png', outside)
cv2.imwrite('piece_detectee.png', black)
cv2.imwrite('piece_disparue.png', inpainted)