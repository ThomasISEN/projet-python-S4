import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.collections as mc
import pylab as pl # matplotlib module
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets # 2D
import csv
import math
import os







def main_fct():
    print("Compilation done")
    


main_fct()


def lire_fichier_csv(nom_fichier):
    donnees = []
    with open(nom_fichier, newline='') as fichier_csv:
        lecteur_csv = csv.reader(fichier_csv, delimiter=',', quotechar='"')
        for ligne in lecteur_csv:
            donnees.append(ligne)
    print("Fin de lecture du fichier")
    return donnees




def transormeLectureInTableauFloat(tableau_str):
    print("Transformation en cours")
    nouveau_tableau_float = []
    for sous_tableau in tableau_str:
        nouveau_sous_tableau = []
        for valeur in sous_tableau:
            nouveau_sous_tableau.append(float(valeur))
        nouveau_tableau_float.append(nouveau_sous_tableau)
    return nouveau_tableau_float



def trouver_max(tableau):
    max_val = -math.inf
    test=0
    for sous_tableau in tableau:
        for valeur in sous_tableau:
            if(test!=0):
                
                if (float(valeur) > max_val) :
                    max_val = valeur
        test=1
    return max_val

def trouver_min(tableau):
    min_val = math.inf
    test=0
    for sous_tableau in tableau:
        for valeur in sous_tableau:
            if(test!=0):
                
                if (float(valeur) < min_val) :
                    min_val = valeur
        test=1
    return min_val




def InitialisationCentroids(): #Tire au sort les index dans le tableau des coordonées 
    Centroide1=random.randint(0, 400)
    Centroide2=random.randint(0, 400)
    Centroide3=random.randint(0, 400)
    Centroide4=random.randint(0, 400)
    print("Les centroides au choisie les coordonees de tableau : "+str(Centroide1)+" "+str(Centroide2)+" "+str(Centroide3)+" "+str(Centroide4))
    coordonnes_centroids_index_tableau=[]
    coordonnes_centroids_index_tableau.append(Centroide1)
    coordonnes_centroids_index_tableau.append(Centroide2)
    coordonnes_centroids_index_tableau.append(Centroide3)
    coordonnes_centroids_index_tableau.append(Centroide4)
    return coordonnes_centroids_index_tableau


def TransformeToCooFloat(coordonnes_centroids_index,TableauFull):
    print("Changement en float")
   
    coordonnes_centroids_float_tableau=[]
    for valeur in coordonnes_centroids_index:
        #print(valeur)
        #print(coordonnes_centroids_index[3])
        #print("La longueuer est "+ str(len(coordonnes_centroids_index)))
        coordonnes_centroids_float_tableau.append(TableauFull[valeur])
    return coordonnes_centroids_float_tableau






#sample = le tableau de co
def draw2D(samples, size=10, drawLinks=True):
	# Formatting the data:
	X, Y, links, centroids = [], [], [], set()
	for sample in samples:
		X.append(sample[0])
		Y.append(sample[1])
		if len(sample) == 4:
			links.append([sample[:2], sample[2:]])
			centroids.add((sample[2], sample[3]))
	centroids = sorted(centroids) # before shuffling, to not depend on data order.
	random.seed(42) # to have consistent results.
	random.shuffle(centroids) # making less likely that close clusters have close colors.
	centroids = { cent : centroids.index(cent) for cent in centroids }
	# Colors map:
	colors = cm.rainbow(np.linspace(0, 1., len(centroids)))
	C = None # unique color!
	if len(centroids) > 0:
		C = [ colors[centroids[(sample[2], sample[3])]] for sample in samples ]
	# Drawing:
	fig, ax = pl.subplots(figsize=(size, size))
	fig.suptitle('Visualisation de %d données' % len(samples), fontsize=16)
	ax.set_xlabel('x', fontsize=12)
	ax.set_ylabel('y', fontsize=12)
	if drawLinks:
		ax.add_collection(mc.LineCollection(links, colors=C, alpha=0.1, linewidths=1))
	ax.scatter(X, Y, c=C, alpha=0.5, s=10)
	for cent in centroids:
		ax.plot(cent[0], cent[1], c='black', marker='+', markersize=8)
	ax.autoscale()
	ax.margins(0.05)

  
	plt.show()
    




#AssignationPointsToCluster(test,coordonnes_centroids_float_tableau)


def CalculDistance1Point(point,AllCentroids):
    choix=[]
    for centroids in AllCentroids:
        firstSquare=math.pow(centroids[0]-point[0],2) #(x2 -x1)²
        
        #print(("Premiere distance "+str(firstSquare)))
        SecondeSquare=math.pow(centroids[1]-point[1],2)
        #print("Seconde distance "+str(SecondeSquare))

        calcul1=math.sqrt(firstSquare+SecondeSquare)
        choix.append(calcul1)
    #print(choix)
    #print(min(choix))
    #print(choix.index(min(choix)))
    return choix.index(min(choix))



def AssignationPointsToCluster(points,TableauCentroids):
    #print(len(points))

    for i in range(len(points)):
        # Assignation de chaque point au cluster le plus proche
        ValeurAssignation=CalculDistance1Point(points[i],TableauCentroids)
        #print(ValeurAssignation)
        #print(TableauCentroids[ValeurAssignation][0])
        points[i].append(TableauCentroids[ValeurAssignation][0])
        points[i].append(TableauCentroids[ValeurAssignation][1])
        
    return points
       
        


def ReturnNbPointWithCoo(points,AllCentroids):
    #print("Voici les points : ")
    #print(points)
    Nb_Point=[0,0,0,0]
    CooX_Sum=[0.0,0.0,0.0,0.0]
    CooY_Sum=[0.0,0.0,0.0,0.0]
    for point in points:
        #print(point)
        #print(AllCentroids)
        if(point[2]==AllCentroids[0][0] and point[3]==AllCentroids[0][1] ): #point 2 -> position x du centroids associé
                                                                            #point 3 -> position y du centroid associé
            Nb_Point[0]=Nb_Point[0]+1
            CooX_Sum[0]=CooX_Sum[0]+point[0]
            CooY_Sum[0]=CooY_Sum[0]+point[1]
        elif(point[2]==AllCentroids[1][0] and point[3]==AllCentroids[1][1] ):
            Nb_Point[1]=Nb_Point[1]+1
            CooX_Sum[1]=CooX_Sum[1]+point[0]
            CooY_Sum[1]=CooY_Sum[1]+point[1]
        elif(point[2]==AllCentroids[2][0] and point[3]==AllCentroids[2][1] ):
            Nb_Point[2]=Nb_Point[2]+1
            CooX_Sum[2]=CooX_Sum[2]+point[0]
            CooY_Sum[2]=CooY_Sum[2]+point[1]
        elif(point[2]==AllCentroids[3][0] and point[3]==AllCentroids[3][1] ):
            Nb_Point[3]=Nb_Point[3]+1
            CooX_Sum[3]=CooX_Sum[3]+point[0]
            CooY_Sum[3]=CooY_Sum[3]+point[1]

    #print(Nb_Point)
    #print(CooX_Sum)
    #print(CooY_Sum)
    return Nb_Point,CooX_Sum,CooY_Sum



def drawV2(samples, windowSize=1000, offset=(0, 0, 0)):
	random.seed(42)
	dimMap = { 2: 2, 3: 3, 4: 2, 6: 3 }
	assert len(samples) > 0, "Received 0 samples."
	assert len(samples[0]) in dimMap, "Unsupported samples size."
	dim = dimMap[len(samples[0])]
	createCoord = lambda c : { "pos": c } if dim == 2 else c

	# Grouping samples in clusters for faster rendering.
	# Note: clusters number defaults to 1 if no centroids are in the data.
	centroidsMap, spotsList = {}, []
	for c in samples:
		centroid = tuple(c[dim:])
		if centroid not in centroidsMap:
			centroidsMap[centroid] = len(centroidsMap)
			spotsList.append([])
		spotsList[centroidsMap[centroid]].append(createCoord(c[:dim]))
	colormap = [ pg.intColor(i, hues=len(centroidsMap), alpha=150) for i in range(len(centroidsMap)) ]
	random.shuffle(colormap) # so close clusters are less likely to have close colors.

	# Adding centroids, if present:
	if () not in centroidsMap:
		spotsList.append([ createCoord(c) for c in centroidsMap ])
		colormap.append((255, 255, 255, 255))

	# Creating a graphical context:
	app = pg.mkQApp("PyQtGraph app")
	if dim == 2:
		w = QtWidgets.QMainWindow()
		view = pg.GraphicsLayoutWidget()
		w.setCentralWidget(view)
		p = view.addPlot()
	else:
		w = gl.GLViewWidget()
		w.setCameraPosition(distance=20.)
		g = gl.GLGridItem()
		w.addItem(g)
	w.setWindowTitle("Clustering data")
	w.resize(windowSize, windowSize)

	# Drawing:
	for i in range(len(spotsList)):
		if dim == 2:
			p.addItem(pg.ScatterPlotItem(spots=spotsList[i], brush=colormap[i], size=10., pxMode=True))
		else:
			s = gl.GLScatterPlotItem(pos=spotsList[i], color=colormap[i], size=10., pxMode=True)
			s.translate(*offset)
			if i < len(spotsList)-1:
				s.setGLOptions("translucent")
			w.addItem(s)
	w.show()
	pg.exec()


def CalculNewPosCentroids(points,AllCentroids,Nb_Point,SumCooX,SumCooY):
    #print("Calcul de la nouvelle position")
    new_centroids = []
    index = 0
    
    for centroid in AllCentroids:
        #print(Nb_Point)
        cnpx = (1 / Nb_Point[index]) * SumCooX[index]
        
        cnpY = (1 / Nb_Point[index]) * SumCooY[index]
        new_centroid = [cnpx, cnpY]
        new_centroids.append(new_centroid)
        index += 1
    return new_centroids




 # Pour chaque cluster k, le nouveau centroïde C_k est calculé en prenant la moyenne des positions des points appartenant à ce cluster :

##C_k = (1 / N_k) * somme(x_i) pour tous les x_i appartenant au cluster k

#Où N_k est le nombre de points appartenant au cluster k.
    
def ModifPosClusterForPoint(points,AllCentroids):

    for i in range(len(points)):
        # Assignation de chaque point au cluster le plus proche
        ValeurAssignation=CalculDistance1Point(points[i],AllCentroids)
        #print(ValeurAssignation)
        #print(TableauCentroids[ValeurAssignation][0])
        points[i][2]=AllCentroids[ValeurAssignation][0]
        points[i][3]=AllCentroids[ValeurAssignation][1]
    return points
    






def IterationAlgo():
   
    for i in range(0,4,1):

        if(i==0):
            print("Initialisation de l'algo ")
            test=lire_fichier_csv("mock_2d_data.csv") #Lecture des données
            test=test[1:]
            test=transormeLectureInTableauFloat(test) #Transformation en float
            coordonnes_centroids_index_tableau=InitialisationCentroids() #Initialisation des centroids
            print(coordonnes_centroids_index_tableau)
            coordonnes_centroids_float_tableau=TransformeToCooFloat(coordonnes_centroids_index_tableau,test) #Float des centroids
            print(coordonnes_centroids_float_tableau)
            ValeurIndex=CalculDistance1Point(test[2], coordonnes_centroids_float_tableau) #Cette ligne permet d'assigner un point à un centroid
            print(coordonnes_centroids_float_tableau)

            test=AssignationPointsToCluster(test,coordonnes_centroids_float_tableau) #On assigne tous les points à un cluster
            #print(test)
            print("Normalement avant iteration  :")
            print(coordonnes_centroids_float_tableau)
            #print(test)
            drawV2(test,1920)
        else:
            print("Algo en cours ")
            CoupageDuRenvoie=ReturnNbPointWithCoo(test,coordonnes_centroids_float_tableau)
            Nb_point=CoupageDuRenvoie[0]
            SumCooX=CoupageDuRenvoie[1]
            SumCooY=CoupageDuRenvoie[2]

            coordonnes_centroids_float_tableau=CalculNewPosCentroids(test,coordonnes_centroids_float_tableau,Nb_point,SumCooX,SumCooY) #On recalcule les positions 
            #des clusters
            print("Normalement apres iteration  :"+str(coordonnes_centroids_float_tableau))



            test=ModifPosClusterForPoint(test,coordonnes_centroids_float_tableau)  #On re-associe chaque points à un centroids (nouvelle position)
            #print(coordonnes_centroids_float_tableau)
            drawV2(test,1920)
            ResultatSomme=EvalutionQualite(test,coordonnes_centroids_float_tableau)

    #Avant d'écrire il faut d'abord lire la donnée de la première ligne pour savoir si la somme est plus petite ou pas (Meilleur score)
    ValeurAncienneSomme=read_first_line_csv("EcritureDonnees")
    if(float(ValeurAncienneSomme[1])<ResultatSomme):
        print("cas ou (ValeurAncienneSomme[1]<ResultatSomme (on ecrit pas)")
    else:
        print("Ecriture en cours ")
        write_csv_file("EcritureDonnees",test,ResultatSomme) #Fin d'algo , on écrit dans le csv
                




def EvalutionQualite(points,AllCentroids):
    NbPoint=len(points)
    ResultatSomme=0
    CalculDistanceEuclidienne=0
    print("Evaluation en cours ")
    for point in points:
        #print(point)
        #print(AllCentroids)
        if(point[2]==AllCentroids[0][0] and point[3]==AllCentroids[0][1] ): #point 2 -> position x du centroids associé
                                                                            #point 3 -> position y du centroid associé
            
            CalculDistanceEuclidienne=math.pow((point[0]-AllCentroids[0][0]),2)+math.pow((point[1]-AllCentroids[0][1]),2)
            CalculDistanceEuclidienne=math.sqrt(CalculDistanceEuclidienne)
        elif(point[2]==AllCentroids[1][0] and point[3]==AllCentroids[1][1] ):
            
            CalculDistanceEuclidienne=math.pow((point[0]-AllCentroids[1][0]),2)+math.pow((point[1]-AllCentroids[1][1]),2)
            CalculDistanceEuclidienne=math.sqrt(CalculDistanceEuclidienne)
        elif(point[2]==AllCentroids[2][0] and point[3]==AllCentroids[2][1] ):
            CalculDistanceEuclidienne=math.pow((point[0]-AllCentroids[2][0]),2)+math.pow((point[1]-AllCentroids[2][1]),2)
            CalculDistanceEuclidienne=math.sqrt(CalculDistanceEuclidienne)
          
        elif(point[2]==AllCentroids[3][0] and point[3]==AllCentroids[3][1] ):
            CalculDistanceEuclidienne=math.pow((point[0]-AllCentroids[3][0]),2)+math.pow((point[1]-AllCentroids[3][1]),2)
            CalculDistanceEuclidienne=math.sqrt(CalculDistanceEuclidienne)
           
        ResultatSomme+=CalculDistanceEuclidienne
    print("Fin de calcul !!!!!!!!!!!!!!!")
    print(ResultatSomme)
   
    return ResultatSomme




def write_csv_file(NomFichier, data,ValeurSomme):
    # Si le fichier n'existe pas, on le crée

    with open(NomFichier, 'w', newline='') as csvfile:
        Ecriture = csv.writer(csvfile)
        construction=['x','y','centroid_x','centroid_x']
        construction2=['Best Valeur for ValeurSomme =',ValeurSomme]
        Ecriture.writerow(construction2)
        Ecriture.writerow(construction)
        for points in data:
            Ecriture.writerow(points)
   
def read_first_line_csv(NomFichier):
    with open(NomFichier, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        first_row = next(csvreader)
        print(first_row)
        return first_row

IterationAlgo()