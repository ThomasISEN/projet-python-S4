import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.collections as mc
import pylab as pl # matplotlib module
import pyqtgraph as pg
import pyqtgraph.opengl as gl # 3D
from pyqtgraph.Qt import QtWidgets # 2D
import csv
import math
import os
import copy
import pyqtgraph.examples






"""
Fonction lire_fichier_csv:
    params : nom_fichier -> On passe en argument le nom du fichier à ouvrir 
    use : ouvre le fichier contenant les datas à utiliser

"""
def lire_fichier_csv(nom_fichier):
    donnees = []
    with open(nom_fichier, newline='') as fichier_csv:
        lecteur_csv = csv.reader(fichier_csv, delimiter=',', quotechar='"')
        for ligne in lecteur_csv:
            donnees.append(ligne)
    print("Fin de lecture du fichier")
    return donnees


"""
Fonction transormeLectureInTableauFloat:
    params : tableau_str -> On passe en argument le tableau (alors en string à cause de la lecture)
    use : Transforme les datas string en datas float exploitable pour l'algo 

"""

def transormeLectureInTableauFloat(tableau_str):
    print("Transformation en cours")
    nouveau_tableau_float = []
    for sous_tableau in tableau_str:
        nouveau_sous_tableau = []
        for valeur in sous_tableau:
            nouveau_sous_tableau.append(float(valeur)) #transformation de chaque valeur en float
        nouveau_tableau_float.append(nouveau_sous_tableau)  #Nouveau tableau
    return nouveau_tableau_float




"""
Fonction InitialisationCentroids:
    params : NbCentroides -> Le nombres de centroides que l'on veut dans l'algo (valeur par defaut = 4)
             version -> en fonction de l'algo en 2d ou en 3d 
    use : Choisit X points au hasard (et tous différent). Le nombre representera l'index d'un point dans le tableau de data

"""

def InitialisationCentroids(NbCentroides=4,version="1"): #Tire au sort les index dans le tableau des coordonées 
    coordonnes_centroids_index_tableau=[]
    TirageDuCentroide=0
    for i in range(0,NbCentroides,1):
        if(version=="1"):
            TirageDuCentroide=random.randint(0,400) #Algo mock
            while TirageDuCentroide in coordonnes_centroids_index_tableau:
                TirageDuCentroide=random.randint(0,400)

            coordonnes_centroids_index_tableau.append(TirageDuCentroide)
        elif(version=="2"):
            TirageDuCentroide=random.randint(0,60000) #version 3d
            while TirageDuCentroide in coordonnes_centroids_index_tableau:
                TirageDuCentroide=random.randint(0,60000)

            coordonnes_centroids_index_tableau.append(TirageDuCentroide)

            
    return coordonnes_centroids_index_tableau



"""
Fonction InitialisationCentroids:
    params : coordonnes_centroids_index -> Le tableau d'index de tous les cenoitrdes
             TableauFull -> le tableau de tous les datas
    use : Transforme le tableau d'index en tableau de coordonnés 

"""

def TransformeToCooFloat(coordonnes_centroids_index,TableauFull):
    print("Changement en float")
   
    coordonnes_centroids_float_tableau=[]
    for valeur in coordonnes_centroids_index:
        #print(valeur)
        #print(coordonnes_centroids_index[3])
        #print("La longueuer est "+ str(len(coordonnes_centroids_index)))
        coordonnes_centroids_float_tableau.append(TableauFull[valeur])
    return coordonnes_centroids_float_tableau






"""
Fonction draw2D:
    Old function

"""
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
    





"""
Fonction CalculDistance1Point:
    params : points -> Le tableau de toutes les datas
             AllCentroids -> le tableau des coordonnées des centroides
             version -> en fonction de l'algo en 2d ou en 3d 
    use : Renvoie l'index de la plus petite distance (represente donc un centroide), on calcule la distance d'un point par rapport à tous les centroides

"""

def CalculDistance1Point(point,AllCentroids,version="1"): #Version marche avec X centroides
    choix=[]
    for centroids in AllCentroids:
        firstSquare=math.pow(centroids[0]-point[0],2) #(x2 -x1)²
        SecondeSquare=math.pow(centroids[1]-point[1],2)
        
        if(version=="2"):
            ThirdSquare=math.pow(centroids[2]-point[2],2)
            calcul1=math.sqrt(firstSquare+SecondeSquare+ThirdSquare)
        else:
            calcul1=math.sqrt(firstSquare+SecondeSquare)
        choix.append(calcul1)
    
    return choix.index(min(choix))



"""
Fonction AssignationPointsToCluster:
    params : points -> Le tableau de toutes les datas
             AllCentroids -> le tableau des coordonnées des centroides
             version -> en fonction de l'algo en 2d ou en 3d 
    use : Assigne lors de la première itération tous les points au centroides le plus proche
"""
def AssignationPointsToCluster(points,TableauCentroids,version="1"):
    #print(len(points))

    for i in range(len(points)):

        ValeurAssignation=CalculDistance1Point(points[i],TableauCentroids) 

        points[i].append(TableauCentroids[ValeurAssignation][0])
        points[i].append(TableauCentroids[ValeurAssignation][1])
        if(version=="2"):
            points[i].append(TableauCentroids[ValeurAssignation][2])
        
    return points
       
        

"""
Fonction ReturnNbPointWithCoo:
    params : points -> Le tableau de toutes les datas
             AllCentroids -> le tableau des coordonnées des centroides
             version -> en fonction de l'algo en 2d ou en 3d 
    use : Permet de renvoiyer tous les calculs necessaires pour modifier la position des clusters (somme en X,Y,Z et le nombre de point assigné à chaque centroides)
"""
def ReturnNbPointWithCoo(points,AllCentroids,version="1"): 
    print("Voici les points : ")
    #print(points)
    #print(AllCentroids)
    Nb_Point=[]
    CooX_Sum=[]
    CooY_Sum=[]
    CooZ_Sum=[]
    for i in range(0,len(AllCentroids),1):
        Nb_Point.append(0)
        CooX_Sum.append(0)
        CooY_Sum.append(0)
        CooZ_Sum.append(0)
    
    for point in points:
        #print(Nb_Point)
        if(version=="1"):
            for i in range(0,len(AllCentroids),1):
                 if(point[2]==AllCentroids[i][0] and point[3]==AllCentroids[i][1] ):
                     #point 2 -> position x du centroids associé
        #                                                                         #point 3 -> position y du centroid associé
                    Nb_Point[i]=Nb_Point[i]+1
                    CooX_Sum[i]=CooX_Sum[i]+point[0]
                    CooY_Sum[i]=CooY_Sum[i]+point[1]
            
        else:

            for i in range(0,len(AllCentroids),1): #2=len de centroides

                 if(point[3]==AllCentroids[i][0] and point[4]==AllCentroids[i][1] and point[5]==AllCentroids[i][2] ):
                                                   #point 3 -> position y du centroid associé
                    Nb_Point[i]=Nb_Point[i]+1
                    CooX_Sum[i]=CooX_Sum[i]+point[0]
                    CooY_Sum[i]=CooY_Sum[i]+point[1]
                    CooZ_Sum[i]=CooZ_Sum[i]+point[2]

    return Nb_Point,CooX_Sum,CooY_Sum,CooZ_Sum
   
    

"""
Fonction drawV2:
   Fonction de dessin
"""

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


"""
Fonction CalculNewPosCentroids:
    params : points -> Le tableau de toutes les datas
             AllCentroids -> le tableau des coordonnées des centroides
             version -> en fonction de l'algo en 2d ou en 3d 
             Nb_Point -> Tableau contenant le nombre de point associé à chaque centroides
             SumCooX -> Tableau contenant la somme des X associé à chaque centroides
             SumCooY -> Tableau contenant la somme des Y associé à chaque centroides
             SumCooZ -> Tableau contenant la somme des Z associé à chaque centroides
    use : Recalcule la position de chaque centroides par rapport au arguments
"""

def CalculNewPosCentroids(points,AllCentroids,Nb_Point,SumCooX,SumCooY,SumCooZ,version="1"): #Marche avec X centroides
    #print("Calcul de la nouvelle position")
    
    new_centroids = []
    index = 0
    if(version=="1"):
        for centroid in AllCentroids:
            #print(Nb_Point)
            cnpx = (1 / Nb_Point[index]) * SumCooX[index]
            
            cnpY = (1 / Nb_Point[index]) * SumCooY[index]
            new_centroid = [cnpx, cnpY]
            new_centroids.append(new_centroid)
            index += 1
            
    else:
        for centroid in AllCentroids:
            #print(Nb_Point)
            cnpx = (1 / Nb_Point[index]) * SumCooX[index]
            
            cnpY = (1 / Nb_Point[index]) * SumCooY[index]
            cnpZ= (1 / Nb_Point[index]) * SumCooZ[index]
            new_centroid = [cnpx, cnpY,cnpZ]
            new_centroids.append(new_centroid)
            index += 1
            
    #print("Voici les NewCentroides : "+str(new_centroids))
    return new_centroids #Fonctionne avec X centroides




 # Pour chaque cluster k, le nouveau centroïde C_k est calculé en prenant la moyenne des positions des points appartenant à ce cluster :

##C_k = (1 / N_k) * somme(x_i) pour tous les x_i appartenant au cluster k

#Où N_k est le nombre de points appartenant au cluster k.
    
"""
Fonction ModifPosClusterForPoint:
    params : points -> Le tableau de toutes les datas
             AllCentroids -> le tableau des coordonnées des centroides
             version -> en fonction de l'algo en 2d ou en 3d 
    use : Permet de reassigner à chaque points son centroides le plus proche
"""
def ModifPosClusterForPoint(points,AllCentroids,version="1"):

    if(version=="1"):
        for i in range(len(points)):
            # Assignation de chaque point au cluster le plus proche
            ValeurAssignation=CalculDistance1Point(points[i],AllCentroids)
            #print(ValeurAssignation)
            #print(TableauCentroids[ValeurAssignation][0])
            points[i][2]=AllCentroids[ValeurAssignation][0]
            points[i][3]=AllCentroids[ValeurAssignation][1]
    else:
        for i in range(len(points)):
            # Assignation de chaque point au cluster le plus proche
            ValeurAssignation=CalculDistance1Point(points[i],AllCentroids,"2")
            #print(ValeurAssignation)
            #print(TableauCentroids[ValeurAssignation][0])
            points[i][3]=AllCentroids[ValeurAssignation][0]
            points[i][4]=AllCentroids[ValeurAssignation][1]
            points[i][5]=AllCentroids[ValeurAssignation][2]
            #print(points[i])
    return points
    




"""
Fonction IterationAlgo:
    d
    use : Boucle principal permettant de faire tourner
"""

def IterationAlgo():
    TableauDistanceMoyenne=[]
    nombreIteration=100
    for i in range(0,nombreIteration,1):

        if(i==0):
            print("Initialisation de l'algo ")
            print("Veuillez choisir la version de l'application :")
            print("1. Version 2D")
            print("2. Version 3D")

           # version = input("Votre choix : ")
            version="2"
            if version == "1":
                print("Vous avez choisi la version 2D.")
                test=lire_fichier_csv("2d_data.csv") #Lecture des données
                test=test[1:]
                 
                test=transormeLectureInTableauFloat(test) #Transformation en float (working in 3d)
                print("Avant transformation float")
            # print(test)
               # print("Avant init des centroids")
                coordonnes_centroids_index_tableau=InitialisationCentroids(10,"1") #Initialisation des centroids (working in 3d)
                
                print(coordonnes_centroids_index_tableau)
                coordonnes_centroids_float_tableau=TransformeToCooFloat(coordonnes_centroids_index_tableau,test) #Float des centroids
                print("Apres la transformation en float des centroids")
                print(coordonnes_centroids_float_tableau)
            # ValeurIndex=CalculDistance1Point(test[2], coordonnes_centroids_float_tableau) #Cette ligne permet d'assigner un point à un centroid
            # print(coordonnes_centroids_float_tableau)
                #print("Association des points aux centroids ")
                test=AssignationPointsToCluster(test,coordonnes_centroids_float_tableau) #On assigne tous les points à un cluster (not working in 3d)[seulement 2 coo]
               # print(test)
                #print("Après associaiton des points au centroids")
                #print(test)
                #print("Normalement avant iteration  :")
               # print(coordonnes_centroids_float_tableau)
                #print(test)
                drawV2(test,1920)
            elif version == "2": #Version 3D
                print("Vous avez choisi la version 3D.")
                test=lire_fichier_csv("3d_data.csv") #Lecture des données
                test=test[1:]
                test=transormeLectureInTableauFloat(test) #Transformation en float (working in 3d)
                print("Avant transformation float")
            # print(test)
                print("Avant init des centroids")
                coordonnes_centroids_index_tableau=InitialisationCentroids(10,"2") #Initialisation des centroids (working in 3d)
                
                print(coordonnes_centroids_index_tableau)
                coordonnes_centroids_float_tableau=TransformeToCooFloat(coordonnes_centroids_index_tableau,test) #Float des centroids
                print("Apres la transformation en float des centroids")
                print(coordonnes_centroids_float_tableau)
            # ValeurIndex=CalculDistance1Point(test[2], coordonnes_centroids_float_tableau) #Cette ligne permet d'assigner un point à un centroid
            # print(coordonnes_centroids_float_tableau)
                print("Association des points aux centroids ")
                test=AssignationPointsToCluster(test,coordonnes_centroids_float_tableau,version) #On assigne tous les points à un cluster (modif pour 3d)
                #print(test)
                #test = [[0,0,0,0.5,0.5,0.5],[1,1,1,0.5,0.5,0.5]]
                
                #pyqtgraph.examples.run()
                #pyqtgraph.examples.run()
                #testb = [[2,2,2,2.5,2.5,2.5],[3,3,3,2.5,2.5,2.5]]
                #drawV2(test) #Init fonctionne avec X clusters
                #pyqtgraph.examples.run()
                #drawV2(test) #Init fonctionne avec X clusters
                #pyqtgraph.examples.run()
                
                
            else:
                print("Choix invalide.")
           
        else:
            if(version=="1"):
                print("Algo en cours ")
                CoupageDuRenvoie=ReturnNbPointWithCoo(test,coordonnes_centroids_float_tableau) 
                Nb_point=CoupageDuRenvoie[0]
                SumCooX=CoupageDuRenvoie[1]
                SumCooY=CoupageDuRenvoie[2]

                coordonnes_centroids_float_tableau=CalculNewPosCentroids(test,coordonnes_centroids_float_tableau,Nb_point,SumCooX,SumCooY,0) #On recalcule les positions 
                #des clusters
                print("Normalement apres iteration  :"+str(coordonnes_centroids_float_tableau))



                test=ModifPosClusterForPoint(test,coordonnes_centroids_float_tableau)  #On re-associe chaque points à un centroids (nouvelle position)
                #print(coordonnes_centroids_float_tableau)
               
                drawV2(test,1920)
                ResultatSomme=EvalutionQualite(test,coordonnes_centroids_float_tableau)
                TableauDistanceMoyenne.append(ResultatSomme)
            else:
                print("Algo en 3d")
                #print(test)
               
                CoupageDuRenvoie=ReturnNbPointWithCoo(test,coordonnes_centroids_float_tableau,"2")
                Nb_point=CoupageDuRenvoie[0]
                SumCooX=CoupageDuRenvoie[1]
                SumCooY=CoupageDuRenvoie[2]
                SumCooZ=CoupageDuRenvoie[3] #Fonctionnel avec X centroides
                #print("Fin du premier tour de l'algo ") 
                #-----------------------------------------------------------------------------------------------------------
                coordonnes_centroids_float_tableau=CalculNewPosCentroids(test,coordonnes_centroids_float_tableau,Nb_point,SumCooX,SumCooY,SumCooZ,"2")
                # print("Coordonées des centroids à l'itération "+str(i)+" :"+str(coordonnes_centroids_float_tableau))
                # print(" ")
                # print(" ")
                # print(" ")
            
                #Bon jusqu'à la pour la 3d -----------------------------------------------------------------------------------------

                test=ModifPosClusterForPoint(test,coordonnes_centroids_float_tableau,"2")
                
                #print("Fin d'algo en 3d")
              
                #drawV2(test,1920)
                ResultatSomme=EvalutionQualite(test,coordonnes_centroids_float_tableau,"2")
                TableauDistanceMoyenne.append(ResultatSomme)
    drawV2(test,1920)
    GraphiqueEvolutionScore(TableauDistanceMoyenne,nombreIteration)
    SauvegardeDesPoints(test,ResultatSomme,version)
    #Avant d'écrire il faut d'abord lire la donnée de la première ligne pour savoir si la somme est plus petite ou pas (Meilleur score)
    

"""
Fonction SauvegardeDesPoints:
    params : points -> Le tableau de toutes les datas
             LastResultat -> la valeur écrite dans le fichier representant l'ancien score
             version -> en fonction de l'algo en 2d ou en 3d 
    use : Permet de sauvegarder les datas et leurs centroides seulement si le score est plus petit que celui deja ecrit dans le fichier
"""

def SauvegardeDesPoints(points,LastResultat,version="1"):
    if(version=="1"):
        ValeurAncienneSomme=read_first_line_csv("EcritureDonnees")
        if(float(ValeurAncienneSomme[1])<LastResultat):
            print("cas ou (ValeurAncienneSomme[1]<ResultatSomme (on ecrit pas)")
        else:
            print("Ecriture en cours ")
            write_csv_file("EcritureDonnees",points,LastResultat) #Fin d'algo , on écrit dans le csv
    else:
        ValeurAncienneSomme=read_first_line_csv("EcritureDonnees3d")
        if(float(ValeurAncienneSomme[1])<LastResultat):
            print("cas ou (ValeurAncienneSomme[1]<ResultatSomme (on ecrit pas)")
        else:
            print("Ecriture en cours ")
            write_csv_file("EcritureDonnees3d",points,LastResultat)

                    
"""
Fonction EvalutionQualite:
    params : points -> Le tableau de toutes les datas
             AllCentroids -> le tableau des coordonnées des centroides
             version -> en fonction de l'algo en 2d ou en 3d 
    use : Permet de calcul un "score" pour savoir si l'algo a bien fonctionner ou pas
"""

def EvalutionQualite(points,AllCentroids,version="1"):
    NbPoint=len(points)
    ResultatSomme=0
    CalculDistanceEuclidienne=0
    print("Evaluation en cours ")
    for point in points:
        #print(point)
        #print(AllCentroids)
        for i in range(0,len(AllCentroids),1):
            if(version=="1"):
                if(point[2]==AllCentroids[i][0] and point[3]==AllCentroids[i][1]):
                    CalculDistanceEuclidienne=math.pow((point[0]-AllCentroids[i][0]),2)+math.pow((point[1]-AllCentroids[i][1]),2)
                    CalculDistanceEuclidienne=math.sqrt(CalculDistanceEuclidienne)
                ResultatSomme+=CalculDistanceEuclidienne
            else:
                if(point[3]==AllCentroids[i][0] and point[4]==AllCentroids[i][1] and point[5]==AllCentroids[i][2]):
                    CalculDistanceEuclidienne=math.pow((point[0]-AllCentroids[i][0]),2)+math.pow((point[1]-AllCentroids[i][1]),2)+math.pow((point[2]-AllCentroids[i][2]),2)
                    CalculDistanceEuclidienne=math.sqrt(CalculDistanceEuclidienne)
                ResultatSomme+=CalculDistanceEuclidienne
        #ResultatSomme+=CalculDistanceEuclidienne
    print("Fin de calcul !!!!!!!!!!!!!!!")
    print(ResultatSomme)
   
    return ResultatSomme



"""
Fonction write_csv_file:
    params : NomFichier -> Le nom du fichier ou l'on veut écrire
             data -> Le tableau de données 
             ValeurSomme -> Le dernier score calculé (et donc normalement le plus petit)
    use : Ecrit dans le fichier pour sauvegarder les datas
"""

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

"""
Fonction write_csv_file:
    params : NomFichier -> Le nom du fichier ou l'on veut écrire

    use : Lit seulement la première ligne du fichier de sauvegarde des datas pour récuperer l'ancien score
"""

def read_first_line_csv(NomFichier):
    with open(NomFichier, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        first_row = next(csvreader)
        print(first_row)
        return first_row

"""
Fonction write_csv_file:
    params : TableauScore -> Le tableau contenant tous les scores pendants les itérations
             NombreIteration -> le nombre d'itérations fait par l'algo dans la boucle principal

    use : Fait un petit graphique pour voir l'évolution du score à la fin de la partie
"""
def GraphiqueEvolutionScore(TableauScore,NombreIteration):
    print("Affichage du graphique")
    print(TableauScore)
    x=[]
    for i in range(0,NombreIteration-1,1):
        x.append(i)
    
    plt.plot(x, TableauScore)
    plt.show()
    plt.close()




IterationAlgo()

