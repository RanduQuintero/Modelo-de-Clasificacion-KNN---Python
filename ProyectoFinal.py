#Bibliotecas necesarias para la creacion del algoritmo
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import cv2
import matplotlib.pyplot as plt 

#Cargar los datos de entrenamiento del archivo rsTraining.dat a txt
Datos= np.loadtxt(r"C:\Users\Randu\Desktop\KNN - Python\rsTrain.dat")
#Revolver los datos
np.random.shuffle(Datos)

#Todos los datos para el modelo final
DatosTotales_X=[]
DatosTotales_Y=[]
for j in Datos:
	DatosTotales_Y.append(j[4])
	DatosTotales_X.append(j[0:4])

DatosEnt=[]
DatosExp=[]
#Seleccion de los datos de entrenamiento y explotacion
for i in range (0,150):
	DatosEnt.append(Datos[i])

for i in range(150,200):
	DatosExp.append(Datos[i])

#Separamos los datos en las caracteristicas y las etiquetas
DatoEnt_X=[]
DatoEnt_Y=[]
#Datos para el entrenamiento
for i in DatosEnt:
	DatoEnt_Y.append(i[4])
	DatoEnt_X.append(i[0:4])

DatoExp_X=[]
DatoExp_Y=[]
#Datos para la explotacion
for j in DatosExp:
	DatoExp_Y.append(j[4])
	DatoExp_X.append(j[0:4])


#Cargar las imagenes binarias y especificamos el tamaño de la imagen
#Esto para cada una de las imagenes (4)

#Banda 1
Imagen1=np.fromfile(r"C:\Users\Randu\Desktop\KNN - Python\band1.irs",dtype=np.int8)
Banda1=np.reshape(Imagen1,(512,512))
#Banda 2
Imagen2=np.fromfile(r"C:\Users\Randu\Desktop\KNN - Python\band2.irs",dtype=np.int8)
Banda2=np.reshape(Imagen2,(512,512))
#Banda 3
Imagen3=np.fromfile(r"C:\Users\Randu\Desktop\KNN - Python\band3.irs",dtype=np.int8)
Banda3=np.reshape(Imagen3,(512,512))
#Banda 4
Imagen4=np.fromfile(r"C:\Users\Randu\Desktop\KNN - Python\band4.irs",dtype=np.int8)
Banda4=np.reshape(Imagen4,(512,512))

#Creamos un vector de caracteristicas con las bandas
Vector=[]
for i in range(0,512):
	for j in range(0,512):
		Vector.append([float(Banda1[i][j]),float(Banda2[i][j]),float(Banda3[i][j]),float(Banda4[i][j])])
		
#Se convierte en array para mantener el mismo formato de los datos cargados
VectorBandas = np.array(Vector)

#Entrenamiento del modelo, para determinar el mejor k vecinos
Modelo = KNeighborsClassifier(n_neighbors=3)
Modelo.fit(DatoEnt_X,DatoEnt_Y)
PredDatoExp_Y= Modelo.predict(DatoExp_X)
Precision= accuracy_score(DatoExp_Y,PredDatoExp_Y)
print("Precision = ", Precision)

#Modelo una vez descubierto el mejor k
MejorModelo = KNeighborsClassifier(n_neighbors=3)
MejorModelo.fit(DatosTotales_X,DatosTotales_Y)
PredDatos = MejorModelo.predict(VectorBandas)

#Se crea una matriz con los datos ya procesados
VectorImagen = np.matrix(PredDatos)
#Se crea la dimension de la imagen con los datos
ImagenSalida = np.reshape(VectorImagen,(512,512))

#Creacion de la imagen
def Imagen(matriz, NombreArchivo):
	cv2.imwrite(NombreArchivo, matriz)

Imagen(ImagenSalida, "imagen.pbm") 



