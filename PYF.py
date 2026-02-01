import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#Cargar los datos
Datos= np.loadtxt("rsTrain.dat")

Datos_X=[]
Datos_Y=[]
for j in Datos:
	Datos_Y.append(j[4])
	Datos_X.append(j[0:4])

Entrenamiento_X, Prueba_X, Entrenamiento_Y, Prueba_Y = train_test_split(Datos_X, Datos_Y, test_size=0.25)

#Banda 1
band1=np.fromfile("band1.irs",dtype=np.int8)
Banda1=np.reshape(band1,(512,512))
#Banda 2
band2=np.fromfile("band2.irs",dtype=np.int8)
Banda2=np.reshape(band2,(512,512))
#Banda 3
band3=np.fromfile("band3.irs",dtype=np.int8)
Banda3=np.reshape(band3,(512,512))
#Banda 4
band4=np.fromfile("band4.irs",dtype=np.int8)
Banda4=np.reshape(band4,(512,512))

#Creamos un vector de caracteristicas
CaracteristicasBandas=[]
for i in range(0,512):
	for j in range(0,512):
		CaracteristicasBandas.append([float(Banda1[i][j]),float(Banda2[i][j]),float(Banda3[i][j]),float(Banda4[i][j])])
		
CaracteristicasBandasBandas = np.array(CaracteristicasBandas)

#Entrenamiento
Modelo = KNeighborsClassifier(n_neighbors=3)
Modelo.fit(Entrenamiento_X,Entrenamiento_Y)
PrediccionPrueba= Modelo.predict(Prueba_X)
Precision= accuracy_score(Prueba_Y,PrediccionPrueba)
print("Precision del modelo = ", Precision)

#Modelo
MejorK = 3
MejorModelo = KNeighborsClassifier(n_neighbors = MejorK)
MejorModelo.fit(Datos_X,Datos_Y)
PredDatos = MejorModelo.predict(CaracteristicasBandasBandas)

#Se crea una matriz
CaracteristicasBandasImagen = np.matrix(PredDatos)
Salida = np.reshape(CaracteristicasBandasImagen,(512,512))

#Imagen
plt.imshow(Salida, cmap="gray")
plt.show()

