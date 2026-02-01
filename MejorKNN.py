import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_excel(r"C:\Users\Randu\Desktop\KNN - Python\MejorKNN.xlsx")
P3="Promedio de K3 es"
P5="Promedio de K5 es"
P7="Promedio de K7 es"
P51="Promedio de K51 es"
P149="Promedio de K149 es"

K3=df["K3"].mean()
Label3=str(K3)
LabelK3= P3 + " " +Label3
print(LabelK3)
K5=df["K5"].mean()
Label5=str(K5)
LabelK5= P5 + " " +Label5
print(LabelK5)
K7=df["K7"].mean()
Label7=str(K7)
LabelK7= P7 + " " +Label7
print(LabelK7)
K51=df["K51"].mean()
Label51=str(K51)
LabelK51= P51 + " " +Label51
print(LabelK51)
K149=df["K149"].mean()
Label149=str(K149)
LabelK149= P149 + " " +Label149
print(LabelK149)

plt.plot(df["K3"],label=LabelK3)
plt.plot(df["K5"],label=LabelK5)
plt.plot(df["K7"],label=LabelK7)
plt.plot(df["K51"],label=LabelK51)
plt.plot(df["K149"],label=LabelK149)

plt.title("Pruebas para sacar el mejor KNN")
plt.xlabel("Cantidad de veces que se hizo las pruebas (30)")
plt.ylabel("Porcentaje de acertividad")
plt.legend(loc="center left",fontsize='small')
plt.grid()
plt.show()



