#import numpy as np
#from matplotlib import pyplot as plt
from math import sqrt, cos, sin, atan2, atan, pi

def suma(c1, c2):
    ''' Esta función realiza la suma de dos números complejos c1 y c2 en notación rectangular.
        a: Parte Real
        b: Parte Imaginaria
        (a, b) + (c, d) --> (e, f)'''
    
    a, b = c1
    c, d = c2
    return (float(a + c), float(b + d))

def producto(c1, c2):
    '''Esta función realiza el producto de dos números complejos c1 y c2 en notación rectangular.
       a: Parte Real
       b: Parte Imaginaria
       (a, b) * (c, d) --> (e, f)'''
    
    a, b = c1
    c, d = c2
    return (float(a * c - b * d), float(a * d + b * c))

def resta(c1, c2):
    '''Esta función realiza la resta de dos números complejos c1 y c2 en notación rectangular.
       a: Parte Real
       b: Parte Imamginaria
       (a, b) - (c, d) --> (e, f)'''
    
    a, b = c1
    c, d = c2
    return (round(float(a - c),2),  round(float(b - d), 2))

def division(c1, c2):
    '''Esta función realiza la división de dos números complejos c1 y c2 en notación rectangular.
       a, c: Parte Real
       b, d: Parte Imaginaria
       (a, b) / (c, d) --> (e, f)'''
    
    a, b = c1
    c, d = c2
    try:
        return (round(float((a*c + b*d)) / float((c**2 + d**2)), 2) , round(float((c*b - a*d)) / float((c**2 + d**2)),  2))
    
    except ZeroDivisionError:
        print('Operación indefinida a / 0')

def modulo(c1):
    '''Esta función realiza el módulo de un número complejo c1 en notación rectangular.
       a: Parte Real
       b: Parte Imaginaria
       m: Módulo Real+
       |c1| --> m '''
    
    a, b = c1
    return sqrt(a**2 + b**2)

def conjugado(c1):
    '''Esta función realiza el conjugado de un número complejo c1 en notación rectangular.
       a: Parte Real
       b: Parte Imaginaria
       z: conjugado de c1
       (a, b) --> z = (a, -b)'''
    
    a, b = c1
    return (round(float(a), 2), round(float(-1 * b), 2))

def convertir_cartesiana(c1):
    '''Esta función realiza la conversión de un número complejo c1 en notación polar a notación cartesiana.
       m: Módulo
       o: Angulo
       a: Parte Real
       b: Parte Imaginaria '''
    
    m, o = c1
    a = round(m * cos(o), 2)
    b = round(m * sin(o), 2)
    return (a, b)

def convertir_polar(c1):
    '''Esta función realiza la conversión de un número complejo c1 en notación cartesiana a notación polar.
       m: Módulo
       o: Angulo
       a: Parte Real
       b: Parte Imaginaria '''
    a, b = c1
    m = modulo(c1)
    try:
        o = atan2(b , a)
        return (m, o)
    
    except ZeroDivisionError:
        print('Operación indefinida a / 0')

def signo(y):
    if y >= 0 : return 1
    else: return -1

def fase(c1):
    '''Esta funcion retorna la fase de un número complejo c1 si se encuentra en notación cartesiana'''
    
    a, b = c1
    return pi/2 * signo(b) - atan(a / b)

def suma_de_vectores(vectorA, vectorB):
    '''Retorna un Vector C = VectorA + VectorB
       VectorA, VectorB, VectorC : vectores complejos'''
    if len(vectorA) != len(vectorB): raise 'Esta operacion esta indefinida'
    else:
        vectorC = []
        for elemento in zip(vectorA, vectorB):
            vectorC.append(suma(elemento[0], elemento[1]))
    return vectorC

def resta_de_vectores(vectorA, vectorB):
    '''Retorna un Vector C = VectorA - VectorB
       VectorA, VectorB, VectorC : vectores complejos'''
    if len(vectorA) != len(vectorB): raise 'Esta operacion esta indefinida'
    else:
        vectorC = []
        for elemento in zip(vectorA, vectorB):
            vectorC.append(resta(elemento[0], elemento[1]))
    return vectorC

def vector_inverso_aditivo(vectorA):
    '''Retorna el VectorB = (-1) * VectorA
       VectorA, VectorB : Vectores complejos '''
    vectorB = [ (float(vi[0]*-1), float(vi[1]*-1)) for vi in vectorA]
    return vectorB

def vector_x_escalar(vectorA, escalar):
    '''Retorna el VectorB producto escalar del VectorA
       vectorA, vectorB: vectores complejos '''
    vectorB = [ producto(elemento, (escalar, 0)) for elemento in vectorA]
    return vectorB

def suma_de_matrices(matrizA, matrizB):
    '''Retorna la matrizC como suma de la matrizA y matrizB
       matrizA, matrizB, matrizC : matrices de complejos'''
    if len(matrizA) != len(matrizB): raise 'Operacion Indefinida'
    else:
        matrizC = []
        for elemento in zip(matrizA, matrizB):
            matrizC.append(suma_de_vectores(elemento[0], elemento[1]))
    return matrizC

def resta_de_matrices(matrizA, matrizB):
    '''Retorna la matrizC como resta de la matrizA y matrizB
       matrizA, matrizB, matrizC : matrices de complejos'''
    if len(matrizA) != len(matrizB): raise 'Operacion Indefinida'
    else:
        matrizC = []
        for elemento in zip(matrizA, matrizB):
            matrizC.append(resta_de_vectores(elemento[0], elemento[1]))
    return matrizC

def matriz_inverso_aditivo(matrizA):
    '''Retorna la matrizB que es el inverso aditivo de la matrizA
       matrizA, matrizB: matrices de complejos'''
    matrizB = [ vector_inverso_aditivo(vector) for vector in matrizA]
    return matrizB

def matriz_x_escalar(matrizA, escalar):
    '''Retorna la matrizB que es igual al producto escalar de la matrizA 
       matrizA, matrizB: matrices de complejos
       escalar: numero Real'''
    matrizB = [vector_x_escalar(vector, escalar) for vector in matrizA]
    return matrizB

def modulo_vector(V):
    '''Retorna el Modulo de un vector V complejo'''
    sumatoria = 0
    for complejo in V:
        sumatoria += modulo(complejo)**2
    return (sumatoria**0.5)

def matriz_transpuesta(matriz):
    '''Retorna la matriz Transpuesta'''
    temp = []
    for i in range(len(matriz[0])):
        row = []
        for j in range(len(matriz)):
            row.append(matriz[j][i])
        temp.append(row)
    return temp

def matriz_conjugada(matriz):
    '''Retorna la matriz conjugada'''
    conjugada = []
    for i in range (len(matriz)):
        row = []
        for j in range (len(matriz[0])):
            row.append(conjugado(matriz[i][j]))
        conjugada.append(row)
    return conjugada

def matriz_adjunta(matriz):
    '''Retorna la matriz adjunta'''
    return matriz_transpuesta(matriz_conjugada(matriz))

def producto_punto(vectorA, vectorB):
    '''Retorna el producto punto real'''
    if len(vectorA )!= len(vectorB): raise 'Operacion Indefinida'
    else:
        sumatoria = (0, 0)
        for par in zip(vectorA, vectorB):
            sumatoria = suma(sumatoria, producto(par[0], par[1]))
        return sumatoria
    
def multiplicar_matrices_complejas(matrizA, matrizB):
    '''Retorna el producto matricial de una matrizA y una matrizB'''
    if len(matrizA[0]) != len(matrizB): raise 'Operacion Indefinida'
    else:
        matrizC = []
        for vector_fila in matrizA:
            fila = []
            for i in range(len(matrizB[0])):
                vector_columna = []
                for j in range(len(matrizB)):
                    vector_columna.append(matrizB[j][i])  
                fila.append(producto_punto(vector_fila, vector_columna))
            matrizC.append(fila)
        return matrizC

def accion_matriz_vector(matriz, vector):
    '''Retorna el producto matricial entre una matriz y un vector'''
    if len(matriz[0]) != len(matriz) != len(vector): raise 'Operacion Indefinida'
    else:
        accion = []
        for fila in matriz:
            accion.append(producto_punto(fila, vector))
        return accion

def producto_interno_vectores(vectorA, vectorB):
    '''Retorna el producto punto o producto interno de dos vectores complejos'''
    a = matriz_conjugada([vectorA])
    b = matriz_transpuesta([vectorB])
    return multiplicar_matrices_complejas(a, b)[0]

def trace(C):
    '''Retorna la sumatoria de la diagonal de una matriz cuadrada'''
    sumatoria = (0, 0)
    for i in range(len(C)):
        sumatoria = suma(sumatoria, C[i][i])
    return sumatoria

def producto_interno_matrices(matrizA, matrizB):
    '''Retorna el producto punto o producto interno de dos matrices de complejos'''
    if len(matrizA) != len(matrizA[0]) != len(matrizB) != len(matrizB[0]) : raise 'Operacion indefinida para esas matrices'
    else:
        return trace(multiplicar_matrices_complejas(matriz_adjunta(matrizA), matrizB))

def sng(i):

    if i == 0:
        sng = 0
    else:
        sng = 1 if i > 0 else -1
    return sng

def square_root(z):
    '''Retorna la raiz cuadrada de un numero complejo z'''
    real = z[0]
    img = z[1]
    norma = modulo(z)
    return (round(sqrt((norma + real)/2), 2), round(sng(img)*sqrt((norma - real)/2), 2))

def norma_de_matriz(matriz):
    '''Retorna la norma de una matriz'''
    if len(matriz) != len(matriz[0]): raise 'Operacion indefinida para esta matriz'
    return square_root(producto_interno_matrices(matriz, matriz))

def distancia_entre_matrices(matrizA, matrizB):
    '''Retorna la distancia entre dos matrices'''
    return norma_de_matriz(matrizA-matrizB)

def es_hermitiana(matriz):
    '''Retorna verdadero si una matriz es hermitiana, falso de lo contrario'''
    if len(matriz) != len(matriz[0]): raise 'Operacion Indefinida para esta Matriz'
    return True if matriz == matriz_adjunta(matriz) else False

def es_unitaria(matriz):
    '''Retorna verdadero si una matriz es unitaria, falso de lo contrario'''
    if len(matriz) != len(matriz[0]): raise 'Operacion Indefinida para esta Matriz'
    identity = [[(float(0), float(0) )for i in range(len(matriz))] for j in range(len(matriz))]
    for i in range(len(identity)):
        identity[i][i] = (float(1), float(0))
    return multiplicar_matrices_complejas(matriz, matriz_adjunta(matriz)) == multiplicar_matrices_complejas(matriz_adjunta(matriz), matriz) == identity

def producto_tensor(A, B):
    '''Retorna el producto tensor de dos matrices A y B'''
    m, m_ = len(A), len(A[0])
    n, n_ = len(B), len(B[0])
    C = [[(0, 0) for i in range(m_*n_)] for i in range(m*n)]
    for j in range(len(C)):
        for k in range(len(C[0])):
            try :
                C[j][k] = producto(A[j//n][k//m], B[j%n][k%m])
            except IndexError:
                next
    return C

def productoTensor(A,B):
    '''Retorna el producto tensor de dos matrices A y B'''
    aux = []
    subLista = []
    conta = len(B)
    for i in A:
        valorB = 0
        valorA = 0
        while valorA < conta:
            for num1 in i:
                for num2 in B[valorB]:
                    subLista.append(producto(num1,num2))
            aux.append(subLista)
            subLista = []
            valorA +=1
            valorB += 1
    return aux

def identidad(n):
    '''Retorna la matriz idenidad en C*n'''
    x = [[(0,0) for i in range(n)] for j in range(n)]
    for i in range(n):
        x[i][i] = (1,0)
    return x

def producto_interno_vectorial(vector1,vector2):
    '''Se ingresan 2 vectores complejos de longitud n, retorna el producto interno entre estos'''
    if len(vector1) != len(vector2): raise 'Los vectores no tienen la misma longitud, su producto interno no esta definido'
    aux = (0,0)
    for i in range(len(vector1)): 
        aux = suma(aux, producto(vector1[i],vector2[i]))
    return aux;


        

