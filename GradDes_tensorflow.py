import numpy as np
import tensorflow as tf
import time


alpha=0.01 #Parámetro de la función leaky ReLU

def ReLU(x):
    return tf.maximum(0,x)

def ReLUder(x):
    return tf.where(x <= 0, 0., 1.)


def leakyReLU(x):
  return tf.where(x <= 0, alpha*x, x)

def leakyReLUder(x):
    return tf.where(x <= 0, alpha, 1.)

def red(x,W,b,fun_act):
    i_list=False
    if type(fun_act)==list:
        f_list=[0,]

        i_list=True
        for k in fun_act:
            if k=='sigmoid':f_list.append(lambda x:1/(1+tf.math.exp(-x)))
            elif k=='tanh':f_list.append(lambda x:tf.math.tanh(x))
            elif k=='ReLU': f_list.append(ReLU)
            elif k=='leakyReLU':f_list.append(leakyReLU)

    elif fun_act=='sigmoid':
        f=lambda x: 1/(1+tf.math.exp(-x))
    elif fun_act=='tanh':
        f=lambda x: tf.math.tanh(x)
    elif fun_act=='ReLU':
        f=ReLU
    elif fun_act=='leakyReLU':
        f=leakyReLU

    a_c=[x,]

    for i in range(len(W)-1):
        if i_list==True:
            f=f_list[i+1]

        a_c.append(f(tf.matmul(W[i+1],a_c[-1])+b[i+1]))


    a=a_c[-1]

    return a

def GradDes(arq,puntos_x,puntos_y,fun_act,fcoste,eta=0.05,Niter=5e5):
        '''
        Función que a partir de:
        + Un array con la arquitectura de la red
        + Nombre de la función de activación o una lista con la función de activación de cada
        capa entre las posibles opciones (sigmoide,tanh, ReLU y leakyReLU)
        + Dos listas con los puntos de entrenamiento y sus salidas
        +Parámetro fcoste que dependiendo de si esta On u Off calcula el valor
        de la función en cada iteración

        Realiza el método del Gradiente Descendiente a partir de un valor de tasa de aprendizaje (eta)
        y de iteraciones máximas (Niter) predeterminados  y devuelve:
        + Lista con las matrices de pesos finales
        + Lista con los vectores de sesgos finales
        + Vector de dimensión Niter con los valores de la función de costes en cada iteración
        '''

        Frec_costes=50000

        W=[0,] #Creación de la lista de matrices de pesos introduciendo un 0 para respetar los índices
        b=[0,] #Creación de la lista de vectores de sesgos introduciendo un 0 para respetar los índices
        coste=np.zeros((int(Niter/Frec_costes))) #Array de ceros para almacenar el valor de la función de costes en cada iteración
        tiempo=[]

        is_f_list=False

        if type(fun_act)==list:
            is_f_list=True
            f_list=[0,]
            der_list=[0,]
            for x in fun_act:
                if x=='sigmoid':
                    f_list.append(lambda x: 1/(1+tf.math.exp(-x)))
                    der_list.append(lambda x: tf.math.exp(-x)/(1+tf.math.exp(-x))**2)
                elif x=='tanh':
                    f_list.append(lambda x:tf.math.tanh(x))
                    der_list.append(1/np.cosh(x)**2)
                elif x=='ReLU':
                    f_list.append(ReLU)
                    der_list.append(ReLUder)
                elif x=='leakyReLU':
                    f_list.append(leakyReLU)
                    der_list.append(leakyReLUder)
        else:
            if fun_act=='sigmoid':
                f=lambda x: tf.math.tanh(x)
                der=lambda x: 1-tf.math.tanh(x)**2
            elif fun_act=='tanh':
                f=lambda x: np.tanh(x)
                der=lambda x: 1/np.cosh(x)**2
            elif fun_act=='ReLU':
                f=ReLU
                der=ReLUder
            elif fun_act=='leakyReLU':
                f=leakyReLU
                der=leakyReLUder
        if is_f_list==True:
            for k in range(len(arq)-1):
                '''
                Bucle que crea las matrices y vectores aleatorios iniciales según las
                condiciones iniciales de Xavier o He dependiendo de la función de activación
                escogida
                '''
                if fun_act[k]=='sigmoid':
                    W.append(np.sqrt(6./(arq[k]+arq[k+1]))*tf.random.uniform((arq[k+1],arq[k]),minval=-1,maxval=1))
                    b.append(tf.zeros((arq[k+1],1)))

                elif fun_act[k]=='tanh':
                    W.append(np.sqrt(6./(arq[k]+arq[k+1]))*tf.random.uniform((arq[k+1],arq[k]),minval=-1,maxval=1))
                    b.append(tf.zeros((arq[k+1],1)))
                elif fun_act[k]=='ReLU':
                    W.append(tf.random.normal((arq[k+1],arq[k]),mean=0,stddev=2/arq[k]))
                    b.append(tf.zeros((arq[k+1],1)))

                elif fun_act[k]=='leakyReLU':
                    W.append(tf.random.normal((arq[k+1],arq[k]),mean=0,stddev=2/arq[k]))
                    b.append(tf.zeros((arq[k+1],1)))

        else:
            for k in range(len(arq)-1):
                '''
                Bucle que crea las matrices y vectores aleatorios iniciales según las
                condiciones iniciales de Xavier o He dependiendo de la función de activación
                escogida
                '''
                if fun_act=='sigmoid':
                  W.append(np.sqrt(6./(arq[k]+arq[k+1]))*tf.random.uniform((arq[k+1],arq[k]),minval=-1,maxval=1))
                  b.append(tf.zeros((arq[k+1],1)))
                elif fun_act=='tanh':
                  W.append(np.sqrt(6./(arq[k]+arq[k+1]))*tf.random.uniform((arq[k+1],arq[k]),minval=-1,maxval=1))
                  b.append(tf.zeros((arq[k+1],1)))
                elif fun_act=='ReLU':
                  W.append(tf.random.normal((arq[k+1],arq[k]),mean=0,stddev=2/arq[k]))
                  b.append(tf.zeros((arq[k+1],1)))
                elif fun_act=='leakyReLU':
                  W.append(tf.random.normal((arq[k+1],arq[k]),mean=0,stddev=2/arq[k]))
                  b.append(tf.zeros((arq[k+1],1)))
        N=0 #Número de iteración para el bucle
        N_puntos=len(puntos_x) #Almacenado del número de puntos de entrenamiento para comodidad

        while N<Niter:

            Inicio=time.time() #Punto de medir el tiempo de inicio de la época

            i=np.random.randint(N_puntos)

            x=tf.constant(puntos_x[i],dtype='float32') #Creación del punto de entrenamiento

            y=tf.constant(puntos_y[i],dtype='float32') #Creación de la salida del punto de entrenamiento

            a=[x,] #Lista de vectores de salida en cada capa
            z=[0,] #Lista de vectores z (W*a+b) en cada capa
            D=[0,] #Lista derivadas en cada capa
            delta=[]  #Lista del valor de delta en  cada capa

            for k in range(1,len(arq)):
                '''
                Bucle que realiza la propagación hacia delante de la red almacenando tanto las salidas en
                cada capa como el vector z como la derivada.
                '''
                if is_f_list==True:
                    f=f_list[k]
                    der=der_list[k]
                z.append(tf.matmul(W[k],a[k-1])+b[k])
                a.append(f(z[k]))
                D.append(der(z[k]))

            delta.append(D[-1]*(a[-1]-y)) #Introducción del valor de delta en la capa de salida

            for i in np.flip(range(1,len(W)-1)):
                '''
                Bucle que realiza la propagación hacia detrás de la red almacenando en una lista
                el valor de delta para cada capa
                '''
                delta.append((W[i+1].T@delta[-1])*D[i])

            delta.append(0) #Añadir un cero a la lista de deltas y voltearla para respetar los índices del pseudocódigo
            delta=list(reversed(delta))
            for i in np.flip(range(1,len(W))):
                '''
                Actualización de las matrices de pesos y vectores de sesgos a partir de los delta calculados
                '''
                W[i]=W[i]-eta*tf.matmul(delta[i],tf.transpose(a[i-1],perm=[1,0]))
                b[i]=b[i]-eta*delta[i]

            #Calculo de la función coste

            if fcoste=='On' and N%Frec_costes==0:

                malla_puntosx=tf.constant(np.array(puntos_x),dtype='float32')
                malla_puntosy=tf.constant(np.array(puntos_y),dtype='float32')
                a=malla_puntosx

                for i in range(1,len(W)):
                  a=f(tf.matmul(W[i],a)+b[i])

                coste[int(N/Frec_costes)]=1/(N_puntos)*0.5*np.linalg.norm(a-malla_puntosy,axis=(1,0))**2 #cálculo del valor de la función de costes en la iteración correspondiente

                Fin_on=time.time() #Punto de medir el tiempo final en el modo fcoste='On'
                tiempo.append(Fin_on-Inicio) #Añadir el tiempo de la época al array de almacenado

                print('========================================')
                print('Epoch %i/%i'%(N+1,int(Niter))) #Impresión en pantalla de la iteración y el valor del coste
                print('Valor Función de costes: ',coste[int(N/Frec_costes)])
                print('Tiempo de ejecución: ',Fin_on-Inicio)
            else:
                Fin_off=time.time() #Punto de medir el tiempo final en el modo fcoste='Off'
                tiempo.append(Fin_off-Inicio) #Añadir el tiempo de la época al array de almacenado
                print('========================================')
                print('Epoch %i/%i'%(N+1,int(Niter)))
                print('Tiempo de ejecución: ',Fin_off-Inicio)
            N+=1


        if fcoste=='On': return W,b,coste,np.array(tiempo)

        else: return W,b,np.array(tiempo)