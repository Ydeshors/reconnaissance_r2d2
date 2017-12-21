import pylab
import numpy as np
import matplotlib.pyplot as plt

# importation de module pour la manipulation de fichiers audios
from fastdtw import fastdtw, dtw
from pydub import AudioSegment
from scipy.io.wavfile import read as wread
from scipy.spatial.distance import euclidean


def w_k(k, n, N):
    return np.exp(2*1j*k*n*np.pi/N)

def dft(s, N):
    result = [0]*int(N)
    k = 0
    while (k<N):
        n=0
        while (n<N):
            result[k] += s[n]*w_k(k,n,N).conjugate()
            n = n + 1
        k = k + 1
    return result


# Fichier en entrée (ce qui a été dit)

f_echant, data = wread('sound/1.wav')
facteur = 16

# representation amplitude fréquence
abscisse = []
ordonne = []
cpt = 0


# normalement on applique la DFT sur tout le signal donc avec N=len(s)
# cependant avec cette méthode de calcul non optimisée, c'est trop lent
# il faut sous-echantilloner, par exemple d'un facteur 4

# print ("Nombre d'echantillons du signal sous-echantillone d'un facteur ", facteur, " : ", len(data)/facteur)
# taille de la transformée de Fourier
# N = int(len(data) / facteur)  # on prend une valeur paire pour N
# if (N%2!=0):
#     N = N -1
# print ("Taille de la DFT : ",N)
# Nouvelle fréquence d'échantillonage
# NFS = f_echant / facteur
# print ("Nouvelle Frequence d'echantillonage : ", NFS )
# nouvelle resolution fréquentielle
# = une des valeurs de la DFT représente (englobe) combien de Hz
# RF = NFS / N
# print ("Resolution frequentielle: ", RF)

# on calcul la DFT du signal sous-echantillone
# dft_signal_ss_echant=dft(data[::facteur],len(data)//facteur)

# on affiche le spectre d'amplitude, que sur la moitié à cause de la symétrie fréquentielle
# on passe les amplitudes en dB
# cpt = 0
# abscisse = []
# ordonne = []
# while (cpt<N):
#         abscisse.append(RF * cpt)
#         ordonne.append(10 * np.log10(abs(dft_signal_ss_echant[cpt])))
        # cpt += 1

# A revoir
def detection_parole(son):
    facteur = 16
    N = int(len(son) / facteur)  # on prend une valeur paire pour N
    if (N % 2 != 0):
        N = N - 1
    dft_signal_ss_echant = dft(son[::facteur], len(son) // facteur)
    log_result = []
    moyenne_log = 0
    for elem in dft_signal_ss_echant:
        log_result.append(10 * np.log10(abs(elem)))
        moyenne_log += 10 * np.log10(abs(elem))
    max_log = max(log_result)
    moyenne_log = moyenne_log / len(dft_signal_ss_echant)

    print("Max log:")
    print(max_log)

    print(moyenne_log)
    # 10% de variation -> parole
    if ((max_log-moyenne_log) > moyenne_log*0.1):
        print("Bruit détecté \n")
    else:
        print("RAS mon colonel \n")


def comparaison(son):
    f_echant, data = wread('audiorecordtest2TMP.wav')
    # f_echant2, data2 = wread('sound/1.wav')

    # 36 pour les 10 chiffres et les 26 lettres
    distance = [100]*36
    # Comparaison avec les chiffres
    cpt = 0
    while (cpt<10):
        nomFichier = 'sound/' + str(cpt) +'.wav'
        f_echant2, data2 = wread(nomFichier)
        distance[cpt], _ = fastdtw(data, data2, dist=euclidean)
        distance[cpt], _ = dtw(data, data2, dist=euclidean)
        print(cpt)
        cpt = cpt + 1
    # facteur = 16
    # N = int(len(son) / facteur)  # on prend une valeur paire pour N
    # if (N % 2 != 0):
    #     N = N - 1
    # NFS = f_echant / facteur
    # RF = NFS / N
    # dft_signal_ss_echant = dft(data[::facteur], len(data) // facteur)
    # dft_signal_ss_echant2 = dft(data2[::facteur], len(data2) // facteur)
    # distance, path = fastdtw(data, data2, dist=euclidean)
    print(distance)

comparaison(data)

# plt.plot(abscisse, ordonne)
# plt.show()
