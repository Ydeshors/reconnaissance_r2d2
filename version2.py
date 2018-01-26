import matplotlib

import pylab
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from scipy import signal, interpolate
from fastdtw import fastdtw, dtw
from pydub import AudioSegment
from scipy.io.wavfile import read as wread
from scipy.spatial.distance import euclidean
from IPython.core.display import display, HTML

tDicoSonsConnus = {}
nLongueurMinSonConnu = np.Inf
bDurees = False

dureeSeconde = 0.01
frequenceEchantillonage = 22050
nTailleFen = round(frequenceEchantillonage*dureeSeconde)

def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))
    
    if len(y_axis) != len(x_axis):
        raise ValueError( 
                "Input vectors y_axis and x_axis must have same length")

    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis

"""Fonction de détection de pics, utilise datacheck_peakdetect:
    lookahead 200 veut dire regarde le pic le plus important sur une fenếtre de 200
    delta représente le seuil de detection
"""
def peakdetect(y_axis, x_axis = None, lookahead = 200, delta=0):
    """
    Converted from/based on a MATLAB script at: 
    http://billauer.co.il/peakdet.html
    
    function for detecting local maxima and minima in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maxima and minima respectively
    
    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks. If omitted an
        index of the y_axis is used.
        (default: None)
    
    lookahead -- distance to look ahead from a peak candidate to determine if
        it is the actual peak
        (default: 200) 
        '(samples / period) / f' where '4 >= f >= 1.25' might be a good value
    
    delta -- this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            When omitted delta function causes a 20% decrease in speed.
            When used Correctly it can double the speed of the function

    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*max_peaks)
    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false
       
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)
    
    
    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")
    
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], 
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    break
                continue
        
        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found 
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
    return [max_peaks, min_peaks]

""" Fenêtrage du signal => fenêtre de hamming"""
def hammi(signal):
    global nTailleFen
    fenPure = np.hamming(nTailleFen)
    nSize = len(signal)
    tSortie = []
    n=0
    nDecal = (nTailleFen+1)//2
    while (n*nDecal) < (nSize-nTailleFen):
        tSortie.append( signal[n*nDecal:n*nDecal +nTailleFen]*fenPure )
        n +=1
    return tSortie

""" Affichage des différents spectres """
def Affichages(sFichier, data, dataModded, tZRC, tZREnergie, tSignalHamminged, tSignalFft, tSilences=[], nDecalageStart=0, bSignalOrignal = True, bHamminged=True, bZRs=True, bFftFused=True, bFFTs=True, bPeaks=True):
    if (bSignalOrignal):
        plt.figure(figsize=(15, 4))
        plt.plot(data)
    if (len(tSilences)>0):
        plt.plot(tSilences, [2000 for _ in range(len(tSilences))], "o")
        plt.plot(tSilences, [2000 for _ in range(len(tSilences))],"o")
        plt.plot([x+nDecalageStart for x in tSilences ], [2000 for _ in range(len(tSilences))])
    if (bZRs):
        # plus la fenetre est courte et plus la courbre est agitée
        plt.plot(tZRC)
        plt.plot(tZREnergie)
    if (bSignalOrignal or bZRs):
        plt.title("Fichier "+str(sFichier))
        plt.show()
    
    if (bHamminged):
        nDecalage = (len(tSignalHamminged[0])+1)//2
        plt.plot([x * nDecalage for x in  list(range(0,len(tSignalHamminged)) ) ], tSignalHamminged)
        plt.title("fenêtré hamming")
        plt.show()

    if (bFftFused):
        nDecalage = (len(tSignalHamminged[0]) + 1) // 2
        plt.plot([x * nDecalage for x in  list(range(0,len(tSignalFft)) ) ], tSignalFft)
        plt.title("fft")
        plt.show()

    if (bFFTs):
        nLongueurMaxFFT = len(tSignalFft[0])
        nLongOpti = nLongueurMaxFFT//2
        
        for x in range (len(tSignalFft)):
            plt.plot(tSignalFft[x][:nLongOpti])
            plt.title("Fenêtre : "+ str(x))

            if (bPeaks):
                nDelta = max(tSignalFft[x]) *0.5
                ligne = peakdetect(tSignalFft[x][:nLongOpti], lookahead=6, delta=nDelta)
                plt.plot([ item[0] for item in ligne[0]], [ item[1] for item in ligne[0]], "o")
            plt.show()

"""
Fonction de détection des pics, utilise peakDetect
peakDetect retourne dans l'odre:
    - Les pics positifs
    - Les pics négatifs
"""
def detectPics(tSignalFft, coeffDelta = 0.5, distancePic = 6) :
    tSuccessionPics = []
    nLongueurMaxFFT = len(tSignalFft[0])
    nLongOpti = nLongueurMaxFFT//2
    for x in range (len(tSignalFft)):
        nDelta = max(tSignalFft[x]) * coeffDelta
        ligne = peakdetect(tSignalFft[x][:nLongOpti], lookahead=distancePic, delta=nDelta)
        [tSuccessionPics.append(item[0]) for item in ligne[0]]
    return tSuccessionPics
    
"""
Fenêtre de Hamming et transformée de Fourier sur cette fenêtre
"""
def HammingPaddingFourier(dataModded):
    tSignalHamminged = hammi(dataModded)
    tSignalFft = tSignalHamminged[:]

    tPaddTableau = np.array([0] * len(tSignalFft[0]) *5)
    for x in range(len(tSignalFft)):
        tSignalFft[x] = np.concatenate((tSignalFft[x],tPaddTableau))

    tSignalFft = list(map(np.fft.fft, tSignalFft) )
    # Reel = lambda x: x.real
    # Image = lambda x: x.imag
    Module = lambda x: abs(x)
    tSignalFft = list(map(Module, tSignalFft) )
    return (tSignalHamminged, tSignalFft)

""" Fonction de détection de la voix (vad = voice active detection 
Utilise les fenêtres de hamming 
calcul du taux de passage par zero
calcul de l'énergie 
"""
def detectVoix(data, nSeuil=30, bExcluSilences = False):
    dataModded = data[:]
    if (bDurees):
        nTps = time.process_time()
    tSignalHamminged = hammi(data)
    if (bDurees):
        print("Durée Hamming: ", time.process_time()-nTps )
        nTps = time.process_time()
    #ZRC zero crossing rate, pour elimination des silences
    tnbZeros = [1]*len(data)
    cpt = 1
    while cpt<len(data):
        if (data[cpt]>0 and data[cpt-1]>0):
            tnbZeros[cpt] = 0
        elif (data[cpt]<0 and data[cpt-1]<0):
            tnbZeros[cpt] = 0
        cpt +=1

    nNbSignauxHam = len(tSignalHamminged)
    tZRC2 = [0]*len(data)
    tZRC3 = [0]*len(data)
    tEnergie = [0]*nNbSignauxHam
    
    nHam = len(tSignalHamminged[0])
    nDecalage = (nHam+1)//2

    # Calcul de l'énergie
    cpt=0
    for indice, item in enumerate(tSignalHamminged):
        nMoyEnergie = 0
        #Entre officiel et alternative différence de temps alt:0.062, offi:0.109
        #officiel, somme de carrés moyennés
        for x in range(nHam):
            nMoyEnergie += (data[x+indice*nDecalage]/nHam)**2
        tEnergie[cpt] = nMoyEnergie
        cpt +=1
    
    if (bDurees):
        print("Durée ZRC et énergie séparés: ", time.process_time()-nTps )
        nTps = time.process_time()

    #  Interpolation car on calcul les énergies par fenetre et non par point
    nMax = nDecalage*nNbSignauxHam
    tAbs = np.arange(0,nMax+3*nDecalage,nDecalage)
    f = interpolate.interp1d(tAbs, [tEnergie[0]]+tEnergie+[tEnergie[-1],tEnergie[-1]])
    
    if (bDurees):
        print("Durée Interpolate de ZRC: ", time.process_time()-nTps )
        nTps = time.process_time()
    
    #ICI choisir un pas + grand pour réduire la durée d'exécution, ce sera comme prendre une plus grande fenêtre, la précision n'est pas d'une grande importance ici
    cpt=0
    nPas = 50
    while cpt<len(tnbZeros) : #len(data)
        nMin = max(cpt-250,0)
        nMax = min(cpt+250,len(tnbZeros))
        nMaxCptBis = min(cpt+nPas,len(tnbZeros))
        nLong = nMax-nMin

        nSommeZeros = sum(tnbZeros[nMin:nMax])

        # Moyenne du taux de passage par 0 pour un fenêtre donnée
        ajout = (nSommeZeros/nLong)* 50000
        ajoutUnchanged = ajout
        
        nEnergieEtZeroRate = f(cpt)*ajout*0.00002
        if (nEnergieEtZeroRate <  nSeuil  ):
            ajout = 0
            for cptBis in range (cpt,nMaxCptBis):
                dataModded[cptBis] = 0
        else:
            ajout = nEnergieEtZeroRate+5000
        for cptBis in range (cpt,nMaxCptBis):
            tZRC2[cptBis] = ajoutUnchanged
            tZRC3[cptBis] = ajout
        cpt += nPas
    
    if (bDurees):
        print("Durée Energie de ZRC appliquée: ", time.process_time()-nTps )
    
    #nettoyage du silence en début et fin de fichier
#     Il serait préférable d'avoir assez de 0 en début et fin pour remplir la moitié de la première(dernière) fenêtre de hamming avec ces 0, les fft seront plus douces aux extrémités
    if (bExcluSilences):
        nDebut = 0
        nFin = len(dataModded)
        cpt=0
        while cpt < len(dataModded):
            while(dataModded[cpt]==0):
                cpt+=1
            nDebut = max (nDebut, cpt-nTailleFen)
            break
        # Suppression du nombre à la fin
        cpt = len(dataModded)-2
        while (cpt>0):
            while(dataModded[cpt]==0):
                cpt-=1
            nFin = min (nFin, cpt+nTailleFen)
            break

        if (nFin> len(dataModded)-1):
            nFin = len(dataModded)
        if (nDebut<0):
            nDebut = 0
        
        dataEX = [0]* (nFin-nDebut)
        cpt = nDebut
        cptEX = 0
        while cpt < nFin:
            dataEX[cptEX] = dataModded[cpt]
            cpt +=1
            cptEX +=1
        dataModded = dataEX
    return (dataModded, tZRC2, tZRC3)

""" Fonction de détection des silences """
def detecteSilences(dataModded):
    tSilences = [] # paires d'indices, début et fin de silences
    bSilenceEnCours = False
    nFin = len(dataModded)
    cpt=0
    while cpt < len(dataModded):
        if(dataModded[cpt]!=0):
            if (bSilenceEnCours):
                tSilences.append(cpt-1) # fin du silence àt-1, début de son
                bSilenceEnCours = False
        elif(not bSilenceEnCours):
            bSilenceEnCours = True
            tSilences.append(cpt) # début d'un silence
        cpt+=1
    #possibilité de fin de silence pas détectée par présence de bruit avant la fin
    if (bSilenceEnCours):
        tSilences.append(nFin-1)
    return tSilences

""" Affichage du signal en "temps réel" """
def AfficheLectureEnregistrement(sFichier):
    f_echant, data = wread(sFichier)
    
    # nTpsStart = time.process_time()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # autre version où je garde un nombre de valeurs fixe en abscisses
    nNbValsAbs = 22050 // 5
    nStart = 0
    nEnd = nStart + nNbValsAbs
    nMax = len(data)
    x = np.linspace(0, 10, nNbValsAbs) # nb d'échantillons qu'on a en un 60ième de seconde réparties de 0à10
    tFenHamV1 = np.hamming(nNbValsAbs)

    if (len(data)>= nNbValsAbs):
        y = data[:nNbValsAbs]
    else:
        y = np.concatenate( ( data[:], np.array([0] * (nNbValsAbs-len(data))) ) )

    ax.set_ylim(-nAmplitudeMax,nAmplitudeMax)
    line1, = ax.plot(x, y, '-')
    fig.canvas.draw()

    # changera après, fixe parce que pas encore de rafraichissement
    nDelta = 1/60

    bV1 = True
    bV2 = not bV1
    
    while nEnd <= nMax:
        nTps = time.process_time()
        nNbAvancee = round(nDelta*frequenceEchantillonage) # equivalent du temps passé en nombre d'echantillons
        #v1
            # affiche une période de son fixe, genre la dernière demi seconde, à chaque rafraichissement
            # donc si update rapide alors affichage d'une partie déjà affichée
        if bV1:
            nStart += nNbAvancee
        #v2
            # Affiche seulement le morceau de son lu pendant la durée d'affichage, donc durée variable 10ième de seconde plus moins ...
            # le mieux serait peut-être de seulement afficher les 1/30ème de seconde de son précédent 
            # même si d'un rafraichissement à l'autre on affiche une partie de ce qui a été affiché avant
        if bV2:
            nStart = nEnd
            nNbValsAbs = nNbAvancee
            x = np.linspace(0, 10, nNbValsAbs)
        
        nEnd = nStart + nNbValsAbs
        tHam = np.hamming(nNbValsAbs)

        if nStart >= nMax:
            break
        if nEnd > nMax:
            y = np.concatenate( (data[nStart:nMax], np.array([0] * (nEnd-nMax)) ) )
        else:
            y = data[nStart:nEnd]
        if bV1:
            line1.set_ydata(y*tFenHamV1)
        if bV2:
            ax.clear()
            ax.set_ylim(-nAmplitudeMax,nAmplitudeMax)
            ax.plot(x,y*tHam,'-')
        
        fig.canvas.draw()
        nDelta = time.process_time() - nTps

""" Echantillonnage puis detection si il y a de la voix
Le moyen utilisé est la fonction detectVoix
"""
def travail(tFichiers, bAffichage=False):
    for sFichier in tFichiers:
        f_echant, data = wread(sFichier)    
        dataModded, tZRC, tZREnergie = detectVoix(data, 1000, bExcluSilences=True)
        global nLongueurMinSonConnu
        if (len(dataModded)<nLongueurMinSonConnu):
            nLongueurMinSonConnu = len(dataModded)
        tSignalHamminged, tSignalFft = HammingPaddingFourier(dataModded)
        if bAffichage:
            Affichages(sFichier, data, dataModded, tZRC, tZREnergie, tSignalHamminged, tSignalFft, bFFTs=False)
        tPics = detectPics(tSignalFft)
        tDicoSonsConnus[sFichier] = tPics
        print ("Enregistrement du fichier ", sFichier, " terminé.")
    print("Travail terminé.")

"""
Comparaison du signal au moyen de DTW (Dynamic Time Wrapping)
"""
def comparaison(sFichier, tPics, bPlot = True, bResult=True):
    nNbSonsConnus = len(tDicoSonsConnus)
    tDistances = [0]*nNbSonsConnus
    tNoms = [""]*nNbSonsConnus
    cpt = 0
    for indice, item in tDicoSonsConnus.items() :
        # distance des fréquences
        tDistances[cpt], _ = fastdtw(item, tPics, dist=euclidean)    #dist=euclidean) #dist=None
        # pondérée par la longueur du fichier, nombre d'échantillons .....ou racine de ( L1+L2 ) si distance euclidienne dans la dtw
        tDistances[cpt] = tDistances[cpt] / ((len(item)+len(tPics))/2)
        #distance des puissances, dB
        
        tNoms[cpt] = indice
        cpt += 1
    if (bPlot):
        plt.figure(figsize=(15,4))
        plt.ylabel('Distances')
        tIndices = np.arange(nNbSonsConnus)
        plt.title('Distances de '+sFichier )
        plt.xticks(tIndices, tNoms , rotation=90)
        plt.plot(tIndices, tDistances)
        plt.show()
    if (bResult):
        print ("Son le plus proche : ", tNoms[np.argmin(tDistances)], " | Distance : ", min(tDistances))
    return (tNoms[np.argmin(tDistances)], min(tDistances) )

"""
Evaluation d'un fichier qui a un seul caractère
"""
def evalue(sFichier, bAffichage=True):
    print("Evaluation de ", sFichier)
    f_echant, data = wread(sFichier)
    # Lancement de l'horloge si bDuree
    if (bDurees):
        nTps = time.process_time()

    dataModded, tZRC, tZREnergie = detectVoix(data, 1000, bExcluSilences=True)

    # Temps nécessaire pour la détection de la voix
    if (bDurees):
        print("Durée : ", time.process_time()-nTps )
        nTps = time.process_time()

    tSignalHamminged, tSignalFft = HammingPaddingFourier(dataModded)

    # Temps nécessaire pour la réalisation de HammingPaddingFourier
    if (bDurees):
        print("Durée : ", time.process_time()-nTps )

    if bAffichage:
        Affichages(sFichier, data, dataModded, tZRC, tZREnergie, tSignalHamminged, tSignalFft, bFFTs=False)

    tPics = detectPics(tSignalFft)
    comparaison(sFichier, tPics, bAffichage)
    print("\n\n")

"""
Evaluation d'un fichier qui a plusieurs caractères
"""
def evalueComplexe(sFichier, bDetails=True):
    print("Détermine les sons de ", sFichier)
    f_echant, data = wread(sFichier)
    if (bDurees):
        nTps = time.process_time()
    chercheSons(data, bDetails)
    print("\n\n")

"""
Détecte les silences du signal
Tente de reconnaitre des signaux de réfenrences entre les silences
A chaque étape, concaténation du son actuel et du suivant
puis comparaison à nouveau aux signaux de références.
Tant que la distance diminue on continue d'agrandir le segment de son analysé
"""
def chercheSons(tSonsInconnus, bDetails=True, nSeuilIgnorer = 1000):
    if (bDurees):
        nTps = time.process_time()
    # on hypothèse un fichier de plusieurs sons avec ou sans silences entre les sons
    # virer le silence en début de fichier
    dataModdedFull, tZRC, tZREnergie = detectVoix(tSonsInconnus, nSeuilIgnorer, bExcluSilences=True)
    # à partir de ce nouveau début prendre X données du fichier, X étant la taille minimale d'un son de référence soit nLongueurMinSonConnu
    dataModded, tZRC, tZREnergie = detectVoix(dataModdedFull, nSeuilIgnorer, bExcluSilences=True)
    tSignalHamminged, tSignalFft = HammingPaddingFourier(dataModded)
    
    tSilences = detecteSilences(tZREnergie)
    nSilenceProchain = 0
    global nLongueurMinSonConnu
    nCurseurDebutLecture = 0
    nCurseurFinLecture = tSilences[nSilenceProchain*2]
    if (nSilenceProchain+1 < len(tSilences)/2 ):
        if (nCurseurFinLecture - nCurseurDebutLecture) < nLongueurMinSonConnu*0.75 :
            nSilenceProchain +=1
            nCurseurFinLecture = tSilences[nSilenceProchain*2]
    nTailleMax = len(tSonsInconnus)
    tSonsTrouves = []
    bFichierParcouru = False

    if (bDetails):
        Affichages("Son(s) inconnu(s)", tSonsInconnus, dataModded, tZRC, tZREnergie, tSignalHamminged, tSignalFft, tSilences=tSilences, bFFTs=False, bFftFused=False, bHamminged=False)
    
    while(not bFichierParcouru):

        bAgrandirEchantillon= True
        nDistanceMin = np.Inf
        sSonTrouve = ""
        
        while (bAgrandirEchantillon):
            #pas opti on va recalculer la fft de tout ce qui a déjà été fait, il faudrait prendre la fin moins la dernière fenetre de hamming
            if (bDetails):
                print ("nDébut: ", nCurseurDebutLecture, " | fin: ",nCurseurFinLecture)
            dataModded = dataModdedFull[nCurseurDebutLecture:nCurseurFinLecture]
            dataModded, tZRC, tZREnergie = detectVoix(dataModded, nSeuilIgnorer, bExcluSilences=True)
            if (bDurees):
                print("Durée : ", time.process_time()-nTps )
                nTps = time.process_time()
            tSignalHamminged, tSignalFft = HammingPaddingFourier(dataModded)
            if (bDurees):
                print("Durée : ", time.process_time()-nTps )
            if (bDetails):
                Affichages("Son(s) inconnu(s)", dataModded, dataModded, tZRC, tZREnergie, tSignalHamminged, tSignalFft, bFFTs=False)
            tPics = detectPics(tSignalFft)
            sNomProche, nDistanceProche = comparaison("Son Inconnu", tPics, bDetails, bDetails)
            
            if (nCurseurFinLecture == nTailleMax):
                bAgrandirEchantillon = False
            # même si la nouvelle distance plus petite est sur un autre son c'est OK
            if (nDistanceProche< nDistanceMin):
                nDistanceMin = nDistanceProche
                sSonTrouve = sNomProche
                nSilenceProchain +=1
                if (nSilenceProchain < len(tSilences)/2 ):
                    nCurseurFinLecture = tSilences[nSilenceProchain*2]
                    nCurseurFinLecture = min (nCurseurFinLecture, nTailleMax)
                    if (bDetails):
                        print("agrandi l'échantillon")
                else:
                    nCurseurFinLecture = nTailleMax
            else:
                bAgrandirEchantillon = False
                if (bDetails):
                    print("n'agrandit plus l'éhantillon")
                    print("fin du fichier")
            
        #on pourrait être à la fin du fichier mais de toute façon on arrive là avec un son reconnu donc à ajouter
        tSonsTrouves.append(sSonTrouve)
        if (nSilenceProchain < len(tSilences)/2 ):
            nCurseurDebutLecture = tSilences[(nSilenceProchain-1)*2 +1]
            nCurseurFinLecture = tSilences[(nSilenceProchain)*2]
        else:
            nCurseurFinLecture = nTailleMax
        if (nCurseurFinLecture >= nTailleMax-(nLongueurMinSonConnu*0.1)):
            bFichierParcouru = True
    
    print("Detection Multiple terminée")
    print (tSonsTrouves)
    print("\n\n")
    
    # Enregistrer la plus petite distance pour cet échantillon
    # augmenter la taille de la fenêtre prise d'un quart de seconde
    # Calculer la nouvelle distance de cet échantillon
    #tant que la nouvelle plus petite distance est plus petite que la précédente, continuer à augmenter la taille de la fenêtre et calculer la distance
    # sortie de boucle, enregistrer le son associé à la plus petite distance enregistrée (celle de la fenêtre d'avant la sortie de boucle)
    # Chercher le premier silence sur la dernière fenêtre ajoutée à l'échantillon
    # s'il y a une silence, mettre la fin de ce silence comme nouveau début d'échantillon, sinon le début de la fenêtre est le nouveau début d'échantillon
    # recommencer le fonctionnement précédent pour rechercher un nouveau son, faire celà jusqu'à la fin du fichier (auquel on a enlevé le silence de fin)
    
    #risque que l'algo donne toujours une distance sur un son à la fin alors qu'il n'y a rien à attribuer

""" Fonction d'initialisation
A améliorer, il faudrait sotcker les résultats et non la relancer à chaque analyse
"""
def initialisation():
    liste_fichier = os.listdir('./sound')
    path_fichier = []
    for nomFichier in liste_fichier:
        path_fichier += ['sound/'+nomFichier]
    travail(path_fichier, bAffichage=False)

""" Fonction principale pour les sons contenant un seul caractère """
def evaluation(nom, affichage=False):
    nomfichier = 'soundTest/'+nom+'.wav'
    evalue(nomfichier, affichage)

""" Fonction principale pour les sons contenant plusieurs caractères"""
def evaluation_complexe(nom, affichage=False):
    nomfichier = 'soundTest/' + nom + '.wav'
    evalueComplexe(nomfichier, affichage)

initialisation()

# Exemple d'évaluation d'un son avec affichage du résultat
evaluation('xxx', True)