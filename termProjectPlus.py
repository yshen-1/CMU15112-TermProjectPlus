import pygame, pyaudio, wave, struct, random
import numpy as np
import math, os, threading
from queue import Queue
################################################################################
#15-112 Term Project
#Yuyi Shen, AndrewId: yuyis1, Section N
################################################################################

################################################################################
#Audio Processing section
################################################################################
#http://www.cs.cmu.edu/~112/notes/hw1.html
def almostEquals(x,y,epsilon=10**(-8)):
    return (abs(x-y)<epsilon)

class audioHandler(object):
    #Audio handling class
    def __init__(self,path):
        #Init audioHandler with wave file object and number of frames to read
        self.wavFile=wave.open(path,'rb') #Use wave module to open .wav file
        self.CHUNK=self.wavFile.getframerate()
        self.sampWidth=self.wavFile.getsampwidth()
    def readWavFile(self,dataCHUNK=None,stream=None):
        #Read 1 sec of audio frames from given wave file and return as np array
        if dataCHUNK==None: dataCHUNK=self.CHUNK
        audioSamples=[]
        fullData=b''
        for frame in range(dataCHUNK):
            #unpack data 1 frame at a time
            audioData=self.wavFile.readframes(1)
            fullData+=audioData
            #Provide option to play audio on speaker
            if stream!=None: stream.write(audioData)
            if len(audioData)>0: #Make sure a frame is read
                #Audio frames store data as short, signed ints
                data=struct.unpack("%dh" % self.sampWidth, audioData)
                audioSamples.append(int(data[0]))
            else: return [] 
        return np.array(audioSamples),fullData

def zeroPadAudio(audioData,windowCenter,factor):
    #Implement zero-phase window zero padding
    #Numpy fft is optimized for N=power of 2
    N=len(audioData)
    nextPowerOfTwo=2**(math.ceil(math.log(factor+N,2)))
    audioData=(audioData[windowCenter:]+[0]*(factor)+
               [0]*(nextPowerOfTwo-factor-N)+
               audioData[:windowCenter])
    return audioData

def besselFun(x,N,meshSize):
    #Returns zeroth order modified bessel fun() of x
    #(of first kind)
    #Io(x)=(1/pi)*integral(e**(xcos(theta))dtheta) from 0 to pi
    #Io(x)=(1/pi)*integral((1/root(1-u**2))*e^(-xu)) from -1 to 1
    besselDeriv=lambda t,u:math.e**(-t*u)/((1-u**2)**0.5)
    #Numerically integrate w/ double exponential transformation
    thetaValue=lambda t:math.tanh(math.pi*0.5*math.sinh(t))
    dThetaValue=lambda t:((1-math.tanh(math.pi*0.5*math.sinh(t))**2)*
                           math.pi*0.5*math.cosh(t))
    fn=lambda t,u: besselDeriv(t,thetaValue(u))*dThetaValue(u)
    lowerLim,upperLim,sample,meshSize,integral=-N,N,-N,meshSize,0
    #Trapezoid rule
    while (sample<upperLim):
        try:
            trapArea=meshSize*(fn(x,sample)+fn(x,sample+meshSize))/2
            integral+=trapArea
            sample+=meshSize
        except:
            #Catch zero division error (arg of fn decays to -1,1 very quickly)
            sample+=meshSize
            continue
    return integral/math.pi

def kaiserWindow(n,windowLength,windowShape,windowRightShift=0):
    #Window function for short time fourier transform
    n-=windowRightShift #Account for window shift
    lowerThreshold=0
    #Center of window is 0.5 windowLength
    upperThreshold=(windowLength-1)
    #Increase windowShape for narrower window
    alpha=windowShape
    if n>upperThreshold or n<lowerThreshold:
        return 0
    else:
        besselMesh=0.1 #meshSize for trapezoid rule (decrease for accuracy)
        besselN=3.2
        upperArg=math.pi*alpha*((1-(2*n/(upperThreshold)-1)**2)**0.5)
        lowerArg=math.pi*alpha
        return (besselFun(upperArg,besselN,besselMesh)/
                besselFun(lowerArg,besselN,besselMesh))

def multDataByKaiserWin(data,kaiserShift,winBeta,winWidth,zeroThreshold=0.05):
    #Apply kaiser window to audio data
    #Approximate Kaiser window with step function to ease computation
    totalWin=len(data)
    steps=50
    N=int(winWidth/steps)
    pointer=0
    kaiserVal=0
    fftSample=[]
    for i in range(0,totalWin):
        if i%N==0:
            kaiserVal=kaiserWindow(i+N/2,winWidth,winBeta,kaiserShift)
            if kaiserVal<zeroThreshold:
                kaiserVal=0
        fftSample.append(data[i]*kaiserVal)
    return fftSample

def shortTimeFourier(data,samplingFreq):
    #Short time fourier transform function
    #Assume music is played at 400 bpm max
    #0.15 sec per beat
    beatTime=0.15 #sec
    beatLength=math.ceil(samplingFreq*beatTime) #Get number of samples per beat
    #minFFTWindow is kaiser window width, each window represents 0.15 sec
    minFFTWindow=int(beatLength) #In number of samples
    sampleSpacing,totalWindow=1/samplingFreq,len(data)
    originalWindowCenter=int(minFFTWindow/2)
    windowBeta=5 #narrow window
    fftFreqs=[]
    #Overlap for windows will be 50%
    windowShift=int(0.5*minFFTWindow)
    shortTimeFT=[]
    #i is window right shift
    for i in range(0,totalWindow-windowShift,windowShift): 
        #Window the audio data with the Kaiser window
        fftSample=multDataByKaiserWin(data,i,windowBeta,minFFTWindow)
        windowCenter=originalWindowCenter+i
        factor=10
        #Zero pad the windowed data (zero phase to eliminate phase shift)
        zeroPadFactor=((factor*minFFTWindow)-minFFTWindow-
                       (totalWindow-minFFTWindow))
        fftSample=zeroPadAudio(fftSample,windowCenter,zeroPadFactor)
        if len(fftFreqs)==0: 
            fftFreqs=np.fft.fftfreq(len(fftSample),sampleSpacing)
        shortTimeFT.append(np.fft.rfft(fftSample))
    return (fftFreqs,shortTimeFT,minFFTWindow,windowShift)

def movingAverage(L,averagePoints):
    #compute symmetric moving average of L
    mid=averagePoints//2
    averageL=np.zeros((len(L),))
    for i in range(len(L)):
        avgSum=0
        pointNum=0
        for j in range(mid+1):
            if j==0 or (i+j)<len(L):
                avgSum+=L[i+j]
                pointNum+=1
            if j!=0 and (i-j)>=0:
                avgSum+=L[i-j]
                pointNum+=1
        averageL[i]=avgSum/pointNum
    return averageL

def stats(L):
    #returns mean and standard deviation of L
    mean=np.sum(L)/len(L)
    residsSquared=(L-mean)**2
    meanResidsSquared=np.sum(residsSquared)/len(residsSquared)
    std=meanResidsSquared**0.5
    return (mean,std)

def diffsBtwnAdjacentVals(L):
    #Calculates average diff btwn adjacent values in L and standard dev.
    numberOfDiffs=len(L)-1
    differences=np.zeros((numberOfDiffs,))
    for i in range(len(L)):
        if (i+1)<len(L):
            difference=abs(L[i+1]-L[i])
            differences[i]=difference
    return stats(differences)
def checkIfBlockContainsMaxima(L,N,i,threshold):
    #Check if a block within list L starting at index i with length N contains
    #a local maximum
    currentBlock,nextBlock,previousBlock=L[i:i+N],L[i+N:i+2*N],L[i-N:i]
    currentMean=np.sum(currentBlock)/len(currentBlock)
    nextMean=np.sum(nextBlock)/len(nextBlock)
    previousMean=np.sum(previousBlock)/len(previousBlock)
    nextDiff,prevDiff=nextMean-currentMean,currentMean-previousMean
    return ((prevDiff>0) and (nextDiff<0) and (abs(nextDiff)>threshold) and
        (abs(prevDiff)>threshold))
def coarseScan(L,N):
    #Coarse scans through list for local maxima, returns indices suspected
    #of harboring maxima within N indices
    suspectIndices=[]
    threshold=0 #Noise threshold
    for i in range(1,len(L),N):
        if (i+2*N)<=len(L) and (i-2*N)>=0:
            if checkIfBlockContainsMaxima(L,N,i,threshold):
                suspectIndices.append(i)
        elif (i-2*N)>=0 and (i+2*N)>len(L): #Boundary condition
            #If block being checked is too close to end of list L...
            currentBlock,previousBlock=L[i:i+N],L[i-N:i]
            currentMean=np.sum(currentBlock)/len(currentBlock)
            previousMean=np.sum(previousBlock)/len(previousBlock)
            diff=currentMean-previousMean
            if diff>0 and diff>threshold: suspectIndices.append(i)
        elif (i+2*N)<=len(L) and (i-2*N)<0: #Boundary condition
            #If block being checked is too close to beginning of list L
            currentBlock,nextBlock=L[i:i+N],L[i+N:i+2*N]
            currentMean=np.sum(currentBlock)/len(currentBlock)
            nextMean=np.sum(nextBlock)/len(nextBlock)
            if (nextMean<currentMean) and (currentMean-nextMean)>threshold:
                suspectIndices.append(i)
        else: print("Error. Enter a smaller N")
    return suspectIndices

def fineScan(L,suspectIndices,N,meanMagnitude,magnitudeSTD,stds):
    #Returns indices associated with peak values in L
    #noiseFloor is minimum value for data to not be considered noise
    maximumIndices,noiseFloor=[],meanMagnitude+stds*magnitudeSTD
    for i in suspectIndices:
        #For each block in L starting at an index in suspectIndices, go through
        #N indices from that index and look for a peak
        for j in range(N):
            index=i+j
            if index<len(L):
                currentVal=L[index]
                if index-1<0:
                    #If current index is 0...
                    nextVal=L[index+1]
                    if (currentVal-nextVal)>0 and L[index]>noiseFloor:
                        maximumIndices.append(index)
                elif index+1>=len(L):
                    #If current index is -1...
                    prevVal=L[index-1]
                    if (currentVal-prevVal)>0 and L[index]>noiseFloor:
                        maximumIndices.append(index)
                else:
                    prevVal,nextVal=L[index-1],L[index+1]
                    if ((currentVal>prevVal) and (currentVal>nextVal) and 
                        L[index]>noiseFloor):
                        maximumIndices.append(index)
    return maximumIndices

def fftPeakDetector(fftFreqs,FFTresults):
    #Detects peaks in FFT results
    magnitude=np.absolute(FFTresults)
    (meanMag,magSTD)=stats(magnitude)
    phase=np.angle(FFTresults)
    peaks=[]
    #Smooth the FFT data
    smoothingFactor=0.05 #percent
    percentConversionFactor=0.01 
    averagePoints=int(len(magnitude)*percentConversionFactor*smoothingFactor)
    smoothed=movingAverage(magnitude,averagePoints)
    (meanDiff,std)=diffsBtwnAdjacentVals(smoothed)
    #Set coarse scan to use blocks of size 10. Set fine scan to use noise
    #threshold of 2 standard deviations
    N,noiseThresh=10,2
    suspectIndices=coarseScan(smoothed,N)
    suspectFreqs=fftFreqs[suspectIndices,]
    smoothedMags=smoothed[suspectIndices,]
    maximaIndices=fineScan(smoothed,suspectIndices,N,meanMag,magSTD,noiseThresh)
    maximaFreqs=fftFreqs[maximaIndices,]
    maxMags=magnitude[maximaIndices,]
    maxPhases=phase[maximaIndices,]
    #Return values associated with max indices
    return (maximaIndices,maximaFreqs,maxMags,maxPhases)

def coarseDiffScanner(val,L,N):
    #Goes through blocks of L, finds block with mean closest to val
    bestDiff=None
    bestIndex=None
    for i in range(0,len(L),N):
        block=L[i:i+N]
        mean=np.sum(block)/N
        diff=abs(mean-val)
        if bestDiff==None or diff<bestDiff:
            bestDiff=diff
            bestIndex=i
    return bestIndex

def fineDiffScanner(val,L,startIndex,N,tolerableFreqDeviation):
    #Returns index of value in L closest to val
    maxIndex=len(L)
    try:
        for i in range(startIndex,startIndex+N):
            if i<maxIndex:
                freq=L[i]
                if abs(freq-val)<tolerableFreqDeviation:
                    return i
            else: break
        return None
    except:
        print("ERROR")

class guideObject(object):
    #guide objects for peak continuation algorithm
    def __init__(self,freq,mag,phase,asleep=False,time=0):
        self.freq=freq
        self.mag=mag
        self.phase=phase
        self.asleep=False
        self.timeSpentAsleep=time
    def get(self):
        return (self.freq,self.mag,self.phase,self.asleep,self.timeSpentAsleep)

def deleteIndicesFromLists(L,i):
    #Delete the elements at index i from each list in L
    newLists=[]
    for j in range(len(L)):
        listForDeletion=L[j]
        np.delete(listForDeletion,[i])
        newLists.append(listForDeletion)
    return tuple(newLists)

def peakContinuation(freqs,STFTdata):
    #Generate guides through peak continuation algorithm from STFT data
    #Each guide object contains freq,mag,phase info on the period of time it
    #corresponds to.
    guides,guideCounter=[],0
    for i in range(len(STFTdata)-1,-1,-1):
        #Go backwards
        (indices,maxFreqs,maxMags,maxPhases)=fftPeakDetector(freqs,STFTdata[i])
        guideDict=dict()
        if len(guides)==0:
            for j in range(len(indices)):
                newKey="g"+str(guideCounter)
                guideCounter+=1
                guideDict[newKey]=guideObject(maxFreqs[j],maxMags[j],
                                              maxPhases[j])
        else:
            prevGuide=guides[0]
            for guide in prevGuide:
                (freq,mag,phase,sleep,timeAsleep),N=prevGuide[guide].get(),3
                if len(maxFreqs)>0:
                    nearInd,freqDev=coarseDiffScanner(freq,maxFreqs,N),10 #hertz
                    nearestInd=fineDiffScanner(freq,maxFreqs,nearInd,N,freqDev)
                else: nearestInd=None
                if nearestInd!=None:
                    (newFreq,newMag,newPhase)=(maxFreqs[nearestInd],
                                             maxMags[nearestInd],
                                             maxPhases[nearestInd])
                    lists=[indices,maxFreqs,maxMags,maxPhases]
                    (indices,maxFreqs,maxMags,maxPhases)=deleteIndicesFromLists(
                                                               lists,nearestInd)
                    guideDict[guide]=guideObject(newFreq,newMag,newPhase)
                else:
                    maxTimeAsleep=3
                    if timeAsleep<maxTimeAsleep:
                        guideDict[guide]=guideObject(freq,mag,phase,
                                                     True,timeAsleep+1)
            for j in range(len(indices)):
                newKey="g"+str(guideCounter)
                guideCounter+=1
                guideDict[newKey]=guideObject(maxFreqs[j],
                                              maxMags[j],maxPhases[j])
        guides=[guideDict]+guides
    return guides

def calculateSMSFun(guideDict,t):
    #Calculate approximate audio value from SMS (sinusoidal) model of audio
    result=0
    for guides in guideDict:
        #For each guideObject corresponding to time t...
        guide=guideDict[guides]
        if not guide.asleep:
            (freq,mag,phase,sleep,timeSleeping)=guide.get()
            angularFreq=2*math.pi*freq
            #Compute the value of the frequency component described by the guide
            component=(mag*math.cos(angularFreq*t+phase))
            result+=component
    return result

def generateModeledAudioData(SMSguides,STFTframeShift,STFTWinWidth,data):
    #Generate sinusoidal model of audio data
    dataLen,maxData=len(data),np.max(data)
    #Init numpy array same size as audio data being modeled
    SMSmodel=np.zeros(dataLen,)
    for i in range(len(SMSguides)):
        #Each guideDict in SMSguides corresponds to a block in time. startDataI
        #is the index corresponding to the start of that block.
        startDataI=i*STFTframeShift
        guideDict=SMSguides[i]
        if i==0: #Account for boundary conditions
            for j in range(STFTframeShift):
                SMSmodel[j]=calculateSMSFun(guideDict,j)
            nextGuideDict=SMSguides[i+1]
            for j in range(STFTframeShift,STFTWinWidth):
                #Guide dict time blocks overlap by 50%, therefore compute AVG
                #of guide dict functions during this overlap
                SMSmodel[j]=(calculateSMSFun(guideDict,j)+
                             calculateSMSFun(nextGuideDict,j))/2
        elif i==(len(SMSguides)-1): #boundary condition
            prevGuideDict=SMSguides[i-1]
            for j in range(startDataI,startDataI+STFTframeShift):
                SMSmodel[j]=(calculateSMSFun(guideDict,j)+
                             calculateSMSFun(prevGuideDict,j))/2
            for j in range(startDataI+STFTframeShift,startDataI+STFTWinWidth):
                if j>=dataLen: break
                SMSmodel[j]=calculateSMSFun(guideDict,j)
        else:
            prevGuideDict,nextGuideDict=SMSguides[i-1],SMSguides[i+1]
            for j in range(startDataI,startDataI+STFTWinWidth):
                if j<(startDataI+STFTframeShift):
                    SMSmodel[j]=(calculateSMSFun(guideDict,j)+
                             calculateSMSFun(prevGuideDict,j))/2
                else:
                    SMSmodel[j]=(calculateSMSFun(guideDict,j)+
                                 calculateSMSFun(nextGuideDict,j))/2
    #Calculate last index covered by Sinusoidal modeling of audio data
    endI=(len(SMSguides)-1)*STFTframeShift+STFTWinWidth
    #If any remaining empty spots in SMSmodel array, fill with values from 
    #function in the last dictionary of guides
    finalGuideDict=SMSguides[-1]
    for i in range(endI,dataLen):
        SMSmodel[i]=calculateSMSFun(finalGuideDict,i)
    #Scale the SMS array to the actual audio data
    maxSMS=np.max(SMSmodel)
    scalingFactor=maxData/maxSMS
    return SMSmodel*scalingFactor
        
def peakDetectorFilter(L,N):
    #Envelope detector (NOT TO BE CONFUSED with local maxima)
    shift=int(N/2)
    envelope=np.zeros(len(L),)
    for i in range(len(L)):
        if (i-shift)>=0:
            peak=np.max(L[i-shift:i+shift])
        else:
            peak=np.max(L[:i+shift])
        envelope[i]=peak
    return envelope

def butterWorthGain(freq,fc,DCgain,order):
    #Low pass filter gain
    gain=DCgain/((1+(freq/fc)**(2*order))**0.5)
    return gain

def butterWorthFilter(signal,sampleRate,order,DCgain):
    #Low pass filter
    scalingFactor=0.0002
    fc=scalingFactor*sampleRate
    fftSig=np.fft.rfft(signal)
    filteredSig=[]
    sampleSpacing=1/sampleRate
    fftFreqs=np.fft.fftfreq(len(signal),sampleSpacing)
    for i in range(len(fftSig)):
        nextPoint=fftSig[i]*butterWorthGain(fftFreqs[i],fc,DCgain,order)
        filteredSig.append(nextPoint)
    smoothingPoints=200
    filteredSig=np.array(filteredSig)
    return movingAverage(np.fft.irfft(filteredSig),smoothingPoints)

#http://www.cs.cmu.edu/~112/notes/notes-strings.html
def readFile(path):
    with open(path, "rt") as f:
        return f.readlines()

def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)

def appendFile(path, contents):
    with open(path, "a") as f:
        f.write(contents)
def findPeakResidualIndices(data,samplingFreq):
    #Take in array of audiodata, model it with a sum of sinusoids, calculate
    #error in the model, find peaks in error, and return indices of those peaks
    envSmoothing=60
    #Compute short time Fourier transform (STFT)
    (fftfreq,STFT,STFTWinWidth,winShift)=shortTimeFourier(data,samplingFreq)
    #Generate guides containing magnitude, freq, phase info from STFT
    guides=peakContinuation(fftfreq,STFT)
    #Calculate audio data values using guide info as a model
    SMSmodel=generateModeledAudioData(guides,winShift,STFTWinWidth,data)
    #Calculate error by subtracting SMSmodel from actual audio data.
    filteredResids=peakDetectorFilter(np.absolute(data-SMSmodel),envSmoothing)
    lowPassResids=butterWorthFilter(filteredResids,samplingFreq,1,1)
    N=10
    (mean,std)=stats(lowPassResids)
    #Find error peaks and return their indices
    nearPeakIndices=coarseScan(lowPassResids,N)
    noiseThresh=1.2
    peakIndices=fineScan(lowPassResids,nearPeakIndices,N,mean,std,noiseThresh)
    return peakIndices

def writeMusicResidualsToFile(targetFile,audioFile):
    #Take in audio file, calculate SMS model of audio, calculate error, write
    #error to targetFile
    writeFile(targetFile,"") #Wipe file
    audioReader=audioHandler(audioFile)
    samplingFreq=audioReader.wavFile.getframerate()
    #Write audio file name to target file
    appendFile(targetFile,str(audioFile)+'\n')
    #Read new audio data from audioFile
    data=audioReader.readWavFile()[0]
    audioFramesRead,sampleSpacing,counter=audioReader.CHUNK,1/samplingFreq,0
    while len(data)>0:
        #Find the indices associated with peaks in the error in the SMS model
        peakIndices=findPeakResidualIndices(data,samplingFreq)
        for i in peakIndices:
            #Calculate the time associated with each error peak
            timeStampWithinWindow=i*sampleSpacing
            startTime=counter*audioFramesRead*sampleSpacing #sec
            timeStamp=startTime+timeStampWithinWindow
            (dataMean,dataSTD)=stats(data)
            dataVal=data[i]
            #If the audio volume is high, add the timeStamp to targetFile
            if dataVal>(dataMean+dataSTD):
                peakString=(str(timeStamp)+","+str(dataVal)+","+
                            str(dataMean)+","+str(dataSTD)+"\n")
                appendFile(targetFile,peakString)
        counter+=1
        data=audioReader.readWavFile()[0]
################################################################################
#Audio thread classes
################################################################################
class audioPlayer(threading.Thread):
    #Audio player class for game
    def __init__(self,queue,output,stream):
        #Inits a thread
        threading.Thread.__init__(self)
        self.queue=queue
        self.output=output
        self.stream=stream
        self.isRunning=True
        self.paused=False
        self.pauseCondition=threading.Condition(threading.Lock())
        self.stopped=False
    def run(self):
        #While thread is running, takes in audio data from self.queue, plays it,
        #and writes it to the output queue
        while self.isRunning:
            with self.pauseCondition: #If thread paused, wait.
                while self.paused:
                    self.pauseCondition.wait()
            (numpyInfo,data)=self.queue.get()
            if self.output.qsize()>0:
                self.output.get()
            #Update output queue with latest data being played
            self.output.put(numpyInfo)
            self.stream.write(data)
            self.queue.task_done()
        self.stopped=True
    def stop(self): #Stop the thread
        self.isRunning=False
    def pause(self): #Pause the thread
        self.paused=True
        self.pauseCondition.acquire()
    def resume(self): #Resume the thread
        self.paused=False
        self.pauseCondition.notify()
        self.pauseCondition.release()

class audioProcessor(threading.Thread):
    #Audio processing class for loading new wav files
    def __init__(self,queue,textFile):
        threading.Thread.__init__(self)
        self.queue=queue
        self.textFile=textFile
        self.isRunning=True
    def run(self):
        while self.isRunning:
            #While thread is running, get a new audio file and process it.
            newAudioFile=self.queue.get()
            writeMusicResidualsToFile(self.textFile,newAudioFile)
            self.queue.task_done()
            self.stop()
    def stop(self):
        self.isRunning=False

################################################################################
#File processing
################################################################################
def almostIn(n,L,epsilon=0.05):
    #Checks if n is within epsilon of any elements in L
    for entry in L:
        if abs(entry-n)<epsilon:
            return (True,entry)
    return (False,None)

def removeSimilarEntries(path):
    #Opens up 'beats.txt' file, removes beat entries that are too close together
    textFile=open(path,'r')
    fileLines=textFile.readlines()
    textFile.close()
    writeFile(path,"") #Clear file
    appendFile(path,fileLines.pop(0)) #First line is audio file title
    encounteredTimeStamps=[]
    for lines in fileLines:
        timeStamp=float(lines.split(',')[0])
        if not almostIn(timeStamp,encounteredTimeStamps)[0]:
            encounteredTimeStamps.append(timeStamp)
            appendFile(path,lines)

################################################################################
#Graphics
################################################################################

def findMaxesOfList(L,numberOfMaxes): #For complex numbers
    #Goes through list of complex numbers, finds n=numberOfMaxes numbers with
    #largest magnitudes 
    maxNumbers=[]
    absL=np.absolute(L) #Find list of magnitudes of complex numbers in L
    for i in range(numberOfMaxes):
        maxIndex=np.nonzero(absL==max(absL))[0][0]
        maximum=L[maxIndex]
        maxNumbers.append(maxIndex)
        L=np.delete(L,maxIndex)
        absL=np.delete(absL,maxIndex)
    return maxNumbers

def mapListToMax(L,maximumValAllowed):
    #Scales values in L to maximumValAllowed
    scalingFactor=maximumValAllowed/np.max(L)
    return L*scalingFactor

class Slider(object):
    #DIY slider widget in pygame
    def initSliderBar(self,pygameSurface,x,y):
        #Initializes slider bar
        self.barX=x
        self.barY=y
        barHeightScaling,barWidthScaling=0.75,0.05
        self.barHeight=pygameSurface.get_height()*barHeightScaling
        self.barWidth=pygameSurface.get_width()*barWidthScaling
        self.barColor=(230,255,255)
        self.barRect=(self.barX,self.barY,self.barWidth,
                      self.barHeight)
    def initSlider(self,pygameSurface):
        #Initializes slider position and dimensions
        self.sliderNumber=0 #0-20
        self.sliderMax=20
        heightScaling,widthScaling=0.025,0.1
        self.sliderHeight=pygameSurface.get_height()*heightScaling
        self.sliderWidth=pygameSurface.get_width()*widthScaling
        self.sliderColor=(255,150,150)
        self.sliderX=(self.barX+self.barWidth/2)-self.sliderWidth/2
        self.sliderMaxY=(self.barHeight+self.barY)-self.sliderHeight/2
        self.sliderMinY=self.barY-self.sliderHeight/2
        self.sliderRange=self.sliderMaxY-self.sliderMinY
        self.sliderY=self.sliderMaxY
        self.sliderRect=(self.sliderX,self.sliderY,self.sliderWidth
                         ,self.sliderHeight)
    def __init__(self,pygameSurface,surfaceX,surfaceY,x,y):
        self.clicked=False
        self.sliderSurface=pygameSurface
        self.surfaceX,self.surfaceY=surfaceX,surfaceY
        #Set slider bar attributes
        self.initSliderBar(pygameSurface,x,y)
        #Init slider 
        self.initSlider(pygameSurface)
        #Init bounding box
        (self.boxWidth,self.boxHeight)=(self.sliderWidth,
                                        self.barHeight+
                                        self.sliderHeight)
        self.boxX,self.boxY=(self.sliderX),(self.barY-self.sliderHeight/2)
        self.boxColor=(150,200,230)
        self.box=(self.boxX,self.boxY,self.boxWidth,self.boxHeight)

    def drawSlider(self,surface):
        #Draw the slider object on surface
        pygame.draw.rect(surface,self.boxColor,self.box)
        pygame.draw.rect(surface,self.barColor,self.barRect)
        pygame.draw.rect(surface,self.sliderColor,self.sliderRect)
        
    def changeSliderPosition(self,newY):
        #Change slider widget position to newY
        newY=newY-self.surfaceY
        if newY>self.sliderMaxY:
            #If newY is too low, set slider to lowest position
            self.sliderY=self.sliderMaxY
            self.sliderNumber=0
        elif newY<self.sliderMinY:
            #If newY is too high, set slider to highest position
            self.sliderY=self.sliderMinY
            self.sliderNumber=self.sliderMax
        else:
            #Compute new slider number for slider position
            self.sliderY=newY
            self.sliderNumber=(self.sliderMax*(self.sliderMaxY-
                               (newY+self.sliderHeight/2))/self.sliderRange)
        self.sliderRect=(self.sliderX,self.sliderY,self.sliderWidth
                         ,self.sliderHeight)

    def checkIfClickedOn(self,mouseX,mouseY):
        #Check if slider has been clicked on
        mouseX=mouseX-self.surfaceX
        mouseY=mouseY-self.surfaceY
        if mouseX>self.sliderX and mouseX<(self.sliderX+self.sliderWidth):
            if mouseY>self.sliderY and mouseY<(self.sliderY+self.sliderHeight):
                return True
        return False

class fourierPad(object):
    #Primary game objects (bubbles to be popped)
    def __init__(self,x,y,r,freqs,mags,phases,sliders,
                 clicked=False,active=True,dying=False):
        self.dying=dying
        self.active=active
        self.clicked=clicked
        self.x=x
        self.y=y
        self.cy=y #y oscillates around cy
        self.r=r
        self.maxAmps=20 #Amplitudes of frequency components are at most 20
        self.freqs=freqs
        self.mags=[]
        #Scale magnitudes to a max of 20
        maxMags=max(mags)
        for mag in mags:
            self.mags.append(mag*self.maxAmps/maxMags)
        self.phases=phases 
        self.sliders=sliders
        #Calculate maximum possible error for player
        self.maxError=sum([max(self.mags[i],self.maxAmps-self.mags[i]) 
                           for i in range(len(self.mags))])
        #Generate coordinate list to be drawn
        self.audioWaveFormCoords=self.generateWaveFormPoints()
        self.playerCoords=self.generatePlayerPoints()
    def scorePlayer(self):
        #Gives the player a score
        error=0
        for i in range(len(self.sliders)):
            error+=abs(self.sliders[i].sliderNumber-self.mags[i])
        scorePercentage=1-error/self.maxError
        lowScore,intermediateScore,highScore=10,50,100
        zeroThreshold,lowThreshold,mediumThreshold=0.3,0.7,0.9
        if scorePercentage<zeroThreshold:
            return 0
        elif scorePercentage<lowThreshold:
            return lowScore
        elif scorePercentage<mediumThreshold:
            return intermediateScore
        else:
            return highScore

    def checkIfClickedOn(self,mouseX,mouseY,gameScreenX,gameScreenY):
        #Check if pad has been clicked on
        mouseX-=gameScreenX
        mouseY-=gameScreenY
        distance=((mouseX-self.x)**2+(mouseY-self.y)**2)**0.5
        #Use distance formula to check distance btwn mouse click and pad center
        if distance<self.r:
            return True
        return False

    def waveFormFunction(self,t):
        #Calculates the value of the pad's waveform as a function of t
        result=0
        for i in range(len(self.freqs)):
            angular=2*math.pi*self.freqs[i]
            result+=self.mags[i]*math.cos(angular*t+self.phases[i])
        return result

    def playerFunction(self,t):
        #Calculates the value of the player waveform as a function of t
        result=0
        for i in range(len(self.freqs)):
            mag=self.sliders[i].sliderNumber
            angular=2*math.pi*self.freqs[i]
            result+=mag*math.cos(angular*t+self.phases[i])
        return result

    def generateWaveFormPoints(self,isPlayer=False):
        #Generate pygame coordinates for fourier pad waveforms
        coordinates,t,step,maxPercentOfR=[],0,0.03,0.2
        periodsToGraph=8
        while t<2*math.pi:
            #Generate coordinates for four periods of the waveform
            freq=min(self.freqs)
            period=1/freq
            if not isPlayer:
                #Generate coordinates for pad waveform
                result=self.waveFormFunction(t*periodsToGraph*
                                             period/(2*math.pi))
            else:
                #Generate coordinates for player waveform
                result=self.playerFunction(t*periodsToGraph*period/(2*math.pi))
            scalingFactor=self.r*maxPercentOfR/(len(self.freqs)*self.maxAmps)
            #Map waveform function value to radius of circular parametric
            #function.
            radius=self.r-result*scalingFactor
            x=radius*math.cos(t)+self.x
            y=-radius*math.sin(t)+self.y
            coordinates.append((x,y))
            t+=step
        return np.array(coordinates)

    def generatePlayerPoints(self):
        #Generate coordinates for player waveform
        return self.generateWaveFormPoints(True)

    def drawHighlight(self,gameSurface,highlightColor,width):
        #If the pad has been clicked on, highlight it
        pygame.draw.lines(gameSurface,highlightColor,
                         True,self.audioWaveFormCoords,width)
        pygame.draw.lines(gameSurface,highlightColor,
                         True,self.playerCoords,width)

    def generatePadColors(self):
        #Generate the colors of the bubble
        highlightColor=(255,235,59)
        baseColor=(90,255,100)
        playerScore=self.scorePlayer()
        fullG=255
        fullScore=100
        G=int(playerScore*fullG/fullScore)
        playerColor=list(baseColor) #playerColor is color of player waveform
        playerColor[1]=G
        playerColor=tuple(playerColor)
        padColor=(245,0,87) #padColor is color of pad audio waveform
        return (padColor,playerColor,highlightColor)

    def drawPad(self,gameSurface):
        #Draw fourier pad as bubble
        if not self.dying:
            #If bubble has not popped
            if self.checkIfSlidersClicked():
                #If the fourier pad's sliders are being moved, update the 
                #player waveform coordinates
                playerCoords=self.generatePlayerPoints()
                self.playerCoords=playerCoords
            else:
                playerCoords=self.playerCoords
            #Generate colors to be used
            #Player waveform will turn green as it becomes more accurate 
            padColor,playerColor,highlight=self.generatePadColors()
            if self.clicked:
                #Highlight the bubble if it has been clicked on
                highlightWidth=3
                self.drawHighlight(gameSurface,highlight,highlightWidth) 
            pygame.draw.lines(gameSurface,padColor,True,
                              self.audioWaveFormCoords,1)
            pygame.draw.lines(gameSurface,playerColor,True,playerCoords,1)
        else:
            #If bubble has been popped, draw it as a collapsing circle
            circleWidth,dr=1,5
            pygame.draw.circle(gameSurface,(255,23,68),
                               (int(self.x),int(self.y)),self.r,circleWidth)
            self.r-=dr
            if self.r<circleWidth:
                self.active=False

    def shiftPad(self,dx,dy):
        #Shift the fourier pad by dx, dy
        waveFormMaxModulation=0.2
        self.x+=dx
        oldY=self.y
        self.y=self.cy+dy
        self.audioWaveFormCoords[:,0]+=dx
        self.audioWaveFormCoords[:,1]-=(oldY-self.y)
        self.playerCoords[:,0]+=dx
        self.playerCoords[:,1]-=(oldY-self.y)
        if (self.x+self.r*(1+waveFormMaxModulation))<0:
            #If the pad has been moved past the boundary of the game display,
            #pop its bubble.
            self.dying=True

    def checkIfSlidersClicked(self):
        #Check if any of the sliders associated with the fourier pad have been
        #clicked.
        for slider in self.sliders:
            if slider.clicked:
                return True
        return False            




class fourierGame(object):
    #Main game object
    def gameScreenSetUp(self):
        #Set up game screens
        gameDisplay1WidthScaling=3/4
        self.gameDisplay1Width=self.gameScreenWidth*gameDisplay1WidthScaling
        self.mainBackground=pygame.Surface(self.gameScreen.get_size())
        self.gameDisplay1=pygame.Surface((self.gameDisplay1Width,
                                         self.gameScreenHeight))
        self.gameDisplayOverlay=pygame.Surface((self.gameDisplay1Width,
                                                self.gameScreenHeight))
        playerControlWidthScaling=1/4
        self.controlScreenWidth=self.gameScreenWidth*playerControlWidthScaling
        ctrlScreenHeightScaling=5/6
        ctrlTitleHeightScaling=1/6
        self.controlScreenHeight=self.gameScreenHeight*ctrlScreenHeightScaling
        self.titleScreenHeight=self.gameScreenHeight*ctrlTitleHeightScaling
        self.playerControlScreen=pygame.Surface((self.controlScreenWidth,
                                                 self.controlScreenHeight))
        self.playerControlTitle=pygame.Surface((self.controlScreenWidth,
                                                self.titleScreenHeight))
        self.gameOverlay=pygame.Surface((self.gameScreenWidth,
                                         self.gameScreenHeight))
        gameDisplayOverlayColor=(197,17,98)
        self.gameDisplayOverlay.fill(gameDisplayOverlayColor)
        self.gameDisplayOverlay=self.gameDisplayOverlay.convert()
    def fileManagerScreenSetUp(self):
        #Set up file manager screens
        fileTitleHeightScaling=1/10
        fileManagerHeightScaling=9/10
        self.fileManagerTitleHeight=fileTitleHeightScaling*self.gameScreenHeight
        self.fileManagerHeight=fileManagerHeightScaling*self.gameScreenHeight
        self.fileManagerTitle=pygame.Surface((self.gameScreenWidth,
                                              self.fileManagerTitleHeight))
        self.fileManager=pygame.Surface((self.gameScreenWidth,
                                         self.fileManagerHeight))
    def transitionScreenSetUp(self):
        #Set up loading and splash screen
        self.loadingScreen=pygame.Surface((self.gameScreenWidth,
                                           self.gameScreenHeight))
        self.splashScreen=pygame.Surface((self.gameScreenWidth,
                                          self.gameScreenHeight))
    def helpScreenSetUp(self):
        #Sets up help screen
        self.helpBackground=pygame.Surface((self.gameScreenWidth,
                                            self.gameScreenHeight))
        self.helpForeground=pygame.Surface((self.gameScreenWidth,
                                            self.gameScreenHeight))
        self.helpBackground.fill((197,17,98))
        self.helpBackground=self.helpBackground.convert()
        self.helpForeground.fill((255,255,255))
        self.helpForegroundAlpha=200
        self.helpForegroundDAlpha=-0.25
    def screenSetUp(self):
        #Init game screen, game display surfaces
        self.computerScreenWidth=1920
        self.computerScreenHeight=1080
        self.gameScreen=pygame.display.set_mode((self.computerScreenWidth//2,
                                            self.computerScreenHeight//2))
        (self.gameScreenWidth,self.gameScreenHeight)=self.gameScreen.get_size()
        self.gameScreenSetUp()
        self.fileManagerScreenSetUp()
        self.transitionScreenSetUp()
        self.helpScreenSetUp()
    def fontSetUp(self):
        #Set up fonts
        small,medium,intermediate,large=10,15,20,35
        self.smallGameFont=pygame.font.SysFont("Calibri", small)
        self.gameFont=pygame.font.SysFont("Calibri", medium)
        self.fileManagerTitleFont=pygame.font.SysFont("Calibri",intermediate)
        self.splashScreenTitleFont=pygame.font.SysFont("Calibri",large)
        self.instructionFont=pygame.font.SysFont("Calibri",intermediate) 

    def messageSetUp(self):
        #setup game messages
        self.gameOverMessage=self.splashScreenTitleFont.render("Game Over",
                                                               1,(255,255,255))
        self.gameOverMessage=self.gameOverMessage.convert_alpha()
        self.scoreMessage=""
    def setGameParams(self):
        #Set game parameters
        self.opaque=255
        self.semiTransparent=100
        self.gameDisplay1Alpha=self.semiTransparent
        self.playerControlScreenAlpha=self.opaque
        self.playerControlTitleAlpha=self.opaque
        self.gameRunning=True
    def initFileManager(self):
        #Initialize file manager
        self.waveFiles=[]
        self.txtFile='beats.txt'
        self.textFileX=0
        self.textFileY=0
        self.textFileWidth=self.fileManager.get_width()
        textFileScaling=1/9
        self.textFileHeight=self.fileManager.get_height()*textFileScaling
        self.waveRectHeight=0
        self.noFiles=True
    def setUpFileButton(self):
        self.buttonWidth=40
        self.buttonHeight=20
        self.buttonX=self.gameDisplay1.get_width()-self.buttonWidth
        self.buttonY=0
        self.buttonColor=(245,0,87)        
    
    def __init__(self):
        pygame.init()
        self.fontSetUp()
        self.screenSetUp()
        self.messageSetUp()
        self.gameMode="splashScreen"
        self.freeze=False        
        self.setGameParams()
        self.initFileManager()
        self.splashScreenInit()
        self.setUpFileButton()
        self.drawAll()
        
        
    def loadingInit(self,audioFile):
        #Init loading screen
        self.processingComplete=False
        self.audioProcessingQueue=Queue()
        self.audioProcessingQueue.put(audioFile)
        self.processorDaemon=audioProcessor(self.audioProcessingQueue,
                                            self.txtFile)
        self.processorDaemon.setDaemon(True)
        self.processorDaemon.start()
        self.loadingScreenColor=(0,0,0)
        self.loadAlpha=255
        self.dAlpha=-1
        loadingText="Loading...this may take some time."
        textColor=(255,255,255)
        self.loadTextSurf=self.fileManagerTitleFont.render(loadingText,1,
                                                       textColor)
    def setUpFourierPads(self):
        #Set up fourier pads
        self.fourierPads=[]
        self.fourierPadSide=-1
        self.fourierPadRadius=60
        self.fourierPadXStart=(self.gameDisplay1.get_width()-
                               self.fourierPadRadius)
        self.fourierFloatAmp=40
        numberOfPeriods=4
        self.fourierPadPeriod=(1/numberOfPeriods)*self.gameDisplay1Width
        self.fourierOmega=(1/self.fourierPadPeriod)*2*math.pi
        fourierPadMaxModulation=0.2
        self.minDistanceFromTop=self.fourierPadRadius*(1+
                                                      fourierPadMaxModulation)
        self.fourierPadYStart=random.randint(self.minDistanceFromTop+
                                             self.fourierFloatAmp,
                                             self.gameScreenHeight-
                                             self.minDistanceFromTop-
                                             self.fourierFloatAmp)
        self.oldPadYStart=None
    def setUpGameAudio(self):
        #Set up game audio
        self.beats=readFile(self.txtFile)
        self.audioFile=self.beats.pop(0).strip()
        self.beatTimes=[float(line.split(',')[0]) for line in self.beats]
        self.audioObject=audioHandler(self.audioFile)
        self.samplingRate=self.audioObject.wavFile.getframerate()
        self.numberOfSamples=self.audioObject.wavFile.getnframes()
        self.musicDuration=self.numberOfSamples/self.samplingRate
        self.notesFound=[]
        #Open audio stream
        self.stream=self.pyAudioObj.open(
                           format=self.pyAudioObj.get_format_from_width(
                           self.audioObject.wavFile.getsampwidth()),
                           channels=self.audioObject.wavFile.getnchannels(),
                           rate=self.audioObject.wavFile.getframerate(),
                           output=True)
    def setUpAudioPlayer(self):
        #Set up audio player
        self.audioQueue=Queue()
        self.fftQueue=Queue()
        self.audioPlayer=audioPlayer(self.audioQueue,self.fftQueue,self.stream)
        self.audioPlayer.setDaemon(True)
        self.audioPlayer.start()
    def gameInit(self):
        self.gameOverlayAlpha=0
        self.gameDisplayOverlayAlpha=0
        self.gameOverlayAlphaV=-1
        self.gameOverlayAlphaBeat=25
        self.gameOver=False
        self.drawBackground(False)
        #Initialize game
        self.setUpFourierPads()
        self.score=0
        self.fails=0
        self.pyAudioObj=pyaudio.PyAudio()
        removeSimilarEntries(self.txtFile)
        self.setUpGameAudio()
        self.clock=pygame.time.Clock()
        self.startTime=pygame.time.get_ticks()
        self.setUpAudioPlayer()
        self.buttonText=self.smallGameFont.render("New File",1,(0,0,0))
        self.buttonText=self.buttonText.convert_alpha()
    
    def splashScreenInit(self):
        #Initialize splash screen
        gameTitle="Bubbles"
        instructionLineOne="Select bubbles and pop them with the sliders!"
        instructionLineTwo="Press h for help. Click to start."
        self.splashScreenAlpha=255
        self.splashScreenDAlpha=-1
        self.splashScreenColor=(0,0,0)
        textColor=(255,255,255)
        self.gameTitle=self.splashScreenTitleFont.render(gameTitle,1,
                                                         textColor)
        self.gameInstrOne=self.instructionFont.render(instructionLineOne,
                                                            1,textColor)
        self.gameInstrTwo=self.instructionFont.render(instructionLineTwo,
                                                            1,textColor)
    def drawMainBackground(self,update=False):
        #Draw game background
        backgroundColor=(255,255,255)
        if not update: self.mainBackground.fill(backgroundColor)
        self.mainBackground=self.mainBackground.convert()
        self.gameScreen.blit(self.mainBackground,(0,0))
    
    def drawGameDisplay1(self,update=False):
        #Draw large upper right display
        displayColor=(255,255,255)
        if not update: self.gameDisplay1.fill(displayColor)
        self.gameDisplay1.set_alpha(self.gameDisplay1Alpha)
        self.gameDisplay1=self.gameDisplay1.convert()
        self.gameScreen.blit(self.gameDisplay1,(self.controlScreenWidth,0))
        self.gameDisplayOverlay.set_alpha(self.gameDisplayOverlayAlpha)
        self.gameDisplayOverlay=self.gameDisplayOverlay.convert()
        self.gameScreen.blit(self.gameDisplayOverlay,(self.controlScreenWidth,0))
    
    def drawPlayerControlScreen(self,update=False):
        #Draw player control display
        controlColor=(255,220,255)
        if not update: self.playerControlScreen.fill(controlColor)
        self.playerControlScreen.set_alpha(self.playerControlScreenAlpha)
        self.playerControlScreen=self.playerControlScreen.convert()
        self.gameScreen.blit(self.playerControlScreen,
                             (0,self.titleScreenHeight))

    def drawPlayerControlTitle(self,update=False):
        #Draw label display for player control display
        titleColor=(255,240,255)
        if not update: self.playerControlTitle.fill(titleColor)
        self.playerControlTitle.set_alpha(self.playerControlTitleAlpha)
        self.playerControlTitle=self.playerControlTitle.convert()
        self.gameScreen.blit(self.playerControlTitle,(0,0))

    def drawGameOverlay(self):
        overlayColor=(0,0,0)
        self.gameOverlay.fill(overlayColor)
        self.gameOverlay.set_alpha(self.gameOverlayAlpha)
        self.gameOverlay=self.gameOverlay.convert()
        self.gameScreen.blit(self.gameOverlay,(0,0))

    def drawBackground(self,update):
        #Draw the game background
        self.drawMainBackground(update)
        self.drawGameDisplay1(update)
        self.drawPlayerControlScreen(update)
        self.drawPlayerControlTitle(update)
    def drawManagerTitle(self):
        #Draw file manager title
        titleColor=(255,128,171)
        self.fileManagerTitle.fill(titleColor)
        self.fileManagerTitle=self.fileManagerTitle.convert()
        self.gameScreen.blit(self.fileManagerTitle,(0,0))
        titleText=("Before playing, you must first process an audio file."+
                   " If a file has already been processed, select 'beats.txt'.")
        textColor=(56,142,60)
        text=self.fileManagerTitleFont.render(titleText,1,textColor)
        textX=0
        textY=self.fileManagerTitle.get_height()/2-text.get_height()/2
        self.gameScreen.blit(text,(textX,textY))
    def drawManagerRects(self):
        #Draw rectangles for file manager
        textFileColor=(185,246,202)
        pygame.draw.rect(self.fileManager,textFileColor,
                        (self.textFileX,self.textFileY,
                         self.textFileWidth,self.textFileHeight))
        numberOfAudioFiles=len(self.waveFiles)
        audioText=[]
        rectStartY=self.textFileY+self.textFileHeight
        for i in range(numberOfAudioFiles):
            waveFileTitle=self.gameFont.render(self.waveFiles[i],1,(0,0,0))
            audioText.append(waveFileTitle)
            waveRectWidth=self.fileManager.get_width()
            waveY=rectStartY+i*self.waveRectHeight
            baseG,dG=221,25
            G=baseG-dG*i
            G=0 if G<0 else G
            rectColor=(185,G,202)
            pygame.draw.rect(self.fileManager,rectColor,(0,waveY,waveRectWidth
                             ,self.waveRectHeight))
        return audioText
    def drawManager(self):
        #Draw file manager
        self.drawManagerTitle()
        fileManagerBackground=(128,216,255)
        self.fileManager.fill(fileManagerBackground)
        if not self.noFiles:
            audioText=self.drawManagerRects()
            #Draw file names
            textFileText=self.gameFont.render(self.txtFile,1,(0,0,0))
        fileManagerY=self.fileManagerTitle.get_height()
        self.gameScreen.blit(self.fileManager,(0,fileManagerY))
        if not self.noFiles:
            textFileTextY=(fileManagerY+self.textFileHeight/2-
                       textFileText.get_height()/2)
            self.gameScreen.blit(textFileText,(self.fileManager.get_width()/2,
                                           textFileTextY))
        numberOfAudioFiles=len(self.waveFiles)
        for i in range(numberOfAudioFiles):
            audioFileName=audioText[i]
            blitX=self.fileManager.get_width()/2
            blitY=(fileManagerY+self.textFileHeight+self.waveRectHeight/2+
                   i*self.waveRectHeight-audioFileName.get_height()/2)
            self.gameScreen.blit(audioFileName,(blitX,blitY))
    def drawFourierPads(self):
        #Draw fourier pads as bubbles
        for i in range(len(self.fourierPads)-1,-1,-1):
            pad=self.fourierPads[i]
            if pad.clicked:
                for slider in pad.sliders:
                    slider.drawSlider(self.playerControlScreen)
            pad.drawPad(self.gameDisplay1)
    def drawGameText(self):
        #Draw all game text
        scoreText=self.fileManagerTitleFont.render(str(self.score),1,(0,0,0))
        scoreTextX=self.playerControlTitle.get_width()/2-scoreText.get_width()/2
        scoreTextY=(self.playerControlTitle.get_height()/2-
                    scoreText.get_height()/2)
        scoreText=scoreText.convert_alpha()
        self.gameScreen.blit(scoreText,(scoreTextX,scoreTextY))
        buttonTextX=(self.gameScreenWidth-self.buttonWidth/2-
                     self.buttonText.get_width()/2)
        buttonTextY=self.buttonHeight/2-self.buttonText.get_height()/2
        self.gameScreen.blit(self.buttonText,(buttonTextX,buttonTextY))
    def drawGameOverText(self):
        #Draw game over text
        self.drawGameOverlay()
        gameOverX=(self.gameScreen.get_width()/2-
                   self.gameOverMessage.get_width()/2)
        gameOverY=(self.gameScreen.get_height()/2-
                   self.gameOverMessage.get_height()/2)
        scoreX=(self.gameScreen.get_width()/2-
                self.scoreMessage.get_width()/2)
        scoreY=(gameOverY+self.gameOverMessage.get_height())
        self.gameScreen.blit(self.gameOverMessage,(gameOverX,gameOverY))
        self.gameScreen.blit(self.scoreMessage,(scoreX,scoreY))

    def drawGame(self):
        #Draw the game
        self.drawBackground(False)
        self.drawFourierPads()
        #Draw new file button
        pygame.draw.rect(self.gameDisplay1,self.buttonColor,
                        (self.buttonX,self.buttonY,self.buttonWidth,
                         self.buttonHeight))
        self.drawBackground(True)
        self.drawGameText()
        if self.gameOver:
            self.drawGameOverText()
        
    def drawLoadingScreen(self):
        #Draw loading screen
        textx=self.gameScreenWidth/2-self.loadTextSurf.get_width()/2
        texty=self.gameScreenHeight/2-self.loadTextSurf.get_height()/2
        self.gameScreen.blit(self.loadTextSurf,(textx,texty))
        self.loadingScreen.fill((0,0,0))
        self.loadingScreen.set_alpha(self.loadAlpha)
        self.gameScreen.blit(self.loadingScreen,(0,0))

    def drawSplashScreen(self):
        #Draw splash screen
        titleHeight=self.gameTitle.get_height()
        titleWidth=self.gameTitle.get_width()
        lineOneHeight=self.gameInstrOne.get_height()
        lineOneWidth=self.gameInstrOne.get_width()
        lineTwoHeight=self.gameInstrTwo.get_height()
        lineTwoWidth=self.gameInstrTwo.get_width()
        cx,cy=self.gameScreen.get_width()/2,self.gameScreen.get_height()/2
        titleX=cx-titleWidth/2
        lineOneX=cx-lineOneWidth/2
        lineTwoX=cx-lineTwoWidth/2
        titleY=cy-titleHeight/2
        lineOneY=titleY+titleHeight
        lineTwoY=lineOneY+lineOneHeight
        self.gameScreen.blit(self.gameTitle,(titleX,titleY))
        self.gameScreen.blit(self.gameInstrOne,(lineOneX,lineOneY))
        self.gameScreen.blit(self.gameInstrTwo,(lineTwoX,lineTwoY))
        self.splashScreen.fill((0,0,0))
        self.splashScreen.set_alpha(self.splashScreenAlpha)
        self.gameScreen.blit(self.splashScreen,(0,0))
    def helpInstructionsPartOne(self):
        instrLineOne="Click inside a bubble to select it."
        instrLineTwo="Each bubble is composed of two circular components."
        instrLineThree=("One component is the approximation of the audio"+
                        " waveform when the bubble was formed.")
        instrLineFour=("This approximation was made by summing the two largest"
                       +" frequency components of the audio.")
        instrLineFive=("The other component is your approximation of the audio "
                       +"waveform when the bubble was formed.")
        return [instrLineOne,instrLineTwo,instrLineThree,
                instrLineFour,instrLineFive]
    def getHelpInstructions(self):
        instrLineSix=("To pop a bubble, adjust the sliders on the side until"
                      +" your approximation matches the game's.")
        instrLineSeven=("Your approximation consists of the sum of the two"+
                        " largest frequency components of the audio.")
        instrLineEight=("By adjusting the sliders, you adjust the amplitudes"+
                        " of the frequency components.")
        instrLineNine=("Bubbles should appear when loud notes are"+
                       " played in the song.")
        instrLineTen=("The locations of these notes are detected when the"+
                      " game loads. This takes some time.")
        return self.helpInstructionsPartOne()+[instrLineSix,instrLineSeven,
                                          instrLineEight,instrLineNine,
                                          instrLineTen]
    def drawHelpScreen(self):
        #Draw the help screen
        self.helpForeground.set_alpha(self.helpForegroundAlpha)
        self.helpForeground=self.helpForeground.convert()
        self.gameScreen.blit(self.helpBackground,(0,0))
        self.gameScreen.blit(self.helpForeground,(0,0))
        instructions=self.getHelpInstructions()
        instructionImages=[self.instructionFont.render(instr,1,(0,0,0)) 
                           for instr in instructions]
        instructionScaling=1/5
        Y=self.gameScreen.get_height()*instructionScaling
        #Draw the text
        for images in instructionImages:
            X=self.gameScreen.get_width()/2-images.get_width()/2
            image=images.convert_alpha()
            self.gameScreen.blit(image,(X,Y))
            Y+=images.get_height()
    
    def drawAll(self):
        #Switchboard draw function
        if self.gameMode=="fileManager":
            self.drawManager()
        elif self.gameMode=="game":
            self.drawGame()
        elif self.gameMode=="loading":
            self.drawLoadingScreen()
        elif self.gameMode=="splashScreen":
            self.drawSplashScreen()
        elif self.gameMode=='help':
            self.drawHelpScreen()

    def fileManagerEvents(self,event,mouseX,mouseY):
        #File manager event handler
        if event.type==pygame.MOUSEBUTTONUP and not self.noFiles:
            textFileY=self.textFileHeight+self.fileManagerTitle.get_height()
            if mouseY>self.fileManagerTitle.get_height() and mouseY<textFileY:
                self.gameMode="game"
                self.gameInit()
            elif mouseY>textFileY:
                y=mouseY-textFileY
                index=int(y//self.waveRectHeight)
                audioFile=self.waveFiles[index]
                self.gameMode="loading"
                self.loadingInit(audioFile)
    def buttonClicked(self):
        self.audioPlayer.stop()
        self.gameMode="fileManager"
        while not self.audioPlayer.stopped:
            pass
        self.stream.stop_stream()
        self.stream.close()
        self.pyAudioObj.terminate()
    def unclickSliders(self):
        for pad in self.fourierPads:
            for slider in pad.sliders:
                slider.clicked=False
    def fourierPadClick(self,mouseX,mouseY):
        #Check if a fourier pad has been clicked on
        foundPad=False
        gameDisplay1X=self.controlScreenWidth
        for i in range(len(self.fourierPads)-1,-1,-1):
            pad=self.fourierPads[i]
            if (pad.checkIfClickedOn(mouseX,mouseY,gameDisplay1X,0)
                and not foundPad):
                pad.clicked=True
                foundPad=True
            else:
                pad.clicked=False
    def gameEvents(self,event,mouseX,mouseY):
        #Game event handler
        if event.type==pygame.MOUSEBUTTONDOWN:
            for pads in self.fourierPads:
                if pads.clicked:
                    for slider in pads.sliders:
                        if slider.checkIfClickedOn(mouseX,mouseY):
                            slider.clicked=True
                        else: slider.clicked=False 
        elif event.type==pygame.MOUSEBUTTONUP:
            gameDisplay1X=self.gameScreenWidth/4
            if ((mouseX-gameDisplay1X)>self.buttonX and 
                (mouseX-gameDisplay1X)<(self.buttonX+self.buttonWidth) 
                and (mouseY>0) and (mouseY<self.buttonHeight)):
                self.buttonClicked()
            if (mouseX>self.gameScreenWidth/4):
                self.fourierPadClick(mouseX,mouseY)
            self.unclickSliders()

    def loadingEvents(self,event,mouseX,mouseY):
        #Loading screen events
        if self.processingComplete:
            if event.type==pygame.MOUSEBUTTONUP:
                self.gameMode="game"
                self.gameInit()

    def splashScreenEvents(self,event,mouseX,mouseY):
        #Splash screen events
        if event.type==pygame.MOUSEBUTTONUP:
            self.gameMode="fileManager"

    def eventHandler(self,event,mouseX,mouseY):
        #Event handler switchboard
        if self.gameMode=="fileManager":
            self.fileManagerEvents(event,mouseX,mouseY)
        elif self.gameMode=="game":
            self.gameEvents(event,mouseX,mouseY)
        elif self.gameMode=="loading":
            self.loadingEvents(event,mouseX,mouseY)
        elif self.gameMode=="splashScreen":
            self.splashScreenEvents(event,mouseX,mouseY)

    def mouseGameHandler(self,mouseX,mouseY):
        #Makes score board more transparent when hovered over
        if (mouseX>0 and mouseX<(self.controlScreenWidth) 
                and mouseY>0 and mouseY<(self.titleScreenHeight)):
            self.playerControlTitleAlpha=self.semiTransparent
        else:
            self.playerControlTitleAlpha=self.opaque
    def fileManagerMouse(self,mouseX,mouseY):
        pass
    def mouseHandler(self,mouseX,mouseY):
        #Mouse handler switchboard
        if self.gameMode=="fileManager":
            self.fileManagerMouse(mouseX,mouseY)
        elif self.gameMode=="game":
            self.mouseGameHandler(mouseX,mouseY)

    def fileTimerFired(self):
        #Updates files listed in file manager
        waveFiles=[]
        filenames=os.listdir()
        for files in filenames:
            if not os.path.isdir(files):
                fileExtension=files.split('.')[-1]
                if fileExtension=='wav':
                    waveFiles.append(files)
        self.waveFiles=waveFiles
        if len(self.waveFiles)>0:
            self.noFiles=False
            self.waveRectHeight=((self.fileManager.get_height()-self.textFileHeight)
                              /len(self.waveFiles))
        else:
            self.noFiles=True
    def gameTimerFired(self):
        self.clock.tick(60)
        if not self.gameOver:
            if self.gameDisplayOverlayAlpha>0:
                self.gameDisplayOverlayAlpha+=self.gameOverlayAlphaV
            audioChunk=1024
            sampleSpace=1/self.samplingRate
            nextAudio=self.audioObject.readWavFile(dataCHUNK=audioChunk)
            if len(nextAudio)>0:
                self.audioQueue.put(nextAudio)
            time=(pygame.time.get_ticks()-self.startTime)/1000
            check=almostIn(time,self.beatTimes,0.05)
            if len(self.fourierPads)<=0 and time>self.musicDuration:
                self.gameOver=True
                self.scoreMessage="Score: %d" % self.score
                self.scoreMessage=self.instructionFont.render(self.scoreMessage,
                                                              1,(255,255,255))
                self.scoreMessage=self.scoreMessage.convert_alpha()
                self.audioPlayer.stop()
            if check[0]:
                uniquenessCheck=almostIn(check[1],self.notesFound,0.05)
                if not uniquenessCheck[0]:
                    self.notesFound.append(check[1])
                    if self.fftQueue.qsize()>0: 
                        FFT=np.fft.rfft(self.fftQueue.get())
                        fftFreqs=np.fft.fftfreq(audioChunk,sampleSpace)
                        mags=np.absolute(FFT)
                        phases=np.angle(FFT)
                        (meanMag,magSTD)=stats(mags)
                        estimatedPeakIndices=coarseScan(mags,10)
                        peakIndices=fineScan(mags,estimatedPeakIndices,
                                             10,meanMag,magSTD,1)
                        mags=[mags[i] for i in peakIndices]
                        phases=[phases[i] for i in peakIndices]
                        fftFreqs=[fftFreqs[i] for i in peakIndices]
                        if len(mags)>1:
                            self.gameDisplayOverlayAlpha+=self.gameOverlayAlphaBeat
                            if self.gameDisplayOverlayAlpha>255:
                                self.gameDisplayOverlayAlpha=255
                            maxMagIndices=findMaxesOfList(mags,2)
                            mags=[mags[i] for i in maxMagIndices]
                            phases=[phases[i] for i in maxMagIndices]
                            fftFreqs=[fftFreqs[i] for i in maxMagIndices]
                            surfaceX=0
                            surfaceY=self.gameScreenHeight/6
                            y=self.playerControlScreen.get_height()/10
                            x=self.playerControlScreen.get_width()/3
                            sliderList=[Slider(self.playerControlScreen,surfaceX,
                                               surfaceY,x*i,y) for i in range(1,3)]
                            self.fourierPads.append(fourierPad(self.fourierPadXStart,
                                                               self.fourierPadYStart,
                                                               self.fourierPadRadius,
                                                               fftFreqs,mags,phases,
                                                               sliderList))
                            self.fourierPadSide*=-1
                            self.oldPadYStart=self.fourierPadYStart
                            fourierPadYI=random.randint(72+self.fourierFloatAmp,
                                                        self.oldPadYStart)
                            fourierPadYJ=random.randint(self.oldPadYStart,
                                                        self.gameScreenHeight
                                                        -72-self.fourierFloatAmp)
                            upperDiff=self.oldPadYStart-fourierPadYI
                            lowerDiff=fourierPadYJ-self.oldPadYStart
                            if upperDiff>lowerDiff:
                                self.fourierPadYStart=fourierPadYI
                            else:
                                self.fourierPadYStart=fourierPadYJ
            for pads in self.fourierPads:
                dy=self.fourierFloatAmp*math.sin(pads.x*self.fourierOmega)
                pads.shiftPad(-0.5,-dy)
                score=pads.scorePlayer()
                if score==100 and not pads.dying:
                    self.score+=score
                    pads.dying=True
            self.fourierPads=[pad for pad in self.fourierPads if pad.active]
            for pads in self.fourierPads:
                if pads.clicked:
                    for slider in pads.sliders:
                        if slider.clicked:
                            (mouseX,mouseY)=pygame.mouse.get_pos()
                            slider.changeSliderPosition(mouseY)
        else:
            self.gameOverlayAlpha+=1
            if self.gameOverlayAlpha>=255:
                self.gameMode="splashScreen"            
            
    def loadingTimerFired(self):
        #Change loading screen overlay transparency
        maxAlpha=255
        self.loadAlpha+=self.dAlpha
        if self.loadAlpha<0:
            self.loadAlpha=0
            self.dAlpha*=-1
        elif self.loadAlpha>maxAlpha:
            self.loadAlpha=maxAlpha
            self.dAlpha*=-1
        if not self.processorDaemon.isAlive():
            loadingText="Click to begin."
            textColor=(255,255,255)
            self.loadTextSurf=self.fileManagerTitleFont.render(loadingText,1,
                                                                  textColor)
            self.processingComplete=True
    def splashScreenTimerFired(self):
        #Change transparency of splash screen overlay
        maxAlpha=255
        self.splashScreenAlpha+=self.splashScreenDAlpha
        if self.splashScreenAlpha<0:
            self.splashScreenAlpha=0
            self.splashScreenDAlpha*=-1
        elif self.splashScreenAlpha>maxAlpha:
            self.splashScreenAlpha=maxAlpha
            self.splashScreenDAlpha*=-1
    def helpTimerFired(self):
        #Change transparency of help screen overlay
        maxAlpha=200
        minAlpha=100
        self.helpForegroundAlpha+=self.helpForegroundDAlpha
        if self.helpForegroundAlpha<minAlpha:
            self.helpForegroundAlpha=minAlpha
            self.helpForegroundDAlpha*=-1
        elif self.helpForegroundAlpha>maxAlpha:
            self.helpForegroundAlpha=maxAlpha
            self.helpForegroundDAlpha*=-1
    def timerFired(self):
        #Timerfired switchboard
        if self.gameMode=="fileManager":
            self.fileTimerFired()
        elif self.gameMode=="game":
            self.gameTimerFired()
        elif self.gameMode=="loading":
            self.loadingTimerFired()
        elif self.gameMode=="splashScreen":
            self.splashScreenTimerFired()
        elif self.gameMode=='help':
            self.helpTimerFired()

    def run(self):
        #Run function
        while self.gameRunning:
            (mouseX,mouseY)=pygame.mouse.get_pos()
            self.mouseHandler(mouseX,mouseY)
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    self.gameRunning=False
                elif event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE:
                    self.gameRunning=False
                elif event.type==pygame.KEYDOWN and event.key==pygame.K_h:
                    #Pressing h enters help mode for all modes
                    if self.gameMode=='help':
                        self.gameMode=self.oldMode
                        #Resume audio playing
                        if self.gameMode=='game': self.audioPlayer.resume()
                    else:
                        self.oldMode=self.gameMode
                        #Pause audio
                        if self.oldMode=='game': self.audioPlayer.pause()
                        self.gameMode='help'
                else:
                    self.eventHandler(event,mouseX,mouseY)
            self.timerFired()
            self.drawAll()
            pygame.display.update()
        #Quit pygame
        pygame.quit()
if __name__=='__main__':
    game=fourierGame()
    game.run()
