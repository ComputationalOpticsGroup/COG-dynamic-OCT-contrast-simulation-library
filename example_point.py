import doctSimulator as dos
import numpy as np

waveLength = 0.84e-6
scaDensity = 0.055 * 1e18
res = np.array([4.8, 4.8, 3.8])* 1e-6
dt = 0.2048
nFrames = 32
totalTime = dt*nFrames
motion = 'randomBallistic'
motionParam = 0.5e-6 # velocity [m/s]

psfParam = dos.complexPsfParameters(res)
n0D = dos.numerical0dParameters(totalTime, res, waveLength)
scaPos = dos.scattererPositions(scaDensity, n0D, motion, motionParam)
scaPos.velocitiesSet()
octSignal = dos.complexOctPixel(n0D)
EE_d = np.ones((nFrames), dtype=np.complex64)

for i in range(nFrames):
    scaPos.positionsUpdate(dt)
    octSignal.generate(scaPos)
    EE = octSignal.complexOctPixel
    EE_d[i] = EE
    I = np.real(EE*np.conjugate(EE))



