import doctSimulator as dos
import numpy as np
import cupy as cp

simuFieldSize = np.array([195., 195., 724.]) * 1e-6
totalFieldSize = np.array([345., 345., 874.]) * 1e-6
pixNum = np.array([100, 100, 100])
lmd_c = 1.31e-6
scaDensity = 0.055 * 1e18
index = 800
vAmp = 0.225e-6 * index / 50
res = np.array([18, 18, 14])

psfParam = dos.complexPsfParameters(res)
n3D = dos.numerical3dFieldParameters(totalFieldSize, simuFieldSize, pixNum, lmd_c)
scaPos = dos.scattererPositions(scaDensity, n3D)
scaPos.velocitiesSet('randomBallistic', vAmp)
scaField = dos.scattererField(n3D)
scaField.generate(scaPos)

psfField = dos.complexPsfField(n3D, psfParam)
psfField.psfSpectrumGet(psfField.buff_zeroMarginSize)

octField = dos.complexOctField(n3D)
octField.generate(scaField, psfField)

