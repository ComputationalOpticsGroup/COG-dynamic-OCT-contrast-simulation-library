import doctSimulator as dos
import numpy as np
import matplotlib.pyplot as plt

simuFieldSize = np.array([195., 195., 724.]) * 1e-6
totalFieldSize = np.array([345., 345., 874.]) * 1e-6
pixNum = np.array([100, 100, 100])
lmd_c = 1.31e-6
scaDensity = 0.055 * 1e18
index = 800
vAmp = 0.225e-6 * index / 50
res = np.array([18, 18, 14])
dt = 0.2048
nFrames = 32


psfParam = dos.complexPsfParameters(res)
n3D = dos.numerical3dFieldParameters(totalFieldSize, simuFieldSize, pixNum, lmd_c)
scaPos = dos.scattererPositions(scaDensity, n3D)
scaPos.velocitiesSet('randomBallistic', vAmp)
scaField = dos.scattererField(n3D)
scaField.generate(scaPos)
psfField = dos.complexPsfField(n3D, psfParam)
psfField.psfSpectrumGet()

octField = dos.complexOctField(n3D)
octField.generate(scaField, psfField)
for i in range(nFrames):
    scaPos.positionsUpdate(dt)    ## scat moving
    scaField.generate(scaPos)
    octField.generate(scaField, psfField)

# Show output
plt.imshow(np.abs(octField.complexOctField[0, :, :]))
plt.show()
