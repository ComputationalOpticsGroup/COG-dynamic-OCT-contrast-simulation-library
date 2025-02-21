import numpy as np
import cupy as cp
import cupyx as cpx
from concurrent.futures import ThreadPoolExecutor


class numerical3dFieldParameters:
    def __init__(self, phyField, anaField, pixNum, waveLength):
        self.phyField = phyField
        self.anaField = anaField
        self.pixNum = pixNum
        self.pixSeparation = np.array(anaField / pixNum, dtype='float64')
        self.waveLength = waveLength



class numerical0dParameters:
    def __init__(self, totalTime, res, waveLength):
        self.totalTime = totalTime
        self.res = np.array(res)
        self.waveLength = waveLength


class complexPsfParameters:
    def __init__(self, res, resType='default'):
        self.res = np.array(res)
        self.resType = resType
        self.psfType = 'Gaussian'


class scattererPositions:
    def __init__(self, scatterDensity, numFieldParam, motion, motionParam, r=1.0):
        if not isinstance(numFieldParam, (numerical3dFieldParameters, numerical0dParameters)):
            raise TypeError("numFieldParam must be an instance of numerical3dFieldParameters or numerical0dParamters")
        if isinstance(numFieldParam, numerical3dFieldParameters):
            self.phyField = numFieldParam.phyField

            self.numScatter = int(scatterDensity * np.prod(self.phyField))

            self.scatterArray = self.phyField[:, None] * np.random.rand(3, self.numScatter)
            self.reflectivity = r

            self.motion = motion
            self.motionParam = motionParam
            self.velocityAmp = None
            self.velocityPhi = None
            self.velocityTheta = None
            self.dCoeff = None


        if isinstance(numFieldParam, numerical0dParameters):
            res_x, res_y, res_z = numFieldParam.res
            std_E = [1/(2*np.sqrt(2))*res_x, 1/(2*np.sqrt(2))*res_y, 1/(2*np.sqrt(np.log(2)))*res_z]
            std_I = 1 / np.sqrt(2) * np.array(std_E)
            if self.motion == 'randomBallistic':
                maxDis = motionParam * numFieldParam.totalTime
            if self.motion == 'flow':
                maxDis = motionParam[0] * numFieldParam.totalTime
            if self.motion == 'diffusion':
                maxDis = np.sqrt(motionParam[0] * numFieldParam.totalTime *2./3.)
            self.phyField = np.array(std_I * 5. + [2.* maxDis, 2.* maxDis, 2.* maxDis])
            self.numScatter = int(scatterDensity * np.prod(self.phyField))
            self.scatterArray = self.phyField[:, None] * np.random.rand(3, self.numScatter)
            self.reflectivity = r
            self.motionParam = motionParam
            self.velocityAmp = None
            self.velocityPhi = None
            self.velocityTheta = None
            self.dCoeff = None





    def velocitiesSet(self):
        motionParam = self.motionParam
        if self.motion == 'randomBallistic':
            self.velocityAmp = motionParam * np.ones(self.numScatter)
            self.velocityPhi = np.random.uniform(0, np.pi, self.numScatter)
            self.velocityTheta = np.random.uniform(0, 2 * np.pi, self.numScatter)

        if self.motion == 'flow':
            self.velocityAmp = motionParam[0] * np.ones(self.numScatter)
            self.velocityPhi = motionParam[1] * np.ones(self.numScatter)
            self.velocityTheta = motionParam[2] * np.ones(self.numScatter)

        if self.motion == 'flow/shiftComputation':
            self.velocityAmp = motionParam[0] * np.ones(self.numScatter)
            self.velocityPhi = motionParam[1] * np.ones(self.numScatter)
            self.velocityTheta = motionParam[2] * np.ones(self.numScatter)

        if self.motion == 'diffusion':
            self.dCoeff = motionParam

    def positionsUpdate(self, dt):
        randomDelta = np.zeros((3, self.numScatter), dtype='float64')
        if self.motion == 'randomBallistic' or self.motion == 'flow' or self.motion == 'flow/shiftComputation' :
            randomDelta[0] = self.velocityAmp * np.sin(self.velocityPhi) * np.cos(self.velocityTheta) * dt
            randomDelta[1] = self.velocityAmp * np.sin(self.velocityPhi) * np.sin(self.velocityTheta) * dt
            randomDelta[2] = self.velocityAmp * np.cos(self.velocityPhi) * dt

        if self.motion == 'diffusion':
            randomDelta = np.random.normal(0, np.sqrt(2 * self.dCoeff * dt), (3, self.numScatter))

        self.scatterArray += randomDelta


class scattererField:
    def __init__(self, numFieldParam):
        if not isinstance(numFieldParam, numerical3dFieldParameters):
            raise TypeError("numFieldParam must be an instance of numerical3dFieldParameters")
        self.numFieldParam = numFieldParam
        self.scattererField = None

    def generate(self, scatPosition, numChunk=128):
        if not isinstance(scatPosition, scattererPositions):
            raise TypeError("scatPosition must be an instance of scattererPositions")

        scatPos = scatPosition.scatterArray
        lmd_c = self.numFieldParam.waveLength
        phase = 4 * np.pi * scatPos[2] / lmd_c
        scaReflectivityList = scatPosition.reflectivity * np.exp(1j * phase, dtype= 'complex64')

        scaReflectivityList = scaReflectivityList[np.newaxis, :]

        allList = np.concatenate((scatPos, scaReflectivityList), axis=0)

        def threaded(allList):

            # initiating
            scattererField = np.zeros(self.numFieldParam.pixNum, dtype='complex64')
            initialShift = (self.numFieldParam.phyField[0] - self.numFieldParam.anaField[0]) / 2

            scaReflectivityList = allList[3]
            pixelSeparation = np.array(self.numFieldParam.pixSeparation, dtype='float64')

            # reshape to 1D array for index citation
            scattererFieldLine = scattererField.reshape(np.prod(self.numFieldParam.pixNum))

            # Finding particles in simulation field in physical size
            scatterPos_in_ana = allList[0:3] - initialShift
            particleIndexField = ((scatterPos_in_ana[0] >= 0) & (scatterPos_in_ana[0] < self.numFieldParam.anaField[0])
                                  & (scatterPos_in_ana[1] >= 0) & (
                                          scatterPos_in_ana[1] < self.numFieldParam.anaField[1]) & (
                                          scatterPos_in_ana[2] >= 0) & (
                                          scatterPos_in_ana[2] < self.numFieldParam.anaField[2]))

            scatterArray_InField = np.array(scatterPos_in_ana[:, particleIndexField], dtype='float64')

            riInField = scaReflectivityList[particleIndexField]

            # Finding particles in simulation field in voxel index size
            voxelIndex = np.zeros(scatterArray_InField.shape, dtype='float64')
            voxelIndex[0] = (scatterArray_InField[0]) // pixelSeparation[0]
            voxelIndex[1] = (scatterArray_InField[1]) // pixelSeparation[1]
            voxelIndex[2] = (scatterArray_InField[2]) // pixelSeparation[2]
            voxelIndex = np.array(voxelIndex,dtype='int')
            voxelIndexT = voxelIndex.T

            # changing voxel index from 3D to 1D
            lineFactor = np.array(
                [self.numFieldParam.pixNum[1] * self.numFieldParam.pixNum[2], self.numFieldParam.pixNum[2], 1])
            voxelIndexLine = np.dot(voxelIndexT, lineFactor)

            # 1D matrix assignment
            scattererFieldLine[voxelIndexLine] = riInField

            # finding multiple particles in the same voxel (1D) and do summation calculation

            uniqueVoxelIndexLine, uIndex, inverseIndex, count = np.unique(voxelIndexLine, return_index=True,
                                                                          return_inverse=True, return_counts=True)

            indexArray = np.array(range(uniqueVoxelIndexLine.shape[0]))
            for doubleIndex in indexArray[count >= 2]:
                multiIndex = uniqueVoxelIndexLine[doubleIndex]
                scattererFieldLine[multiIndex] = np.sum(riInField[inverseIndex == doubleIndex])
            scattererField = scattererFieldLine.reshape(self.numFieldParam.pixNum)
            return scattererField

        scattererField = np.zeros(self.numFieldParam.pixNum, dtype='complex64')
        with ThreadPoolExecutor(max_workers=12) as executor:

            for result in executor.map(threaded, np.array_split(allList, numChunk, axis=1)):
                scattererField += result
        self.scattererField = scattererField
    def flowShift(self, scatPosition, dt):
        if not isinstance(scatPosition, scattererPositions):
            raise TypeError("scatPosition must be an instance of scattererPositions")
        if isinstance(scatPosition, scattererPositions):
            if not scatPosition.motion == 'flow/shiftComputation':
                raise TypeError("Motion type must be flow and users want to fast compute the speckle")
        randomDelta = np.zeros((3, scatPosition.numScatter), dtype='float64')
        randomDelta[0] = scatPosition.velocityAmp * np.sin(scatPosition.velocityPhi) * np.cos(scatPosition.velocityTheta) * dt
        randomDelta[1] = scatPosition.velocityAmp * np.sin(scatPosition.velocityPhi) * np.sin(scatPosition.velocityTheta) * dt
        randomDelta[2] = scatPosition.velocityAmp * np.cos(scatPosition.velocityPhi) * dt

        shift_pixel = randomDelta / self.numFieldParam.pixSeparation

        space_x = np.fft.fftfreq(self.numFieldParam.pixNum[0])
        space_y = np.fft.fftfreq(self.numFieldParam.pixNum[1])
        space_z = np.fft.fftfreq(self.numFieldParam.pixNum[2])
        fx3, fy3, fz3 = np.meshgrid(space_x, space_y, space_z, indexing='ij')

        shift_matrix = np.exp(-2j * np.pi * (fx3 * shift_pixel[0] + fy3 * shift_pixel[1] + fz3 * shift_pixel[2]))

        def dataShift(data, shift_matrix):
            data_cupy = cp.asarray(data)
            shift_matrix_cupy = cp.asarray(shift_matrix)
            data_FFT = cpx.scipy.fft.fftn(data_cupy)

            shifted_FFT = data_FFT * shift_matrix_cupy

            data_shifted = cp.asnumpy(cpx.scipy.fft.ifftn(shifted_FFT))
            return data_shifted

        self.scattererField = dataShift(self.scattererField, shift_matrix)
        scatPosition.positionsUpdate(dt)


class complexPsfField:
    def __init__(self, numFieldParam, psfParam, peak_intensity=1.0):
        self.PSFSpectrum = None
        self.numFieldParam = numFieldParam
        self.psfPram = psfParam
        w = None
        if not isinstance(numFieldParam, numerical3dFieldParameters):
            raise TypeError("numFieldParam must be an instance of numerical3dFieldParameters")
        if not isinstance(psfParam, complexPsfParameters):
            raise TypeError("psfParam must be an instance of complexPsfParameters")

        if psfParam.resType == 'default':
            w = np.array(
                [psfParam.res[0] / 2, psfParam.res[1] / 2, psfParam.res[2] / np.sqrt(2.0 * np.log(2.0))]) 
        interMediate = np.array(w/numFieldParam.pixSeparation, dtype='float64')
        print(interMediate)

        xField = np.multiply.outer(np.ones((numFieldParam.pixNum[0], numFieldParam.pixNum[0])),
                                   np.arange(-numFieldParam.pixNum[0] / 2 + 1, numFieldParam.pixNum[0] / 2 + 1, 1))

        yField = np.rot90(np.transpose(xField))
        zField = np.transpose(xField, (2, 1, 0))
        self.psfField = np.exp(-(xField / interMediate[0]) ** 2) * np.exp(-(yField / interMediate[1]) ** 2) * np.exp(-(zField / interMediate[2]) ** 2)

        self.flag_psfSpectrum = False

    def psfSpectrumGet(self):
        if not self.flag_psfSpectrum:
            psfField_cp = cp.asarray(self.psfField)
            self.PSFSpectrum = cp.fft.fftn(psfField_cp)
            self.flag_psfSpectrum = True
            return self.PSFSpectrum
        else:
            return self.PSFSpectrum


class complexOctField:
    def __init__(self, numFieldParam):
        self.complexOctField = None
        self.numFieldParam = numFieldParam

    def generate(self, scattererField, psfField):
        scatField = cp.asarray(scattererField.scattererField)

        scatField_spectrum = cp.fft.fftn(scatField)
        PSFSpectrum = psfField.PSFSpectrum
        convolved_spectrum = scatField_spectrum * PSFSpectrum
        convolved_field = cp.fft.ifftn(convolved_spectrum)
        convolved_field = cp.asnumpy(convolved_field)
        self.complexOctField = convolved_field

class complexOctPixel:
    def __init__(self, num0dParam):
        self.num0dParam = num0dParam
        self.complexOctPixel = None

    def generate(self,scattererPositions):
        res_x, res_y, res_z = self.num0dParam.res
        std_E = [1 / (2 * np.sqrt(2)) * res_x, 1 / (2 * np.sqrt(2)) * res_y, 1 / (2 * np.sqrt(np.log(2))) * res_z]
        x, y, z = scattererPositions.scatterArray
        refArray = scattererPositions.reflectivity * np.ones(scattererPositions.numScatter)
        self.complexOctPixel = np.sum(refArray * np.exp(1j * (4. * np.pi / self.num0dParam.waveLength * z ))*np.exp(-1/2*((x/std_E[0])**2 + (y/std_E[1])**2 + (z/std_E[2])**2 )))

class complexNoiseField:
    def __init__(self, numFieldParam, psfField):
        self.numFieldParam = numFieldParam
        self.psfField = psfField
        self.field = None

    def generate(self, noiseEnergies, del_lmd = 50):
        var_r_det = noiseEnergies[0] / 2
        std_Det = np.sqrt(var_r_det)
        Det_real = np.random.normal(0, std_Det, self.numFieldParam.pixNum)
        Det_imag = np.random.normal(0, std_Det, self.numFieldParam.pixNum)
        Det = Det_real + 1j * Det_imag
        Det_rescaled = noiseEnergies[0]* Det / np.mean(np.abs(Det)**2)

        var_r_rin = noiseEnergies[1] / 2
        std_rin = np.sqrt(var_r_rin)
        RIN_real = np.random.normal(0, std_rin, self.numFieldParam.pixNum)
        RIN_imag = np.random.normal(0, std_rin, self.numFieldParam.pixNum)
        Nrin = RIN_real + 1j * RIN_imag

        lmd = self.numFieldParam.waveLength
        k0 = (2 * np.pi) / lmd
        del_k = np.pi / (np.sqrt(np.log(2))) * (del_lmd / lmd ** 2)
        k = np.linspace(k0 - 2 * del_k, k0 + 2 * del_k, self.numFieldParam.pixNum[0])
        S_k = 1 / (del_k * np.sqrt(cp.pi)) * (cp.exp(-((k - k0) / del_k) ** 2))

        RIN_freq = Nrin * S_k[:, None, None]
        RIN = cp.fft.ifftn(RIN_freq)
        RIN_rescaled = noiseEnergies[1] * RIN / np.mean(np.abs(RIN) ** 2)

        var_r_sh = noiseEnergies[2] / 2
        std_shot = np.sqrt(var_r_sh)
        Shot_real = np.random.normal(0, std_shot, self.numFieldParam.pixNum)
        Shot_imag = np.random.normal(0, std_shot, self.numFieldParam.pixNum)
        Nshot = Shot_real + 1j * Shot_imag

        S_k_shot = np.sqrt(S_k)
        Shot_freq = Nshot * S_k_shot
        Shot = cp.fft.ifftn(Shot_freq)
        Shot_rescaled = noiseEnergies[2] * Shot / np.mean(np.abs(Shot) ** 2)

        self.field = Det_rescaled + RIN_rescaled + Shot_rescaled
        