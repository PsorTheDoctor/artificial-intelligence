import numpy as np
import cv2


class BackProjectionColorDetector:
    def __init__(self):
        self.templateHsv = None

    def setTemplate(self, template):
        self.templateHsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

    def returnMask(self, frame, morphOpening=True, blur=True, kernelSize=5, iters=1):
        if self.templateHsv is None:
            return None

        frameHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        templateHist = cv2.calcHist(self.templateHsv, (0, 1), None, (180, 256), (0, 180, 0, 256))
        cv2.normalize(templateHist, templateHist, 0, 255, cv2.NORM_MINMAX)
        frameHsv = cv2.calcBackProject(frameHsv, (0, 1), templateHist, (0, 180, 0, 256), 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelSize, kernelSize))
        frameHsv = cv2.filter2D(frameHsv, -1, kernel)
        if morphOpening:
            kernel = np.ones((kernelSize, kernelSize), np.uint8)
            frameHsv = cv2.morphologyEx(frameHsv, cv2.MORPH_OPEN, kernel, iterations=iters)
        if blur:
            frameHsv = cv2.GaussianBlur(frameHsv, (kernelSize, kernelSize), 0)

        _, frameThresh = cv2.threshold(frameHsv, 50, 255, 0)
        return cv2.merge((frameThresh, frameThresh, frameThresh))


class BinaryMaskAnalyser:
    def returnNumberOfContours(self, mask):
        """
        Returns number of contours present on the mask.
        """
        if mask is None:
            return None

        mask = np.copy(mask)
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        contours, hierarchy = cv2.findContours(mask, 1, 2)
        if hierarchy is None:
            return 0
        else:
            return len(hierarchy)

    def returnMaxAreaRect(self, mask):
        """
        Returns the rectangle surrounding the contour with the largest area.
        """
        if mask is None:
            return (None, None)

        mask = np.copy(mask)
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        contours, hierarchy = cv2.findContours(mask, 1, 2)
        areaArray = np.zeros(len(contours))
        i = 0
        for cnt in contours:
            areaArray[i] = cv2.contourArea(cnt)
            i += 1

        if areaArray.size == 0:
            return (None, None)

        maxAreaIndex = np.argmax(areaArray)
        cnt = contours[maxAreaIndex]
        (x, y, w, h) = cv2.boundingRect(cnt)
        return (x, y, w, h)

    def returnMaxAreaCenter(self, mask):
        """
        Returns the center of the contour with largest area.
        """
        if mask is None:
            return (None, None)

        mask = np.copy(mask)
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        contours, hierarchy = cv2.findContours(mask, 1, 2)
        areaArray = np.zeros(len(contours))
        i = 0
        for cnt in contours:
            areaArray[i] = cv2.contourArea(cnt)
            i += 1

        if areaArray.size == 0:
            return (None, None)

        maxAreaIndex = np.argmax(areaArray)
        cnt = contours[maxAreaIndex]
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            return (None, None)

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)


class ParticleFilter:
    def __init__(self, width, height, nParticles):
        if nParticles <= 0 or nParticles > (width * height):
            raise ValueError('')

        self.particles = np.empty((nParticles, 2))
        self.particles[:, 0] = np.random.uniform(0, width, size=nParticles)
        self.particles[:, 1] = np.random.uniform(0, height, size=nParticles)
        self.weights = np.array([1.0 / nParticles] * nParticles)

    def predict(self, xVelocity, yVelocity, std):
        """
        Predicts the position of the point in ther next frame.
        """
        self.particles[:, 0] += xVelocity + (np.random.randn(len(self.particles)) * std)
        self.particles[:, 1] += yVelocity + (np.random.randn(len(self.particles)) * std)

    def update(self, x, y):
        """
        Updates the weights associated with each particle basing on the x, y coords.
        """
        pos = np.empty((len(self.particles), 2))
        pos[:, 0].fill(x)
        pos[:, 1].fill(y)
        dist = np.linalg.norm(self.particles - pos, axis=1)
        maxDist = np.amax(dist)
        dist = np.add(-dist, maxDist)
        self.weights.fill(1.0)
        self.weights *= dist
        self.weights += 1.e-300
        self.weights /= sum(self.weights)

    def estimate(self):
        """
        Estimates the position of the point given the particles weights.
        """
        xMean = np.average(self.particles[:, 0], weights=self.weights, axis=0).astype(int)
        yMean = np.average(self.particles[:, 1], weights=self.weights, axis=0).astype(int)
        return xMean, yMean, 0, 0

    def resample(self, method='residual'):
        """
        Resamples the particles basing on their weights.
        """
        nParticles = len(self.particles)
        if method == 'multinomal':
            cumulativeSum = np.cumsum(self.weights)
            cumulativeSum[-1] = 1.0
            indices = np.searchsorted(cumulativeSum, np.random.uniform(low=0.0, high=1.0, size=nParticles))

        elif method == 'residual':
            indices = np.zeros(nParticles, dtype=np.int32)
            nCopies = (nParticles * np.asarray(self.weights)).astype(int)
            k = 0
            for i in range(nParticles):
                for _ in range(nCopies[i]):
                    indices[k] = i
                    k += 1
            # Multinomal resample
            residual = self.weights - nCopies
            residual /= sum(residual)
            cumulativeSum = np.cumsum(residual)
            cumulativeSum[-1] = 1.0
            indices[k:nParticles] = np.searchsorted(cumulativeSum, np.random.random(nParticles - k))

        elif method == 'stratified':
            positions = (np.random.random(nParticles) + range(nParticles)) / nParticles
            indices = np.zeros(nParticles, dtype=np.int32)
            cumulativeSum = np.cumsum(self.weights)
            i = 0
            j = 0
            while i < nParticles:
                if positions[i] < cumulativeSum[j]:
                    indices[i] = j
                    i += 1
                else:
                    j += 1

        elif method == 'systematic':
            positions = (np.arange(nParticles) + np.random.random()) / nParticles
            indices = np.zeros(nParticles, dtype=np.int32)
            cumulativeSum = np.cumsum(self.weights)
            i = 0
            j = 0
        else:
            raise ValueError('Method {} is not implemented'.format(method))

        self.particles[:] = self.particles[indices]
        self.weights /= np.sum(self.weights)

    def returnParticlesContribution(self):
        """
        Gives the number of particles which are contributing to the probability distribution.
        """
        return 1.0 / np.sum(np.square(self.weights))

    def returnParticlesCoords(self, index=1):
        if index < 0:
            return self.particles.astype(int)
        else:
            return self.particles[index, :].astype(int)

    def drawParticles(self, frame, color=(0, 0, 255), radius=2):
        """
        Draws the particles on a frame and returns it.
        """
        for xParticle, yParticle in self.particles.astype(int):
            cv2.circle(frame, (xParticle, yParticle), radius, color, -1)


windowId = 'Particle Filter'
cap = cv2.VideoCapture(0)
std = 25
_, frame = cap.read()
h, w = frame.shape[:2]
filter = ParticleFilter(w, h, nParticles=3000)
noiseProbability = 0.15
roi = cv2.selectROI(windowId, frame, fromCenter=False, showCrosshair=False)
(roiX, roiY, roiW, roiH) = roi
xCenter = roiX + roiW // 2
yCenter = roiY + roiY // 2
template = frame[roiY : roiY+roiH, roiX : roiX+roiW]
analyser = BinaryMaskAnalyser()
detector = BackProjectionColorDetector()
detector.setTemplate(template)

while cap.isOpened():
    _, frame = cap.read()
    mask = detector.returnMask(frame, morphOpening=True, blur=True, kernelSize=5, iters=2)
    if analyser.returnNumberOfContours(mask) > 0:
        x, y, w, h = analyser.returnMaxAreaRect(mask)
        xCenter, yCenter = analyser.returnMaxAreaCenter(mask)
        # Adding noise to the coords
        if np.random.uniform() >= 1.0 - noiseProbability:
            xNoise = int(np.random.uniform(-300, 300))
            yNoise = int(np.random.uniform(-300, 300))
        else:
            xNoise = 0
            yNoise = 0

        x += xNoise
        y += yNoise
        xCenter += xNoise
        yCenter += yNoise
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    filter.predict(xVelocity=0, yVelocity=0, std=std)
    filter.drawParticles(frame)
    xEstimated, yEstimated, _, _ = filter.estimate()
    cv2.circle(frame, (xEstimated, yEstimated), 3, (0, 255, 0), -1)
    filter.update(xCenter, yCenter)
    filter.resample(method='residual')

    cv2.imshow(windowId, frame)
    if cv2.waitKey(50) == 27:
        break

cap.release()
cv2.destroyAllWindows()
