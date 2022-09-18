import numpy as np
import cv2
import matplotlib.pyplot as plt


def movingAvg(curve, radius):
    windowSize = 2 * radius + 1
    f = np.ones(windowSize) / windowSize
    curvePad = np.lib.pad(curve, (radius, radius), 'edge')
    curveSmoothed = np.convolve(curvePad, f, mode='same')
    curveSmoothed = curveSmoothed[radius:-radius]
    return curveSmoothed


def smooth(trajectory):
    smoothedTrajectory = np.copy(trajectory)
    for i in range(3):
        smoothedTrajectory[:, i] = movingAvg(trajectory[:, i], radius=5)

    return smoothedTrajectory


def fixBorder(frame):
    s = frame.shape
    # Scale the image by 4% without movinf the center
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


cap = cv2.VideoCapture('IMG_0182.MOV')

nFrames = 50  # int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter('result.mp4', fourcc, 24, (w, h))

_, prev = cap.read()
prevGray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

transforms = np.zeros((nFrames - 1, 3), np.float32)

for i in range(nFrames - 2):
    prevPts = cv2.goodFeaturesToTrack(prevGray,
                                      maxCorners=200,
                                      qualityLevel=0.01,
                                      minDistance=30,
                                      blockSize=3)
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pts, status, err = cv2.calcOpticalFlowPyrLK(prevGray, gray, prevPts, None)

    idx = np.where(status == 1)[0]
    prevPts = prevPts[idx]
    pts = pts[idx]

    # Transformation matrix
    m = cv2.estimateRigidTransform(prevPts, pts, fullAffine=False)
    # Translation
    dx = m[0, 2]
    dy = m[1, 2]
    # Rotation angle
    da = np.arctan2(m[1, 0], m[0, 0])
    # Store transformation
    transforms[i] = [dx, dy, da]

    prevGray = gray
    print('Frame: {}/{} Tracked points: {}'.format(str(i), str(nFrames), str(len(prevPts))))

trajectory = np.cumsum(transforms, axis=0)
smoothedTrajectory = smooth(trajectory)
plt.plot(trajectory[:, 2])
plt.plot(smoothedTrajectory[:, 2])
plt.show()

diff = smoothedTrajectory - trajectory
transformsSmooth = transforms + diff

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

for i in range(nFrames - 2):
    _, frame = cap.read()

    dx = transformsSmooth[i, 0]
    dy = transformsSmooth[i, 1]
    da = transformsSmooth[i, 2]

    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy

    stabilized = cv2.warpAffine(frame, m, (w, h))
    stabilized = fixBorder(stabilized)
    out = cv2.hconcat([frame, stabilized])
    out = cv2.resize(out, (out.shape[1]//2, out.shape[0]//2))

    cv2.imshow('Before and After', out)
    writer.write(out)
    cv2.waitKey(10)
