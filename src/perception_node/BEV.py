import cv2
import numpy as np
import glob

##chess board with 10*7
chessboard_size = (9, 6)

# Prepare world coordinates (Z = 0 plane)
objp = np.zeros((9*6, 2), np.float32)
objp[:, :] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) #flat 54 rows and 3 cols


## Getting homographies from images
images = ["1.png", "2.png", "3.png"]

homographies = []

for image in images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        img_pts = corners.reshape(-1, 2)

        H, _ = cv2.findHomography(objp, img_pts)

        homographies.append(H)



## vij -> h1T B h2= 0 => v12T b =0

def v_ij(hi, hj):
    return np.array([
        hi[0]*hj[0],
        hi[0]*hj[1] + hi[1]*hj[0],
        hi[1]*hj[1],
        hi[2]*hj[0] + hi[0]*hj[2],
        hi[2]*hj[1] + hi[1]*hj[2],
        hi[2]*hj[2]
    ])

V = []

for H in homographies:
    h1 = H[:, 0]
    h2 = H[:, 1]

    v12 = v_ij(h1, h2)
    v11 = v_ij(h1, h1)
    v22 = v_ij(h2, h2)

    V.append(v12)
    V.append(v11 - v22)

V = np.array(V)

##Vb = 0 solution: by SVD
U, S, Vt = np.linalg.svd(V)
b = Vt[-1]

## Extracting K
B11, B12, B22, B13, B23, B33 = b

v0 = (B12*B13 - B11*B23) / (B11*B22 - B12**2)

lambda_ = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11

alpha = np.sqrt(lambda_ / B11)
beta = np.sqrt(lambda_ * B11 / (B11*B22 - B12**2))
gamma = -B12 * alpha**2 * beta / lambda_
u0 = gamma*v0/beta - B13 * alpha**2 / lambda_

K = np.array([
    [alpha, gamma, u0],
    [0,     beta,  v0],
    [0,     0,     1]
])


# Rotational matrix:
K_inv = np.linalg.inv(K)

for i, H in enumerate(homographies):

    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    lambda_ = 1 / np.linalg.norm(K_inv @ h1)

    r1 = lambda_ * (K_inv @ h1)
    r2 = lambda_ * (K_inv @ h2)
    r3 = np.cross(r1, r2)

    Translation_Mat = lambda_ * (K_inv @ h3)

    Rotation_Mat = np.column_stack((r1, r2, r3))


    U_r, _, Vt_r = np.linalg.svd(Rotation_Mat)
    Rotation_Mat = U_r @ Vt_r
    
    print(f"\nImage {i+1}")
    print("Rotation Matrix R:\n", Rotation_Mat)
    print("Translation Vector t:\n", Translation_Mat)



####################
## BEV:
cam_h = 0.76
Camera_cent = np.array([0, 0, cam_h])

#homograph Matrix
r1 = Rotation_Mat[:, 0].reshape(3,1)
r2 = Rotation_Mat[:, 1].reshape(3,1)

H = K @ np.hstack((r1, r2, Translation_Mat))

# H bev -> inverse 
Hinv = np.linalg.inv(H)

#Wrap Mask
mask = cv2.imread("1_mask_0.png", cv2.IMREAD_GRAYSCALE)

bev = cv2.warpPerspective(
    mask,
    Hinv,
    (640, 480),
    flags=cv2.INTER_NEAREST
)

cv2.imshow("BEV", bev)
cv2.waitKey(0)
cv2.destroyAllWindows()


