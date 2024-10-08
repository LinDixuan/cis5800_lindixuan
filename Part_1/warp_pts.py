import numpy as np
from est_homography import est_homography


def warp_pts(X, Y, interior_pts):
    """
    First compute homography from video_pts to logo_pts using X and Y,
    and then use this homography to warp all points inside the soccer goal

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
        interior_pts: Nx2 matrix of points inside goal
    Returns:
        warped_pts: Nx2 matrix containing new coordinates for interior_pts.
        These coordinate describe where a point inside the goal will be warped
        to inside the penn logo. For this assignment, you can keep these new
        coordinates as float numbers.

    """

    # You should Complete est_homography first!
    H = est_homography(X, Y)

    warped_pts = []
    ##### STUDENT CODE START #####
    for i in range(interior_pts.shape[0]):
        pts_homo = np.array([interior_pts[i, 0], interior_pts[i, 1], 1]).astype(np.float64).reshape(3, 1)
        pts_trans = (H @ pts_homo).reshape(3)
        warped_pts.append(pts_trans[:2] / pts_trans[2])

    warped_pts = np.stack(warped_pts, axis=0)

    ##### STUDENT CODE END #####
    
    # Delete this line after you implemented the function
    #raise NotImplementedError("warps_pts() is not implemented!")
    return warped_pts
