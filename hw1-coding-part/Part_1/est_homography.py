import numpy as np

def get_eq(x, x_t, y, y_t):
    ax = np.array([-x, -y, -1, 0, 0, 0, x * x_t, y * x_t, x_t]).astype(np.float64)
    ay = np.array([0, 0, 0, -x, -y, -1, x * y_t, y * y_t, y_t]).astype(np.float64)
    return ax, ay

def est_homography(X, Y):
    """
    Calculates the homography of two planes, from the plane defined by X
    to the plane defined by Y. In this assignment, X are the coordinates of the
    four corners of the soccer goal while Y are the four corners of the penn logo

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X

    """

    ##### STUDENT CODE START #####
    A_list = []
    for i in range(4):
        x, y = X[i, 0], X[i, 1]
        x_t, y_t = Y[i, 0], Y[i, 1]
        ax, ay = get_eq(x, x_t, y, y_t)
        A_list.append(ax)
        A_list.append(ay)

    A_mat = np.stack(A_list, axis=0)

    [U, S, V] = np.linalg.svd(A_mat)

    H = np.array([V[8, 0], V[8, 1], V[8, 2],
                  V[8, 3], V[8, 4], V[8, 5],
                  V[8, 6], V[8, 7], V[8, 8]]).reshape(3, 3)
    ##### STUDENT CODE END #####
    
    # Delete this line after you implemented the function
    #raise NotImplementedError("est_homography() is not implemented!")

    return H


if __name__ == "__main__":
    # You could run this file to test out your est_homography implementation
    #   $ python est_homography.py
    # Here is an example to test your code, 
    # but you need to work out the solution H yourself.
    X = np.array([[0, 0],[0, 10], [5, 0], [5, 10]])
    Y = np.array([[3, 4], [4, 11],[8, 5], [9, 12]])
    H = est_homography(X, Y)

    for i in range(4):
        a = np.array([X[i, 0], X[i, 1], 1]).reshape(3, 1)
        y = (H @ a).reshape(3)
        y = y / y[2]
        print(y)

    