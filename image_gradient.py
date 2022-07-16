from scipy import ndimage

def mygradient(image):
    """
    This function takes a grayscale image and returns two arrays of the
    same size, one containing the magnitude of the gradient, the second
    containing the orientation of the gradient.
    
    
    Parameters
    ----------
    image : 2D float array of shape HxW
         An array containing pixel brightness values
    
    Returns
    -------
    mag : 2D float array of shape HxW
        gradient magnitudes
        
    ori : 2Dfloat array of shape HxW
        gradient orientations in radians
    """
    
    dx = np.zeros(image.shape)
    dy = np.zeros(image.shape)
    
    weight_dx = np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 0, 0]])
    
    weight_dy = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 1, 0]])
    
    cor_dx = ndimage.correlate(image, weight_dx, dx, mode='nearest')
    cor_dy = ndimage.correlate(image, weight_dy, dy, mode='nearest')
    
    dx = dx - image
    dy = dy - image
    
    ori = np.zeros(image.shape)
    
    dydx = np.zeros(image.shape)
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            ans = 0
            if dx[y,x] == 0 and dy[y,x] == 0:
                ans = 0
            elif dx[y,x] == 0:
                ans = dy[y,x]
            else:
                ans = dy[y,x] / dx[y,x]
            dydx[y,x] = ans
    
    ori = np.arctan(dydx)
    
    mag = np.zeros(image.shape)
    mag = np.sqrt( dx * dx + dy * dy )
    
    return (mag,ori)