import skimage as ski 

def hog(image,bsize=8,norient=9):
    
    """
    This function takes a grayscale image and returns a 3D array
    containing the histogram of gradient orientations descriptor (HOG)
    We follow the convention that the histogram covers gradients starting
    with the first bin at -pi/2 and the last bin ending at pi/2.
    
    Parameters
    ----------
    image : 2D float array of shape HxW
         An array containing pixel brightness values
    
    bsize : int
        The size of the spatial bins in pixels, defaults to 8
        
    norient : int
        The number of orientation histogram bins, defaults to 9
        
    Returns
    -------
    ohist : 3D float array of shape (H/bsize,W/bsize,norient)
        edge orientation histogram
        
    """   
    
    # determine the size of the HOG descriptor
    (h,w) = image.shape
    h2 = int(np.ceil(h/float(bsize)))
    w2 = int(np.ceil(w/float(bsize)))
    ohist = np.zeros((h2,w2,norient))
    
    # pad the input image on right and bottom as needed so that it 
    # is a multiple of bsize
    
    nearest_mult_w = bsize * np.ceil( w / float(bsize) )
    nearest_mult_h = bsize * np.ceil( h / float(bsize) )
    
    pad_w = int(nearest_mult_w - w)
    pad_h = int(nearest_mult_h - h)
    
    pw = (0,pad_w) #amounts to pad on left and right side
    ph = (0,pad_h) #amounts to pad on bottom and top side
    image = np.pad(image,(ph,pw),'edge')
    
    # make sure we did the padding correctly
    assert(image.shape==(h2*bsize,w2*bsize))
    
    # compute image gradients
    (mag,ori) = mygradient(image)
    
    # choose a threshold which is 10% of the maximum gradient magnitude in the image
    thresh = float(np.amax(mag)) * 0.1
    
    
    # separate out pixels into orientation channels, dividing the range of orientations
    # [-pi/2,pi/2] into norient equal sized bins and count how many fall in each block    
    binEdges = np.linspace(-np.pi/2, np.pi/2, norient+1);
    
    # as a sanity check, make sure every pixel gets assigned to at most 1 bin.
    bincount = np.zeros((h2*bsize,w2*bsize))   
    for i in range(norient):
        #create a binary image containing 1s for pixels at the ith 
        #orientation where the magnitude is above the threshold.
        B = np.zeros(mag.shape)
        B[(mag > thresh) & ((ori>=binEdges[i]) & (ori<binEdges[i+1]))] = 1
        B[(mag <= thresh) | ((ori<binEdges[i]) | (ori>=binEdges[i+1]))] = 0
    
        #sanity check: record which pixels have been selected at this orientation
        bincount = bincount + B
        
        #pull out non-overlapping bsize x bsize blocks
        chblock = ski.util.view_as_windows(B,(bsize,bsize),step=bsize)
    
        #sum up the count for each block and store the results
        ohist[:,:,i] = np.sum(chblock, axis=(2,3))
       
    #each pixel should have only selected at most once
    assert(np.all(bincount<=1))

    # lastly, normalize the histogram so that the sum along the orientation dimension is 1
    # note: don't divide by 0! If there are no edges in a block (i.e. the sum of counts
    # is 0) then your code should leave all the values as zero. 
    
    block_sum = ohist.sum(axis=2)
    block_sum[block_sum == 0] = 1
    ohist = ohist / block_sum[:,:,np.newaxis]
    
    assert(ohist.shape==(h2,w2,norient))
#     print(ohist)
    
    return ohist