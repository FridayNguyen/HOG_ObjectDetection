from skimage.transform import resize

def learn_template(posfiles,negfiles,tsize=np.array([16,16]),bsize=8,norient=9):
    """
    This function takes a list of positive images that contain cropped
    examples of an object + negative files containing cropped background
    and a template size. It produces a HOG template and generates visualization
    of the examples and template
    
    Parameters
    ----------
    posfiles : list of str
         Image files containing cropped positive examples
    
    negfiles : list of str
        Image files containing cropped negative examples

    tsize : (int,int)
        The height and width of the template in blocks
    
    Returns
    -------
    template : float array of size tsize x norient
        The learned HOG template
    
    """           
    
    #compute the template size in pixels 
    #corresponding to the specified template size (given in blocks)
    tsize_pix=bsize*tsize

    #figure to show positive training examples
    fig1 = plt.figure()
    pltct = 1

    #accumulate average positive and negative templates
    pos_t = np.zeros((tsize[0],tsize[1],norient),dtype=float) 
    for file in posfiles:
        #load in a cropped positive example
        img = plt.imread('images/' + file)
        if (img.dtype == np.uint8):
            img = img.astype(float) / 256
        img = np.average(img, -1)
        img_scaled = resize(img, tsize_pix)

        #convert to grayscale and resize to fixed dimension tsize_pix
        #using skimage.transform.resize if needed.

        #display the example. if you want to train with a large # of examples, 
        #you may want to modify this, e.g. to show only the first 5.
        ax = fig1.add_subplot(1,len(posfiles),pltct)
        ax.imshow(img_scaled,cmap=plt.cm.gray)
        pltct = pltct + 1
        
        #extract feature
        fmap = hog(img_scaled)

        #compute running average
        pos_t = pos_t + fmap

    pos_t = (1/len(posfiles))*pos_t
    plt.tight_layout()
    plt.show()
    
    # repeat same process for negative examples
    fig2 = plt.figure()
    pltct = 1  
    neg_t = np.zeros((tsize[0],tsize[1],norient),dtype=float) 
    for file in negfiles:
        #load in a cropped positive example
        img = plt.imread('images/' + file)
        if (img.dtype == np.uint8):
            img = img.astype(float) / 256
        img = np.average(img, -1)
        img_scaled = resize(img, tsize_pix)
        
        ax = fig2.add_subplot(1,len(negfiles),pltct)
        ax.imshow(img_scaled,cmap=plt.cm.gray)
        pltct = pltct + 1
        
        #extract feature
        fmap = hog(img_scaled)

        #compute running average
        neg_t = neg_t + fmap
        

    neg_t = (1/len(negfiles))*neg_t
    plt.tight_layout()
    plt.show()

    # add code here to visualize the positive and negative parts of the template
    # using hogvis. you should separately visualize pos_t and neg_t rather than
    # the final tempalte.
    fig3 = plt.figure()
    plt.imshow(hogvis(pos_t))
    plt.show()
    
    fig4 = plt.figure()
    plt.imshow(hogvis(neg_t))
    plt.show()
 

    # now construct our template as the average positive minus average negative
    template = pos_t - neg_t

    fig5 = plt.figure()
    plt.imshow(hogvis(template))
    plt.show()
    
    return template 