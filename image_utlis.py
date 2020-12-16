def query(X, Y = None, i=208, j = 104):
    
    img1 = X[i,:,:].T;  
    img1=(img1-np.min(img1))/(np.max(img1)-np.min(img1))
    
    if Y is None: 
        ref1 = None
    else: 
        ref1 = Y[i,:,:].T; 
        ref1=(ref1-np.min(ref1))/(np.max(ref1)-np.min(ref1))

    img2 = X[:,j,:].T; 
    img2=(img2-np.min(img2))/(np.max(img2)-np.min(img2))
    
    if Y is None:
        ref2 = None
    else: 
        ref2 = Y[:,j,:].T; 
        ref2=(ref2-np.min(ref2))/(np.max(ref2)-np.min(ref2))

    return (img1, img2, ref1, ref2)


def query_test(x, y = None, i=208, j = 104):
    img1 = x[i,:,:].T; img1 = np.where(img1>0, 0, img1); img1=(img1-np.min(img1))/(np.max(img1)-np.min(img1))
 
    if y is None: ref1 = None
    else: ref1 = y[i,:,:].T; ref1=(ref1-np.min(ref1))/(np.max(ref1)-np.min(ref1))
 
    img2 = x[:,j,:].T; img2 = np.where(img2>0, 0, img2); img2=(img2-np.min(img2))/(np.max(img2)-np.min(img2))
 
    if y is None: ref2 = None
    else: ref2 = y[:,j,:].T; ref2=(ref2-np.min(ref2))/(np.max(ref2)-np.min(ref2))

    return (img1, img2, ref1, ref2)
def cv_canny(t1=150, t2=20, *args):
    img1, img2, ref1, ref2 = args
    edg1, edg2 = cv.Canny( (img1*255).astype(np.uint8),t1,t2 ),cv.Canny((img2*255).astype(np.uint8),t1,t2)
    return (img1, ref1, img2, ref2, edg1, edg2)
def get_height(i, j, threshold=100, *args):
    
    (_, _, _, _, edg1, edg2) = args
    
    h1 = np.where(edg1[:, j]>threshold)[0]  # retrieve height indices
    h2 = np.where(edg2[:, i]>threshold)[0]  
    
    return h1, h2
def filter_height(h1, h2):
    
    n1, n2 = len(h1), len(h2)
    
    #  bad points control
    if n2 >= n1 and n1 > 1:  return h1
    elif n1 >= n2 and n2 > 1: return h2
    elif n1 > 0: return h1
    elif n2 > 0: return h2
    else: return None
def plot_edge(*args):
    (img1, ref1, img2, ref2, edg1, edg2) = args
    
    if ref1 is None:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        ax1, ax2, ax3, ax4 = ax.flatten()

        ax1.imshow(img1); ax1.set_title("original")
        ax2.imshow(edg1); ax2.set_title("Canny")
        
        ax3.imshow(img2); ax3.set_title("original")
        ax4.imshow(edg2); ax4.set_title("Canny")
    else:
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
        ax1, ax2, ax3, ax4, ax5, ax6 = ax.flatten()

        ax1.imshow(img1); ax1.set_title("original")
        ax2.imshow(ref1); ax2.set_title("reference")
        ax3.imshow(edg1); ax3.set_title("in-line Canny")
        ax4.imshow(img2); ax4.set_title("original")
        ax5.imshow(ref2); ax5.set_title("reference")
        ax6.imshow(edg2); ax6.set_title("cross-line Canny")
