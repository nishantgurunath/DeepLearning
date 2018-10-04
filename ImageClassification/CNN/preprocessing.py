import numpy as np

def sample_zero_mean(x):
    """
    Make each sample have a mean of zero by subtracting mean along the feature axis.
    :param x: float32(shape=(samples, features))
    :return: array same shape as x
    """
    U = np.mean(x,axis=1)
    D = np.subtract(x,U.reshape(len(U),1))
    return D


def gcn(x, scale=55., bias=0.01):
    """
    GCN each sample (assume sample mean=0)
    :param x: float32(shape=(samples, features))
    :param scale: factor to scale output
    :param bias: bias for sqrt
    :return: scale * x / sqrt(bias + sample variance)
    """
    var = np.var(x,axis=1)
    return np.divide(scale*x,np.sqrt(bias + var.reshape(len(var),1)))
    


def feature_zero_mean(x, xtest):
    """
    Make each feature have a mean of zero by subtracting mean along sample axis.
    Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features))
    :param xtest: float32(shape=(samples, features))
    :return: tuple (x, xtest)
    """
    U = np.mean(x,axis=0)
    D = np.subtract(x,U)
    Dtest = np.subtract(xtest,U)
    return (D, Dtest)


def zca(x, xtest, bias=0.1):
    """
    ZCA training data. Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features)) (assume mean=0)
    :param xtest: float32(shape=(samples, features))
    :param bias: bias to add to covariance matrix
    :return: tuple (x, xtest)
    """
    n = len(x)
    x = x.T
    xtest = xtest.T
    xxT = x.dot(x.T)/(n)
    P, eigenval,PT = np.linalg.svd(xxT)
    Dinv_sqrt = np.diag(1/np.sqrt(eigenval))
    xxT_minushalf = (P.dot(Dinv_sqrt)).dot(PT)
    W = xxT_minushalf + bias
    Y = W.dot(x)
    Ytest = W.dot(xtest)
    return (Y.T, Ytest.T)


def cifar_10_preprocess(x, xtest, image_size=32):
    """
    1) sample_zero_mean and gcn xtrain and xtest.
    2) feature_zero_mean xtrain and xtest.
    3) zca xtrain and xtest.
    4) reshape xtrain and xtest into NCHW
    :param x: float32 flat images (n, 3*image_size^2)
    :param xtest float32 flat images (n, 3*image_size^2)
    :param image_size: height and width of image
    :return: tuple (new x, new xtest), each shaped (n, 3, image_size, image_size)
    """
    x1 = gcn(sample_zero_mean(x))
    xtest1 = gcn(sample_zero_mean(xtest))
    x2, xtest2 = feature_zero_mean(x1,xtest1)
    x2, xtest2 = zca(x2,xtest2)
    xnew = x2.reshape(len(x2),3,image_size,image_size)
    xtestnew = xtest2.reshape(len(xtest2),3,image_size,image_size)
    return xnew, xtestnew


A = np.array([[1,1.1,0.9,1,1,1,1,1,1,1,1,1],
     [6,6,6.1,6,5.9,6,6,6,6,6,6,6],
     [11,11.1,11,11.1,10.9,11,11,11,11,11,11,11]])
#print A
U = np.mean(A,axis=0)
U1 = np.mean(A,axis=1) 
#print U1
#D = np.subtract(A,U1.reshape(len(U1),1))
#print D

#print np.mean(D, axis = 1)

B, Btest = feature_zero_mean(A,A)
#print B
#print np.mean(B,axis=0)
#print np.var(B,axis=0)

#print np.sqrt((n-1)*(np.linalg.inv(B.dot(B.T))))

#print np.linalg.inv(B.dot(B.T))

C, Ctest = zca(B, B, 1e-10)
#print np.mean(C,axis=0)
#print np.var(C,axis=0)
#print C.T.dot(C)
#print B.T.dot(B)
#print C

D,E = cifar_10_preprocess(A,A,image_size=2)

#print D.shape
