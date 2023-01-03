import tensorflow.keras.backend as K


def Entropy(wealth=None, w=None, loss_param=1):
    _lambda = loss_param
    print("lambda is ", _lambda)

    # Entropy (exponential) risk measure
    print("mean is ", K.mean(K.exp(-_lambda*wealth)))
    return (1/_lambda)*K.log(K.mean(K.exp(-_lambda*wealth)))
