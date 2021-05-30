import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    temp=e_x / e_x.sum(axis=0) # only difference
    
    if np.isnan(temp).any()==True:
        return [0.0,1.0,0.0]
    else:
        return temp