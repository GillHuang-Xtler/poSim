import numpy as np
import matplotlib.pyplot as plt

def calColor(weights):
    res=sum(weights)
    weights=weights/res
    weights-=min(weights)
    weights*=(1/max(weights))
    return weights

if __name__ == "__main__":
    # m=calColor(weights=np.array([[0.97,0.96,0.90,0.92,0.95,0.95,0.95,0.90,0.91,0.92],[0.86,0.85,0.88,0.89,0.88,0.79,0.86,0.83,0.80,0.83]]))
    m=np.array([[0.97, 0.96, 0.90, 0.92, 0.95, 0.95, 0.95, 0.90, 0.91, 0.92],[0.86, 0.85, 0.88, 0.89, 0.88, 0.79, 0.86,
               0.83, 0.80, 0.83]])
    m.reshape(2,10)
    plt.imshow(m, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar()
    plt.show()
    plt.savefig('./imgs/attention.png')