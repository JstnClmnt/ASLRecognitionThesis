import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import imageio
import pandas as pd
static=pd.read_csv("50-50/totalresults.csv")
def plot_for_offset(label, n):
    matplotlib.rc('figure', figsize=(40, 35))
    plt.rcParams.update({'font.size': 55})
    fig=plt.figure()
    fig.suptitle("Accuracy of Different Test Sets at Number of States="+str(n)) 
    plt.yticks(np.arange(0,1,step=0.1))
    plt.ylim(0,1)
    y_values=static[static["numHiddenStates"]==n]["Accuracy"].tolist()
    plt.ylabel("Accuracy")
    print(label)
    print(y_values)
    plt.bar(label,y_values)

    # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image
label=static[static["numHiddenStates"]==1]["test case"].tolist()
kwargs_write = {'fps':1.0, 'quantizer':'nq'}
imageio.mimsave('./powers.gif', [plot_for_offset(label, i) for i in range(1,25)], fps=1)