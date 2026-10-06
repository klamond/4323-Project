import control as ct
import numpy as np
import matplotlib.pyplot as plt

# take in a given workspace. Should have x/y dimension, equation definine x axis topography, equation definine y axis topography

# start off with an example x-function

# set a workspace discretization


class ext_workspace:
    

    def __init__(self, x_dim, y_dim, x_func, y_func):



        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_func = x_func # function that CONTAINS x, ect.
        self.y_func = y_func # use lambda statents in workspace defn.

W1 = ext_workspace(500, 500, lambda x : (x*0.04)**2, lambda y : - (y*0.04)**2)





# x_max = 500 
# y_max = 500



# xc = np.arange(0, x_max, res)
# yc = np.arange(0, y_max, res)


res = 1 # dx and dy

x_max = W1.x_dim
y_max = W1.y_dim

xc = np.arange(0, x_max, res)
yc = np.arange(0, y_max, res)
xx, yy = np.meshgrid(xc,yc)
zz = W1.x_func(xx) + W1.y_func(yy)



# print(xc)
# print(yc)



# print(xx)

#sc = 0.04

# #zz = np.sin(np.sqrt(xx**2 + yy**2))/np.sqrt(xx**2 + yy**2) # height map
#zz = ((xx*sc)**2 -  (yy*sc) ** 2)



h = plt.contourf(xx, yy, zz)
plt.colorbar(label = 'Z-Position')
plt.axis('scaled')
plt.title('Discretized Workspace')
plt.xlabel('X-Position')
plt.ylabel('Y-Position')
plt.savefig('contwrk.png', dpi = 600)
plt.show()










