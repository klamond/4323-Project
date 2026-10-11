
import numpy as np
import matplotlib.pyplot as plt


# take in a given workspace. Should have x/y dimension, equation definine x axis topography, equation definine y axis topography

# start off with an example x-function

# set a workspace discretization

# camera i think we should use -> https://www.robotshop.com/products/yahboom-yahboom-usb-industrial-ai-vision-camera-separate-camera?qd=f3c1473128dc16999d6625ff76df9b67
# we can also use https://iflightrc.shop/products/racecam-mini-night-camera?v_id=variant_01KS6BBN1H4M5PK5X08CNQ18Q9



class ext_workspace:
    

    def __init__(self, x_dim, y_dim, x_func, y_func):



        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_func = x_func # function that CONTAINS x, ect.
        self.y_func = y_func # use lambda statents in workspace defn.

W1 = ext_workspace(500, 500, lambda x : (x*0.04)**2, lambda y : - (y*0.04)**2)





# Function that essensially converts between realative and absolute headings. can use for velocity, ect.
# vector_body defines the vector that is in the body frame. look in the image in /docs for more information
def body2NED(yaw, pitch, roll, vector_body):

    y = np.deg2rad(yaw) #movement about the axis
    p = np.deg2rad(pitch)
    r = np.deg2rad(roll)


    _R = np.array([[np.cos(y)*np.cos(p), np.cos(y)*np.sin(r)*np.sin(p)-np.cos(r)*np.sin(y), np.sin(r)*np.sin(y)+np.cos(r)*np.cos(y)*np.sin(p)],
              [np.cos(p)*np.sin(y), np.cos(p)*np.cos(y)+np.sin(r)*np.sin(y)*np.sin(p), -np.cos(y)*np.sin(r)+np.cos(r)*np.sin(y)*np.sin(p)],
              [-np.sin(p), np.cos(p)*np.sin(r), np.cos(p)*np.cos(r)]])

    #_R is your rotation matrix, just need to figure out how the vector relates to yaw, pitch, and roll
    # your vector represnts the axis of yaw, pitch, and roll (x y z respectivley)
    # NED output is an absolute position

    vector_body = np.array(vector_body).reshape(3,1)
    res_NED =  _R @ vector_body
    #to go the other way, just use the transpose of R

    print(res_NED)

#viginette function, measurement model confidence distribution

def vis_confidence(x_i, y_i, z_ref, z_t):
    p_max = 0.99
    variance_cam = 0.3

    p_ci = p_max * np.exp(- (x_i ** 2 * y_i ** 2) / (2*variance_cam)) * (z_ref / z_ref + np.abs(z_t - z_ref))

    return p_ci


m = 0.65 #kg
l = 0.225 #moment arm len - meters
i_xx = i_yy = 0.0086 #roll and pitch moments of inertia kg*m^2
i_zz = 0.0172 #yaw moment of inertia
j_t = 3.7404e-5 #rotor rotational inertia
k_f = 3.13e-5 #thrust coeff
k_m = 9e-7 #moment/drag coeff
    
body2NED(90,30,60, [1,1,1])



z_t = 50 # quadrotor height
fov_x = 81.8 # from camera datasheet
fov_y = 66



field_x = 2*z_t*np.tan(np.deg2rad(fov_x)/2) #how many meters the camera can see in the horizontal direction
field_y = 2*z_t*np.tan(np.deg2rad(fov_y)/2)

print(field_x * field_y)

# particle filter code/ EKF code
# EKF is faster, but particle filter is more accurate
# imu = motion model, camera = measurement model

# measurement model uncertainty



# we have an initial position, p0


res = 1 # dx and dy

x_max = W1.x_dim
y_max = W1.y_dim

xc = np.arange(0, x_max, res)
yc = np.arange(0, y_max, res)
xx, yy = np.meshgrid(xc,yc)
zz = W1.x_func(xx) + W1.y_func(yy)


h = plt.contourf(xx, yy, zz)
plt.colorbar(label = 'Z-Position')
plt.axis('scaled')
plt.title('Discretized Workspace')
plt.xlabel('X-Position')
plt.ylabel('Y-Position')
plt.savefig('contwrk.png', dpi = 600)
plt.show()










