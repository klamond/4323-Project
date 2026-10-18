
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


def update_A(stateVectorX, input, motionParams, omega_r): # takes in a x_dot akin to the one in the overleaf doc. 12 rows
    # need to find the jacobian to make an actual "A" matrix, as our system is nonlinear
    # set the _dot values instead of copying for readability putposes
    # input is a tuple, could be a list

    g, m, l, i_xx, i_yy, i_zz, j_r, k_f, k_m = map(motionParams.get, ['g', 'm', 'l', 'i_xx', 'i_yy', 'i_zz', 'j_r', 'k_f', 'k_m'])

    u1 = input[0]
    u2 = input[1]
    u3 = input[2]
    u4 = input[3]

    roll = stateVectorX[0]
    pitch = stateVectorX[2]
    yaw = stateVectorX[4]

    phi_dot = stateVectorX[1]
    theta_dot = stateVectorX[3]
    psi_dot = stateVectorX[5]
    z_dot = stateVectorX[7]
    x_dot = stateVectorX[9]
    y_dot = stateVectorX[11]

    cr = np.cos(roll)
    cp = np.cos(pitch)
    cy = np.cos(yaw)
    sr = np.sin(roll)
    sp = np.sin(pitch)
    sy = np.sin(yaw)

    phi_2dot = (theta_dot * psi_dot * (i_xx - i_yy) - j_r * theta_dot * omega_r + u2) / i_xx
    
    theta_2dot = (phi_dot * psi_dot * (i_zz - i_xx) + j_r * phi_dot * omega_r + u3) / i_yy

    psi_2dot = (phi_dot * theta_dot * (i_xx - i_yy) + u4)
    
    z_2dot = g - u1 / m * (cr * cp)
    
    x_2dot = -u1 / m * (cr * sp * cy + sr * sy)
    
    y_2dot = u1 / m * (cy * sr - cr * sy * sp)

    result = np.array([phi_dot, phi_2dot, theta_dot, theta_2dot, psi_dot, psi_2dot, z_dot, z_2dot, x_dot, x_2dot, y_dot, y_2dot])

    return result
    

# Initial Variable declarations

params = {
    'g' : 9.81, #m/s2
    'm' : 0.65, #kg
    'l' : 0.225, #moment arm len - meters
    'i_xx' : 0.0086, #roll moment of inertia kg*m^2
    'i_yy' : 0.0086, #pitch moment of inertia
    'i_zz' : 0.0172, #yaw moment of inertia
    'j_r' : 3.7404e-5, #rotor rotational inertia
    'k_f' : 3.13e-5, #thrust coeff
    'k_m' : 9e-7, #moment/drag coeff
}

initial_cnds = {
    'roll' : 0, # in radians
    'roll_d' : 0,
    'pitch' : 0.1,
    'pitch_d' : 0,
    'yaw' : 0,
    'yaw_d' : 0,
    'z' : 0,
    'z_d' : 0,
    'x' : 1,
    'x_d' : 0,
    'y' : 0,
    'y_d' : 0,
}


# Noise

# statistics of the process disturbance
meanProcess=np.array([0,0])
covarianceProcess=np.array([[0.002, 0],[0, 0.002]])
 
# statistics of the measurement noise
meanNoise=np.array([0])
covarianceNoise=np.array([[0.001]]) # this stuff is for later



dt = 0.01

omegaR = 0


# These should technically be transposed, we just have to keep this in mind
# Definitions for the state vectors used in the 


time = np.arange(0, 100, dt) # ti, tf, dt

u = np.array([-10,0,0,0]) # thrust, roll moment, pitch moment, yaw moment #TODO, check if we have the right rotation eqn. i don't know why veloity puts it down.

x = np.array(list(initial_cnds.values()))

x_dot = np.array([0] * 12)

x_log = np.zeros((len(time), 12)) # log of everything, NOT transposed, no acceleration recorded



# The motion model, how it moves (precisley, add uncert. later)

for t in range(len(time)):

    x_dot = update_A(x, u, params, 0)


    for k in range(12): #static, will not have to calculate len every time.
        x[k] = x[k] + x_dot[k] * dt

    x_log[t] = x

y =[x_log[11][x] for x in range(12)]

print(y)

fig = plt.plot([x_log[x][10] for x in range(len(time))],[x_log[x][8] for x in range(len(time))])
plt.title("position, absolute")
plt.show()

fig2 = plt.plot(time,[x_log[x][6] for x in range(len(time))])
plt.title("height")
plt.show()


# x_dot = [phi_dot phi_2dot theta_dot theta_2dot tri_dot tri_2dot z_dot z_2dot x_dot x_2dot y_dot y_2dot]T



body2NED(90,30,60, [1,1,1])





z_t = 50 # quadrotor height
fov_x = 81.8 # from camera datasheet
fov_y = 66



field_x = 2*z_t*np.tan(np.deg2rad(fov_x)/2) #how many meters the camera can see in the horizontal direction
field_y = 2*z_t*np.tan(np.deg2rad(fov_y)/2)

print(f"{field_x}, {field_y}")

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