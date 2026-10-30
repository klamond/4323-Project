import numpy as np
import matplotlib.pyplot as plt

# Camera used -> https://www.robotshop.com/products/yahboom-yahboom-usb-industrial-ai-vision-camera-separate-camera?qd=f3c1473128dc16999d6625ff76df9b67
# Alternative option -> https://iflightrc.shop/products/racecam-mini-night-camera?v_id=variant_01KS6BBN1H4M5PK5X08CNQ18Q9


class workspace:
    
    def __init__(self, x_dim, y_dim, x_func, y_func, res): #initialization function

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_func = x_func # function that CONTAINS x, ect.
        self.y_func = y_func # use lambda statents in workspace defn.
        self.res = res

        # Everything within the object is computed
        # All relevant data should be contained in object for easy acess


        # nx/ny signfies the number of grid cells.
        self.nx = int(x_dim / self.res)
        self.ny = int(y_dim / self.res)
        self.confidence = np.ones((self.nx, self.ny), dtype = float) * 0.5 # Confidence starts at 0.5

        self.xc = np.arange(0, x_dim, self.res) 
        self.yc = np.arange(0, y_dim, self.res)
        self.xx, self.yy = np.meshgrid(self.xc,self.yc, indexing = 'ij')
        self.zz = self.x_func(self.xx) + self.y_func(self.yy) # Store height map within object


class camera:

    def __init__(fovx, fovy):
        pass





def body2NED(yaw, pitch, roll, vector_body):

    """Converts between realative (body) and absolute headings (NED)"""

    # Probably a redundant function

    # vector_body defines the vector that is in the body frame. look in the image in /docs for more information

    y = np.deg2rad(yaw)
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

    return res_NED



def vis_confidence(x_i, y_i, z_ref, z_t): # z_t is the quadrotor height, 

    """Viginette function, measurement model confidence distribution"""

    p_max = 0.99
    variance_cam = 1.0

    p_ci = p_max * np.exp(- (x_i ** 2 + y_i ** 2) / (2*variance_cam)) * (z_ref / (z_ref + np.abs(z_t - z_ref))) # Given equation

    p_scaled = 0.5 + 0.5 * p_ci # Scales probablility to within [0.5,1]. I think this is what he wants?

    return p_scaled


def update_A(stateVectorX, input, motionParams, omega_r): # takes in a x_dot akin to the one in the overleaf doc. 12 rows

    """Finds your x_dot, so that you can add it to your state space model.
    Accepts (12,) or (N, 12). Returns (12,) or (N, 12)."""

    

    # allow scalar-state calls and batched calls through the same function
    single = (stateVectorX.ndim == 1) # Make sure to leave this in, keeps things working
    X = np.atleast_2d(stateVectorX)          # (N, 12)

    # need to find the jacobian to make an actual "A" matrix, as our system is nonlinear
    # set the _dot values instead of copying for readability putposes
    # input is a tuple, could be a list

    g, m, l, i_xx, i_yy, i_zz, j_r, k_f, k_m = map(motionParams.get, ['g', 'm', 'l', 'i_xx', 'i_yy', 'i_zz', 'j_r', 'k_f', 'k_m'])

    u1 = input[0]
    u2 = input[1]
    u3 = input[2]
    u4 = input[3]

    roll = X[:,0]
    pitch = X[:,2]
    yaw = X[:,4]

    phi_dot = X[:,1]
    theta_dot = X[:,3]
    psi_dot = X[:,5]
    z_dot = X[:,7]
    x_dot = X[:,9]
    y_dot = X[:,11]

    cr = np.cos(roll)
    cp = np.cos(pitch)
    cy = np.cos(yaw)
    sr = np.sin(roll)
    sp = np.sin(pitch)
    sy = np.sin(yaw)

    phi_2dot = (theta_dot * psi_dot * (i_yy - i_zz) - j_r * theta_dot * omega_r + u2) / i_xx
    
    theta_2dot = (phi_dot * psi_dot * (i_zz - i_xx) + j_r * phi_dot * omega_r + u3) / i_yy

    psi_2dot = (phi_dot * theta_dot * (i_xx - i_yy) + u4) / i_zz
    
    z_2dot = g - u1 / m * (cr * cp)
    
    x_2dot = -u1 / m * (cr * sp * cy + sr * sy)
    
    y_2dot = u1 / m * (cy * sr - cr * sy * sp)

    result = np.array([phi_dot, phi_2dot, theta_dot, theta_2dot, psi_dot, psi_2dot, z_dot, z_2dot, x_dot, x_2dot, y_dot, y_2dot]).T

    return result[0] if single else result
    


def jacobian(x, u, params, omega_r):

    """Take the Jacobian of the state vecotor. Using the euler backward difference formula.
    Performed by passing the entire matrix to update_A"""

    n = len(x)
    A = np.zeros((n, n))
    j_dt = 1e-6

    x_minus = np.tile(x,(n,1))
    for i in range(n):
        x_minus[i,i] -= j_dt

    f_current = update_A(x, u, params, omega_r)
    f_minus = update_A(x_minus, u, params, omega_r)

    for i in range(n):
        A[:, i] = (f_current - f_minus[i])/j_dt


    # OLD IMPLEMENTATION (for reference)

    # f_current = update_A(x, u, params, omega_r)

    # for i in range(n):
    #     x_minus = x.copy()
    #     x_minus[i] -= j_dt
    #     f_minus = update_A(x_minus, u, params, omega_r)     
    #     A[:, i] = (f_current - f_minus) / (j_dt)

    return A



def EKF(x_est, P, u, measurement, Q, R, H, params, omega_r, dt):

    """Extended Kalman Filter implementation. With reference to class notes, P is sigma, and measurement z."""

    # 1. PREDICTION UPDATE

    # Predict state
    x_pred = x_est + update_A( #returns the change in each variable.
        x_est, u, params, omega_r
    ) * dt

    # Calculate Jacobian
    A = jacobian(x_est, u, params, omega_r)
    # Convert to discrete-time Jacobian
    F = np.eye(12) + A * dt

    # Predicted covariance
    Sp = F @ P @ F.T + Q

    # 2. MEASUREMENT UPDATE

    # Predicted measurement
    h_x = H @ x_pred

    # Innovation
    innovation = measurement - h_x

    

    # Kalman gain

    # Innovation covariance
    S = H @ Sp @ H.T + R # what does this do???

    K = np.linalg.solve(S, H @ Sp).T # implementation using linalg.solve method


    # K = Sp @ H.T @ np.linalg.inv(H @ Sp @ H.T + R) #implementation from textbook, does the same thing as above

    # Correct state
    x_new = x_pred + K @ innovation


    # Correct covariance (Joseph form) higher compute time than textbook but is more numerically robust

    I = np.eye(12)

    S_new = ((I - K @ H) @ Sp @ (I - K @ H).T + K @ R @ K.T)

    return x_new, S_new


def camera_ft(state_vec, workspace):
    """Camera footprint size and probablility calculator. Rectangular footprint"""
    # Note: do NOT try to factor pitch & roll into implementation.

    height = state_vec[6]
    x_pos = state_vec[8]
    y_pos = state_vec[10]
    yaw = state_vec[4]

    # Calculate relative local field, then superimpose that onto the workspace

    field_x = 2*height*np.tan(np.deg2rad(fov_x)/2) #how many meters the camera can see in the horizontal direction, ect.
    field_y = 2*height*np.tan(np.deg2rad(fov_y)/2) 

    halfx = int(field_x/2)
    halfy = int(field_y/2)

    xvals = np.arange(-halfx, +halfx, res)
    yvals = np.arange(-halfy, +halfy, res)

    xvv, yvv = np.meshgrid(xvals, yvals)

    c_yaw = np.cos(yaw)
    s_yaw = np.sin(yaw)


    # Same rotation method used as on quiz 3, not considering trapezoidal footprint due to complexity and other factors.
    # Can do same thing with a transformation matrix


    x_rot = c_yaw * xvv - s_yaw * yvv
    y_rot = s_yaw * xvv + c_yaw * yvv


    # The matrix for graph rotation 

    x_world = x_rot + x_pos
    y_world = y_rot + y_pos
    
    prb = vis_confidence(xvv, yvv, zref, height) #calculates confidence for each cell, a confidence map for the relative frame
    
    ix = np.rint((x_world) / res).astype(np.int32)
    iy = np.rint((y_world) / res).astype(np.int32)

    valid = (ix >= 0) & (ix < workspace.nx) & (iy >= 0) & (iy < workspace.ny) #finds the cells that are valid, so we dont add up a bunch of 0s

    np.maximum.at(workspace.confidence, (ix[valid], iy[valid]), prb[valid]) # Max fusion, as required by the assignment

    
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
    'z' : 10,
    'z_d' : 0,
    'x' : 250,
    'x_d' : 0,
    'y' : 250,
    'y_d' : 0,
}

W1 = workspace(500, 500, lambda x : (x*0.04)**2, lambda y : - (y*0.04)**2, 1)

res = 1 # dx and dy (cell size(m))

r = 0.0025 # measurement noise covariance

# Initial state estimation uncertainty, initial Sigma
P = np.eye(12) * 0.1

# Process noise covariance (discrete-time)
Q = np.eye(12) * 0.0001

# Measurement noise covariance matrix
R = np.eye(6) * r

# Measurement matrix
H = np.zeros((6, 12))

# What is getting measured (IMU & GPS)?
H[0, 0] = 1 # roll
H[1, 2] = 1 # pitch
H[2,4] = 1 # yaw
H[3, 8] = 1   # X-position
H[4, 10] = 1  # Y-position
H[5, 6] = 1   # Z-position

dt = 0.01

omegaR = 0

# Values from camera datasheet
fov_x = 81.8 # gets used by camera_ft
fov_y = 66
zref = 10 #idk wtf this means, its the designed altitude

time = np.arange(0, 15, dt) # (ti, tf, dt). Makes an array of time values

# replaced by x_true (the actual drone state in your simulation) and x_est (the drone state estimated by the EKF).
# x = np.array(list(initial_cnds.values()))
x_true = np.array(list(initial_cnds.values()), dtype=float)
x_est = x_true.copy()





# Logs for comparing actual and estimated states
x_true_log = np.zeros((len(time), 12))
x_est_log = np.zeros((len(time), 12))

# Use positive hover thrust for initial control input
u = np.array([params['m'] * params['g'], 0, 0, 0]) # Control this


for t in range(len(time)):

    
    x_true = x_true + update_A( #update position
        x_true, u, params, omegaR
    ) * dt + np.random.multivariate_normal(
        np.zeros(12), Q
    )

    
    measurement = H @ x_true + np.random.multivariate_normal( #noise
        np.zeros(6), R
    )

    
    x_est, P = EKF( # P is covariance. should probably be 'S' to allign with the notes
        x_est, P, u, measurement,
        Q, R, H, params, omegaR, dt
    )


    camera_ft(x_est, W1)


    
    x_true_log[t] = x_true #base camera measurement off of true or estimate?
    x_est_log[t] = x_est

    

    #create grid around x, + 50% FOV to either side


#plt.plot(xvv,yvv, marker='.', color='k', linestyle='none')


plt.figure()

plt.contourf(W1.xx, W1.yy, W1.confidence, cmap='plasma')
plt.colorbar(label = 'Confidence')
plt.axis('scaled')
plt.title('Confidence Function')
plt.xlabel('X-Position')
plt.ylabel('Y-Position')


# plt.figure() # Terrain Map
# h = plt.contourf(W1.xx, W1.yy, W1.zz)
# plt.colorbar(label = 'Z-Position')
# plt.axis('scaled')
# plt.title('Discretized Workspace Terrain')
# plt.xlabel('X-Position')
# plt.ylabel('Y-Position')
# plt.savefig('contwrk.png', dpi = 600)



# Actual X-position agaisnt the EKF estimate
plt.figure()

plt.plot(time, x_true_log[:, 8], label="Actual X-position")
plt.plot(time, x_est_log[:, 8], "--", label="EKF Estimate")

plt.xlabel("Time (s)")
plt.ylabel("X-position (m)")
plt.title("EKF Position Estimation")
plt.legend()
plt.grid(True)

# plt.show()

# Relative estimation error
error_x = x_true_log[:, 8] - x_est_log[:, 8]

plt.figure()
plt.plot(time, error_x)
plt.xlabel("Time (s)")
plt.ylabel("X-position error (m)")
plt.title("EKF Estimation Error")
plt.grid(True)


plt.figure()
plt.plot(x_true_log[:, 8], x_true_log[:, 10])
plt.title("X-Y graph")
plt.show()

