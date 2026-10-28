
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

    phi_2dot = (theta_dot * psi_dot * (i_yy - i_zz) - j_r * theta_dot * omega_r + u2) / i_xx
    
    theta_2dot = (phi_dot * psi_dot * (i_zz - i_xx) + j_r * phi_dot * omega_r + u3) / i_yy

    psi_2dot = (phi_dot * theta_dot * (i_xx - i_yy) + u4) / i_zz
    
    z_2dot = g - u1 / m * (cr * cp)
    
    x_2dot = -u1 / m * (cr * sp * cy + sr * sy)
    
    y_2dot = u1 / m * (cy * sr - cr * sy * sp)

    result = np.array([phi_dot, phi_2dot, theta_dot, theta_2dot, psi_dot, psi_2dot, z_dot, z_2dot, x_dot, x_2dot, y_dot, y_2dot])

    return result
    
# Calculates the Jacobian matix, backward difference, good for control problems

def jacobian(x, u, params, omega_r):

    n = len(x)
    A = np.zeros((n, n))
    j_dt = 1e-6



    f_current = update_A(x, u, params, omega_r)


    for i in range(n):

        x_minus = x.copy()
        

        x_minus[i] -= j_dt
        

        f_minus = update_A(x_minus, u, params, omega_r)
        

        A[:, i] = (f_current - f_minus) / (j_dt)

    return A

# Extended Kalman Filter

def EKF(x_est, P, u, measurement, Q, R, H, params, omega_r, dt): #measurement is z in the textbook.

    # PREDICTION

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

    # CORRECTION

    # Predicted measurement
    h_x = H @ x_pred

    # Innovation
    innovation = measurement - h_x

    # Innovation covariance
    # S = H @ Sp @ H.T + R # what does this do???

    # # Kalman gain
    # K = np.linalg.solve(S, H @ Sp).T


    K = Sp @ H.T @ np.linalg.inv(H @ Sp @ H.T + R) #implementation from textbook

    # Correct state
    x_new = x_pred + K @ innovation

    # should take partial derivative if H is not constant (later idea)

    # Correct covariance (Joseph form) don't have to do this, can just use the equation from the textbook.
    # J. form has a higher compute time but is more numerically robust.

    


    #Textbook just does 
    I = np.eye(12)

    S_new = ((I - K @ H) @ Sp @ (I - K @ H).T + K @ R @ K.T)

    return x_new, S_new


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
# meanProcess=np.array([0,0])
# covarianceProcess=np.array([[0.002, 0],[0, 0.002]])
 
# statistics of the measurement noise
# meanNoise=np.array([0])
# covarianceNoise=np.array([[0.001]]) # this stuff is for later





r = 0.0025 # measurement noise covariance

# Initial state estimation uncertainty, initial Sigma
P = np.eye(12) * 0.1

# Process noise covariance (discrete-time)
Q = np.eye(12) * 0.0001

# Measurement noise covariance matrix
R = np.diag([r]*6)

# Measurement matrix
H = np.zeros((6, 12))
# What is getting measured (IMU & GPS)?
H[0, 0] = 1 # roll
H[0, 2] = 1 # pitch
H[0,4] = 1 # yaw
H[0, 8] = 1   # X-position
H[1, 10] = 1  # Y-position
H[2, 6] = 1   # Z-position

dt = 0.01

omegaR = 0

# These should technically be transposed, we just have to keep this in mind


time = np.arange(0, 100, dt) # ti, tf, dt

u = np.array([-10,0,0,0]) # thrust, roll moment, pitch moment, yaw moment #TODO, check if we have the right rotation eqn. i don't know why veloity puts it down.

# replaced by x_true (the actual drone state in your simulation) and x_est (the drone state estimated by the EKF).
# x = np.array(list(initial_cnds.values()))
x_true = np.array(list(initial_cnds.values()), dtype=float)
x_est = x_true.copy()

# x_est error, for testing, remove later.
x_est[8] += 0.5   # Estimated X-position is 0.5 m too high
x_est[10] -= 0.5  # Estimated Y-position is 0.5 m too low

# The motion model, how it moves (precisley, add uncert. later)

# for t in range(len(time)):

    # x_dot = update_A(x, u, params, 0)


    # for k in range(12): #static, will not have to calculate len every time.
        # x[k] = x[k] + x_dot[k] * dt

    # x_log[t] = x


# Logs for comparing actual and estimated states
x_true_log = np.zeros((len(time), 12))
x_est_log = np.zeros((len(time), 12))

# Use positive hover thrust
u = np.array([params['m'] * params['g'], 0, 0, 0])

for t in range(len(time)):

    # 1. Update actual drone position
    x_true = x_true + update_A(
        x_true, u, params, omegaR
    ) * dt + np.random.multivariate_normal(
        np.zeros(12), Q
    )

    # 2. Generate noisy sensor measurements
    measurement = H @ x_true + np.random.multivariate_normal(
        np.zeros(6), R
    )

    # 3. Run EKF
    x_est, P = EKF(
        x_est, P, u, measurement,
        Q, R, H, params, omegaR, dt
    )

    # 4. Store results
    x_true_log[t] = x_true
    x_est_log[t] = x_est


print(jacobian(x_est, u, params, omegaR))

# y =[x_log[11][x] for x in range(12)]

# print(y)

# fig = plt.plot([x_log[x][10] for x in range(len(time))],[x_log[x][8] for x in range(len(time))])
# plt.title("position, absolute")
# plt.show()

# fig2 = plt.plot(time,[x_log[x][6] for x in range(len(time))])
# plt.title("height")
# plt.show()


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
# plt.show()


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
plt.show()