import numpy as np


class HandPositionKalmanFilter:
    """
    Kalman filter for tracking 2D hand position with constant velocity model.

    State vector: [x, y, vx, vy]
    - (x, y): position
    - (vx, vy): velocity
    """

    def __init__(self, fps=25, process_noise=0.1, measurement_noise=0.005):
        """
        Args:
            fps: Frames per second of video
            process_noise: How much we expect hand motion to vary (higher = less smooth)
            measurement_noise: Expected noise in MediaPipe detections
        """
        self.dt = 1.0 / fps  # Time between frames

        # State: [x, y, vx, vy]
        self.x = np.zeros(4)  # State estimate
        self.P = np.eye(4) * 1.0  # Estimate covariance (uncertainty)

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, self.dt, 0],      # x = x + vx*dt
            [0, 1, 0, self.dt],      # y = y + vy*dt
            [0, 0, 1, 0],            # vx = vx (constant velocity)
            [0, 0, 0, 1]             # vy = vy
        ])

        # Measurement matrix (we only observe position, not velocity)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process noise covariance (models uncertainty in constant velocity
        # assumption)
        q = process_noise
        self.Q = np.array([
            [self.dt**4 / 4, 0, self.dt**3 / 2, 0],
            [0, self.dt**4 / 4, 0, self.dt**3 / 2],
            [self.dt**3 / 2, 0, self.dt**2, 0],
            [0, self.dt**3 / 2, 0, self.dt**2]
        ]) * q

        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise

        self.initialized = False

    def predict(self):
        """Predict next state (time update) - ALWAYS called each timestep"""
        if not self.initialized:
            return None

        # Predict state
        self.x = self.F @ self.x

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x[:2].copy()  # Return predicted position

    def update(self, measurement):
        """
        Update state with measurement (measurement update).
        Should be called AFTER predict() if measurement is available.
        """
        if not self.initialized:
            # Initialize with first measurement
            self.x[:2] = measurement
            self.x[2:] = 0  # Zero velocity initially
            self.P = np.eye(4) * 1.0  # Reset covariance
            self.initialized = True
            return self.x[:2].copy()

        # Innovation (measurement residual)
        z = np.array(measurement)
        y_innov = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y_innov

        # Update covariance
        identity_matrix = np.eye(4)
        self.P = (identity_matrix - K @ self.H) @ self.P

        return self.x[:2].copy()  # Return filtered position

    def get_velocity(self):
        """Get current velocity estimate"""
        if not self.initialized:
            return np.array([0, 0])
        return self.x[2:].copy()

    def get_position_uncertainty(self):
        """Get uncertainty in position estimate (standard deviation)"""
        if not self.initialized:
            return np.array([float('inf'), float('inf')])
        return np.sqrt(np.diag(self.P[:2, :2]))

    def reset(self):
        """Reset the filter"""
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1.0
        self.initialized = False
