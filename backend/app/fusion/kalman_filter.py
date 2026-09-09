# 10Hz Dead Reckoning Backend Engine - SIH 2026
import numpy as np
from typing import Dict, Any, Optional
from collections import deque
from app.fusion.coordinates import (
    body_to_nav_acceleration,
    calculate_tilt_compensated_heading,
    GRAVITY
)

class IMUKalmanFilter:
    """
    9-DOF Extended Kalman Filter for Dead Reckoning with Pedestrian Step & Cadence Fusion.
    Fuses Accelerometer (3-axis), Gyroscope (3-axis), and Magnetometer (3-axis).
    
    Supports 3 Operational Modes:
    1. STATIONARY (Rest): Hard-locks velocity to 0.0 m/s and freezes position.
    2. MOVING (Walking): Actively integrates step kinematics along the estimated heading.
    3. ADAPTIVE: Auto-detects transition between REST and MOVING from sensor kinematics.
    """
    def __init__(self, initial_state: Optional[np.ndarray] = None, initial_covariance: Optional[np.ndarray] = None):
        # 9x1 State Vector: [px, py, pz, vx, vy, vz, roll, pitch, yaw]^T
        if initial_state is not None:
            self.x = initial_state.astype(np.float64)
            if self.x.ndim == 1:
                self.x = self.x.reshape((9, 1))
        else:
            self.x = np.zeros((9, 1), dtype=np.float64)

        # 9x9 Error Covariance Matrix
        if initial_covariance is not None:
            self.P = initial_covariance.astype(np.float64)
        else:
            self.P = np.eye(9, dtype=np.float64) * 0.1

        # Process Noise Covariance (Q) - Tuned for 10Hz (dt = 0.1s)
        self.Q = np.eye(9, dtype=np.float64) * 0.05
        self.Q[0:3, 0:3] *= 0.02
        self.Q[3:6, 3:6] *= 0.10
        self.Q[6:9, 6:9] *= 0.05

        # Measurement Noise Covariance (R)
        self.R = np.eye(3, dtype=np.float64) * 0.2

        self.last_timestamp: Optional[float] = None

        # Kinematic Motion State & Step Detection
        self.movement_state: str = "REST"  # "REST" or "MOVING"
        self.step_count: int = 0
        self._accel_history = deque(maxlen=10)  # 1-second rolling window at 10Hz
        self._last_step_time = 0.0
        self._step_length = 0.68  # Standard pedestrian stride (meters)

    def predict(self, accel_body: np.ndarray, gyro_body: np.ndarray, dt: float, mode: str = "adaptive"):
        """
        Kinematic prediction step over elapsed time dt (seconds).
        mode can be: 'stationary', 'moving', or 'adaptive'
        """
        if dt <= 0.0:
            dt = 0.10

        roll, pitch, yaw = float(self.x[6, 0]), float(self.x[7, 0]), float(self.x[8, 0])

        # 1. Attitude Propagation from Gyroscope
        gx, gy, gz = gyro_body[0], gyro_body[1], gyro_body[2]
        self.x[6, 0] += gx * dt
        self.x[7, 0] += gy * dt
        self.x[8, 0] += gz * dt
        self.x[8, 0] = (self.x[8, 0] + np.pi) % (2 * np.pi) - np.pi

        # 2. Transform body acceleration into world navigation frame (gravity-free)
        a_nav = body_to_nav_acceleration(accel_body, roll, pitch, yaw)

        # 3. Kinematic Motion State & Step Analysis
        accel_mag = float(np.linalg.norm(accel_body))
        gyro_mag = float(np.linalg.norm(gyro_body))
        self._accel_history.append(accel_mag)
        
        # Calculate dynamic acceleration variance
        accel_var = float(np.var(self._accel_history)) if len(self._accel_history) >= 4 else 0.0

        mode = str(mode).lower().strip()
        # Mode Determination:
        if mode == "stationary":
            is_moving = False
        elif mode == "moving":
            is_moving = True
        else:
            # Adaptive detection: dynamic variance > 0.15 or deviation from 1g > 0.40 indicates human movement
            is_moving = (accel_var > 0.15) or (abs(accel_mag - GRAVITY) > 0.45) or (gyro_mag > 0.25)

        if is_moving:
            self.movement_state = "MOVING"

            # Step detection & velocity projection
            current_time = self.last_timestamp or 0.0
            # Check for walking step peak (at least 0.35s between steps = max 2.8 steps/s)
            is_step_peak = (accel_mag - GRAVITY) > 0.35 and (current_time - self._last_step_time) > 0.35
            cadence_step = (mode == "moving") and ((current_time - self._last_step_time) >= 0.55)
            
            if is_step_peak or cadence_step:
                self.step_count += 1
                self._last_step_time = current_time

            if is_step_peak or mode == "moving":
                # Walking velocity along current heading
                speed = 1.15  # ~1.15 m/s typical walking speed
                self.x[3, 0] = speed * np.cos(yaw)  # Vx (East)
                self.x[4, 0] = speed * np.sin(yaw)  # Vy (North)
                self.x[5, 0] = 0.0                  # Vz (Vertical)

                # Advance position
                self.x[0, 0] += self.x[3, 0] * dt
                self.x[1, 0] += self.x[4, 0] * dt
            else:
                # Coast with decay
                self.x[3:6, 0] *= 0.90
                self.x[0, 0] += self.x[3, 0] * dt
                self.x[1, 0] += self.x[4, 0] * dt

        else:
            # Hard-lock to REST / STATIONARY
            self.movement_state = "REST"
            self.x[3:6, 0] = 0.0
            a_nav = np.zeros(3)
            self.P[3:6, 3:6] = np.eye(3) * 0.001
            # Position does NOT change!

        # 4. Covariance Propagation
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)  # Enforce numerical symmetry
        np.fill_diagonal(self.P, np.maximum(np.diag(self.P), 1e-6))

    def update_magnetometer(self, mag_body: np.ndarray):
        """Measurement update using tilt-compensated compass heading."""
        roll, pitch = float(self.x[6, 0]), float(self.x[7, 0])
        mag_yaw = calculate_tilt_compensated_heading(mag_body, roll, pitch)

        H = np.zeros((1, 9))
        H[0, 8] = 1.0

        y = np.array([[mag_yaw]]) - (H @ self.x)
        y[0, 0] = (y[0, 0] + np.pi) % (2 * np.pi) - np.pi

        R_yaw = np.array([[0.3]])
        S = H @ self.P @ H.T + R_yaw
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)  # Enforce numerical symmetry
        np.fill_diagonal(self.P, np.maximum(np.diag(self.P), 1e-6))

    def process_sample(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single IMU reading packet (at 10Hz) with mode specification.
        """
        current_ts = packet.get("timestamp", 0.0)
        dt = (current_ts - self.last_timestamp) if self.last_timestamp else 0.10
        dt = max(0.01, min(1.0, dt))
        self.last_timestamp = current_ts

        mode = str(packet.get("mode", "adaptive")).lower().strip()

        accel = np.array([packet.get("ax", 0.0), packet.get("ay", 0.0), packet.get("az", 0.0)])
        gyro = np.array([packet.get("gx", 0.0), packet.get("gy", 0.0), packet.get("gz", 0.0)])
        
        # Prediction step with mode awareness
        self.predict(accel, gyro, dt, mode=mode)

        # Magnetometer measurement update
        if "mx" in packet and "my" in packet and "mz" in packet:
            mag = np.array([packet["mx"], packet["my"], packet["mz"]])
            if np.linalg.norm(mag) > 1e-3:
                self.update_magnetometer(mag)

        return {
            "x": float(self.x[0, 0]),
            "y": float(self.x[1, 0]),
            "z": float(self.x[2, 0]),
            "vx": float(self.x[3, 0]),
            "vy": float(self.x[4, 0]),
            "vz": float(self.x[5, 0]),
            "roll": float(self.x[6, 0]),
            "pitch": float(self.x[7, 0]),
            "yaw": float(self.x[8, 0]),
            "movement_state": self.movement_state,
            "step_count": self.step_count,
            "covariance": self.P.tolist()
        }

    def to_state_dict(self) -> Dict[str, Any]:
        """Export state for caching or serialization."""
        return {
            "x": self.x.tolist(),
            "P": self.P.tolist(),
            "last_timestamp": self.last_timestamp,
            "movement_state": self.movement_state,
            "step_count": self.step_count
        }

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Any]) -> "IMUKalmanFilter":
        """Reconstruct filter instance from cached state dict."""
        kf = cls(
            initial_state=np.array(state_dict["x"]),
            initial_covariance=np.array(state_dict["P"])
        )
        kf.last_timestamp = state_dict.get("last_timestamp")
        kf.movement_state = state_dict.get("movement_state", "REST")
        kf.step_count = state_dict.get("step_count", 0)
        return kf
