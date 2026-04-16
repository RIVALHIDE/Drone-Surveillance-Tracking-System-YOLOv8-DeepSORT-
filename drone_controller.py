"""
Drone Controller — Abstract interface for physical drone follow.

Future: implement subclasses for DJI SDK, MAVLink (ArduPilot/PX4), etc.
Currently provides:
  - DroneController (ABC) — the interface any drone backend must implement
  - SimulatedDroneController — no-op stub for UI testing without hardware
  - Data classes for gimbal/movement commands and follow state
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class DroneStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    ARMED = "armed"
    FOLLOWING = "following"
    RETURNING = "returning"


@dataclass
class GimbalCommand:
    """Gimbal adjustment to keep target centered in frame."""
    pitch_delta_deg: float  # positive = tilt down
    yaw_delta_deg: float    # positive = pan right


@dataclass
class MovementCommand:
    """Velocity-based movement command in body frame."""
    forward_m_s: float      # positive = forward
    right_m_s: float        # positive = right
    up_m_s: float           # positive = ascend
    yaw_rate_deg_s: float   # positive = clockwise


@dataclass
class FollowState:
    """Tracking state sent from vision to controller each frame."""
    target_offset_x: float              # pixels from frame center (+ = right)
    target_offset_y: float              # pixels from frame center (+ = down)
    frame_width: int
    frame_height: int
    target_velocity_px_s: float
    target_class: str
    target_bbox: tuple[int, int, int, int]


class DroneController(ABC):
    """Abstract interface for commanding a physical drone."""

    @abstractmethod
    def connect(self, connection_string: str) -> bool:
        """Establish connection to the drone. Returns True on success."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        ...

    @abstractmethod
    def get_status(self) -> DroneStatus:
        ...

    @abstractmethod
    def get_telemetry(self) -> dict:
        """Return dict with lat, lon, alt, heading, battery_pct, etc."""
        ...

    @abstractmethod
    def send_gimbal_command(self, cmd: GimbalCommand) -> None:
        ...

    @abstractmethod
    def send_movement_command(self, cmd: MovementCommand) -> None:
        ...

    @abstractmethod
    def return_to_home(self) -> None:
        ...

    @abstractmethod
    def emergency_stop(self) -> None:
        """Immediately halt all movement (hover in place)."""
        ...

    def compute_follow_commands(
        self, state: FollowState
    ) -> tuple[GimbalCommand, MovementCommand]:
        """Default PID-style logic: pixel offset -> gimbal + velocity commands.

        Subclasses can override for drone-specific tuning.
        """
        # Normalize offset to [-1, 1]
        norm_x = state.target_offset_x / (state.frame_width / 2)
        norm_y = state.target_offset_y / (state.frame_height / 2)

        gimbal = GimbalCommand(
            pitch_delta_deg=norm_y * 5.0,
            yaw_delta_deg=norm_x * 8.0,
        )

        # Dead zone — don't move for small offsets
        dead = 0.15
        move_x = 0.0 if abs(norm_x) < dead else norm_x
        move_y = 0.0 if abs(norm_y) < dead else norm_y

        movement = MovementCommand(
            forward_m_s=0.0,            # future: depth estimation
            right_m_s=move_x * 2.0,     # max 2 m/s lateral
            up_m_s=-move_y * 1.0,       # max 1 m/s vertical
            yaw_rate_deg_s=move_x * 15.0,
        )

        return gimbal, movement


class SimulatedDroneController(DroneController):
    """No-op controller that logs commands. For UI testing without hardware."""

    def __init__(self):
        self._status = DroneStatus.DISCONNECTED
        self._log: list[tuple] = []

    def connect(self, connection_string: str) -> bool:
        self._status = DroneStatus.CONNECTED
        return True

    def disconnect(self) -> None:
        self._status = DroneStatus.DISCONNECTED

    def get_status(self) -> DroneStatus:
        return self._status

    def get_telemetry(self) -> dict:
        return {"lat": 0, "lon": 0, "alt": 0, "heading": 0, "battery_pct": 100}

    def send_gimbal_command(self, cmd: GimbalCommand) -> None:
        self._log.append(("gimbal", cmd))

    def send_movement_command(self, cmd: MovementCommand) -> None:
        self._log.append(("move", cmd))

    def return_to_home(self) -> None:
        self._status = DroneStatus.RETURNING

    def emergency_stop(self) -> None:
        self._log.append(("ESTOP",))
