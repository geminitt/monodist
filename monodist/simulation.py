"""Local longitudinal world and obstacle-response policy; independent of UI and perception source.

This is a teaching simulation. Its configurable policy is not calibrated on TEST or validated for driving.
World truth is returned separately from controller observations; the sensor must declare its own source.
"""
from dataclasses import dataclass, field
import math


@dataclass
class Obstacle:
    id: int
    z: float
    speed: float = 0.0
    x: float = 0.0
    lateral_speed: float = 0.0
    length: float = 4.3
    width: float = 1.8
    kind: str = 'Car'
    brake_at: float | None = None
    deceleration: float = 6.0


@dataclass(frozen=True)
class Observation:
    id: int
    depth: float
    closing_speed: float
    x: float = 0.0
    width: float = 1.8
    length: float = 4.3
    kind: str = 'Car'


@dataclass(frozen=True)
class Decision:
    mode: str
    acceleration: float
    target_id: int | None
    ttc: float | None
    reason: str


@dataclass
class World:
    speed: float = 15.0
    position: float = 0.0
    time: float = 0.0
    obstacles: list[Obstacle] = field(default_factory=list)
    dt: float = 0.1
    ego_length: float = 4.3
    ego_width: float = 1.8
    collision: bool = False
    collision_with: int | None = None

    def step(self, acceleration):
        """Control changes ego motion and hence all later relative observations."""
        if not math.isfinite(acceleration) or not -9 <= acceleration <= 3:
            raise ValueError('acceleration must be finite and between -9 and 3 m/s²')
        dt = self.dt
        previous = {obj.id: obj.z - self.position for obj in self.obstacles}
        # Integrate only until zero speed; a hard brake must never make a vehicle reverse.
        moving = min(dt, self.speed / -acceleration) if acceleration < 0 else dt
        self.position += self.speed * moving + 0.5 * acceleration * moving ** 2
        self.speed = max(0.0, self.speed + acceleration * dt)
        for obj in self.obstacles:
            brake = -obj.deceleration if obj.brake_at is not None and self.time >= obj.brake_at else 0.0
            moving = min(dt, obj.speed / -brake) if brake < 0 else dt
            obj.z += obj.speed * moving + 0.5 * brake * moving ** 2
            obj.speed = max(0.0, obj.speed + brake * dt)
            obj.x += obj.lateral_speed * dt
            delta = obj.z - self.position
            same_lane = abs(obj.x) <= (obj.width + self.ego_width) / 2
            contact = abs(delta) <= (obj.length + self.ego_length) / 2
            swept = previous[obj.id] > 0 and delta < 0
            if same_lane and (contact or swept):
                self.collision, self.collision_with = True, obj.id
        self.time += dt

    def snapshot(self):
        return {'time': self.time, 'ego_speed': self.speed, 'ego_position': self.position,
                'collision': self.collision, 'collision_with': self.collision_with,
                'obstacles': [dict(id=obj.id, depth=obj.z-self.position, x=obj.x, speed=obj.speed,
                                   length=obj.length, width=obj.width, kind=obj.kind) for obj in self.obstacles]}


@dataclass
class Controller:
    """Illustrative stopping-distance policy, with explicit tunable simulation parameters."""
    cruise_speed: float = 15.0
    max_brake: float = 6.0
    reaction_s: float = 0.3
    clearance_m: float = 3.0
    corridor_half_width: float = 1.0
    warning_ttc: float = 4.0

    def decide(self, speed, observations):
        candidates = [obj for obj in observations if math.isfinite(obj.depth) and math.isfinite(obj.x)
                      and math.isfinite(obj.closing_speed) and obj.depth > 0
                      and abs(obj.x) <= self.corridor_half_width + obj.width / 2]
        cruise = max(-2.0, min(2.0, (self.cruise_speed-speed) * 0.7))
        if not candidates:
            return Decision('cruise', cruise, None, None, 'No observed obstacle in the forward corridor')
        obj = min(candidates, key=lambda candidate: candidate.depth)
        gap = obj.depth - (4.3 + obj.length) / 2
        closing = max(0.0, obj.closing_speed)
        ttc = obj.depth / closing if closing > 1e-6 else None
        stop = speed ** 2 / (2*self.max_brake) + speed*self.reaction_s + self.clearance_m
        if gap <= self.clearance_m or (closing > 0.1 and gap <= stop):
            return Decision('brake', -self.max_brake, obj.id, ttc, 'Observed gap is within the nominal stopping envelope')
        if ttc is not None and ttc < self.warning_ttc:
            return Decision('warning', min(cruise, 0.0), obj.id, ttc, 'Observed optical-axis TTC is below the warning threshold')
        return Decision('cruise', cruise, obj.id, ttc, 'Observed obstacle is outside the stopping envelope')


def scenario(name):
    """Repeatable nominal teaching cases, not empirical claims about KITTI or real-world safety."""
    if name == 'stationary_obstacle':
        return World(obstacles=[Obstacle(0,42.0)])
    if name == 'lead_brakes':
        return World(obstacles=[Obstacle(0,38.0,speed=12.0,brake_at=1.5)])
    if name == 'adjacent_lane':
        return World(obstacles=[Obstacle(0,35.0,x=3.6)])
    if name == 'late_obstacle':
        return World(obstacles=[Obstacle(0,12.0)])
    raise ValueError(f'unknown scenario: {name}')
