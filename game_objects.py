import math
import uuid


class Tower:
    def __init__(
        self,
        owner_id,
        position,
        tower_type="princess",
        health=2000,
        attack_damage=50,
        attack_range=7.5,
        attack_speed=0.8,
        targets="both",
    ):
        self.owner_id = owner_id
        self.position = position  # (x, y) tuple
        self.tower_type = tower_type  # "king" or "princess"
        self.health = health
        self.max_health = health
        self.attack_damage = attack_damage
        self.attack_range = attack_range
        self.attack_speed = attack_speed  # Seconds per attack
        self.last_attack_time = 0.0
        self.targets = targets
        self.is_flying = False
        self.target = None

    def is_alive(self):
        return self.health > 0

    def take_damage(self, amount):
        self.health -= amount
        if self.health < 0:
            self.health = 0


class Troop:
    def __init__(
        self,
        owner_id,
        name,
        health,
        damage,
        speed,
        attack_range,
        attack_speed,
        position,
        cost,
        is_flying=False,
        targets="ground",
        splash_radius=0.0,
    ):
        self.id = str(uuid.uuid4())
        self.owner_id = owner_id
        self.name = name
        self.health = health
        self.max_health = health
        self.damage = damage
        self.speed = speed  # Units per second
        self.attack_range = attack_range
        self.attack_speed = attack_speed
        self.position = list(position)  # [x, y] list for mutability
        self.cost = cost
        self.last_attack_time = 0
        self.target = None
        self.is_flying = is_flying
        self.targets = targets
        self.splash_radius = splash_radius

    def is_alive(self):
        return self.health > 0

    def take_damage(self, amount):
        self.health -= amount
        if self.health < 0:
            self.health = 0

    def move(self, dt, target_pos):
        tx, ty = target_pos
        px, py = self.position

        dx = tx - px
        dy = ty - py
        dist = math.hypot(dx, dy)

        if dist == 0:
            return

        move_dist = self.speed * dt
        if move_dist >= dist:
            self.position = [tx, ty]
        else:
            ratio = move_dist / dist
            self.position[0] += dx * ratio
            self.position[1] += dy * ratio

    def can_attack(self, target_pos, current_time):
        dist = math.hypot(
            self.position[0] - target_pos[0], self.position[1] - target_pos[1]
        )
        return (
            dist <= self.attack_range
            and (current_time - self.last_attack_time) >= self.attack_speed
        )


class Building:
    def __init__(
        self,
        owner_id,
        name,
        position,
        health,
        damage,
        attack_range,
        attack_speed,
        lifetime,
        cost,
        targets="ground",
    ):
        self.owner_id = owner_id
        self.name = name
        self.position = position  # (x, y) tuple
        self.health = health
        self.max_health = health
        self.damage = damage
        self.attack_range = attack_range
        self.attack_speed = attack_speed
        self.lifetime = lifetime  # total lifetime in seconds
        self.max_lifetime = lifetime
        self.cost = cost
        self.targets = targets
        self.last_attack_time = 0.0
        self.target = None

    def is_alive(self):
        return self.health > 0

    def take_damage(self, amount):
        self.health -= amount
        if self.health < 0:
            self.health = 0
