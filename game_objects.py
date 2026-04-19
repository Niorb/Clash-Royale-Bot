import uuid

class Tower:
    def __init__(self, owner_id, position, health=2000, attack_damage=50, attack_range=7.5, attack_speed=0.8):
        self.owner_id = owner_id
        self.position = position  # Scalar position (0 to 32)
        self.health = health
        self.max_health = health
        self.attack_damage = attack_damage
        self.attack_range = attack_range
        self.attack_speed = attack_speed  # Seconds per attack
        self.last_attack_time = 0

    def is_alive(self):
        return self.health > 0

    def take_damage(self, amount):
        self.health -= amount
        if self.health < 0:
            self.health = 0

class Troop:
    def __init__(self, owner_id, name, health, damage, speed, attack_range, attack_speed, position, cost):
        self.id = str(uuid.uuid4())
        self.owner_id = owner_id
        self.name = name
        self.health = health
        self.max_health = health
        self.damage = damage
        self.speed = speed  # Units per second
        self.attack_range = attack_range
        self.attack_speed = attack_speed
        self.position = position
        self.cost = cost
        self.last_attack_time = 0
        self.target = None

    def is_alive(self):
        return self.health > 0

    def take_damage(self, amount):
        self.health -= amount
        if self.health < 0:
            self.health = 0

    def move(self, dt, target_pos):
        direction = 1 if target_pos > self.position else -1
        self.position += direction * self.speed * dt
        # Snap to target if we overshot
        if (direction == 1 and self.position > target_pos) or (direction == -1 and self.position < target_pos):
            self.position = target_pos

    def can_attack(self, target_pos, current_time):
        dist = abs(self.position - target_pos)
        return dist <= self.attack_range and (current_time - self.last_attack_time) >= self.attack_speed
