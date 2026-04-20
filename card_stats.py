# Centralized statistics for troops and spells in the Clash Royale Bot
# REFERENCE: Level 11 (Tournament Standard)

# Speed mapping: 
# Slow: 0.9, Medium: 1.2, Fast: 1.5, Very Fast: 1.8 (Adjusted for better balance)

TROOP_STATS = {
    "Knight": {
        "health": 1766,
        "damage": 202,
        "speed": 1.2,
        "attack_range": 1.2,
        "attack_speed": 1.2,
        "cost": 3.0,
        "count": 1,
        "is_flying": False,
        "targets": "ground"
    },
    "Archer": {
        "health": 304,
        "damage": 107,
        "speed": 1.2,
        "attack_range": 5.0,
        "attack_speed": 0.9,
        "cost": 3.0,
        "count": 2,
        "is_flying": False,
        "targets": "both"
    },
    "Minion": {
        "health": 230,
        "damage": 107,
        "speed": 1.5,
        "attack_range": 1.2,
        "attack_speed": 1.1,
        "cost": 3.0,
        "count": 3,
        "is_flying": True,
        "targets": "both"
    },
    "Giant": {
        "health": 3968,
        "damage": 253,
        "speed": 0.9,
        "attack_range": 1.2,
        "attack_speed": 1.5,
        "cost": 5.0,
        "count": 1,
        "is_flying": False,
        "targets": "building"
    },
    "PEKKA": {
        "health": 3760,
        "damage": 816,
        "speed": 0.9,
        "attack_range": 1.2,
        "attack_speed": 1.8,
        "cost": 7.0,
        "count": 1,
        "is_flying": False,
        "targets": "ground"
    },
    "MegaKnight": {
        "health": 3993,
        "damage": 268,
        "speed": 1.2,
        "attack_range": 1.2,
        "attack_speed": 1.7,
        "cost": 7.0,
        "count": 1,
        "is_flying": False,
        "targets": "ground"
    },
    "HogRider": {
        "health": 1697,
        "damage": 317,
        "speed": 1.8,
        "attack_range": 1.2,
        "attack_speed": 1.6,
        "cost": 4.0,
        "count": 1,
        "is_flying": False,
        "targets": "building"
    },
    "BabyDragon": {
        "health": 1152,
        "damage": 161,
        "speed": 1.5,
        "attack_range": 3.5,
        "attack_speed": 1.5,
        "cost": 4.0,
        "count": 1,
        "is_flying": True,
        "targets": "both",
        "splash_radius": 1.5
    },
    "Wizard": {
        "health": 721,
        "damage": 281,
        "speed": 1.2,
        "attack_range": 5.5,
        "attack_speed": 1.4,
        "cost": 5.0,
        "count": 1,
        "is_flying": False,
        "targets": "both"
    },
    "Musketeer": {
        "health": 721,
        "damage": 217,
        "speed": 1.2,
        "attack_range": 6.0,
        "attack_speed": 1.0,
        "cost": 4.0,
        "count": 1,
        "is_flying": False,
        "targets": "both"
    },
    "Skeletons": {
        "health": 81,
        "damage": 81,
        "speed": 1.5,
        "attack_range": 1.2,
        "attack_speed": 1.1,
        "cost": 1.0,
        "count": 3,
        "is_flying": False,
        "targets": "ground"
    }
}

BUILDING_STATS = {
    "Cannon": {
        "health": 742,
        "damage": 169,
        "attack_range": 5.5,
        "attack_speed": 0.8,
        "cost": 3.0,
        "lifetime": 30.0,
        "targets": "ground"
    }
}

SPELL_STATS = {
    "Fireball": {
        "damage": 688,
        "tower_damage": 207,
        "radius": 2.5,
        "cost": 4.0
    }
}

TOWER_STATS = {
    "king": {
        "health": 4824,
        "damage": 109,
        "range": 7.0,
        "speed": 1.0
    },
    "princess": {
        "health": 3052,
        "damage": 109,
        "range": 8.5,
        "speed": 0.8
    }
}

# Observation vector representation for troop types
TROOP_TYPE_MAP = {
    "Knight": 0.1,
    "Archer": 0.2,
    "Minion": 0.3,
    "Giant": 0.4,
    "PEKKA": 0.5,
    "MegaKnight": 0.6,
    "HogRider": 0.7,
    "BabyDragon": 0.8,
    "Wizard": 0.9,
    "Musketeer": 1.0,
    "Skeletons": 0.11 # or something distinct
}

BUILDING_TYPE_MAP = {
    "Cannon": 0.1
}

