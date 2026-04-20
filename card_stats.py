# Centralized statistics for troops and spells in the Clash Royale Bot
# REFERENCE: Level 11 (Tournament Standard)

TROOP_STATS = {
    "Knight": {
        "health": 1766,
        "damage": 202,
        "speed": 1.2,
        "attack_range": 1.2,
        "attack_speed": 1.2,
        "cost": 3.0,
        "count": 1
    },
    "Archer": {
        "health": 304,
        "damage": 112,
        "speed": 1.2,
        "attack_range": 5.0,
        "attack_speed": 0.9,
        "cost": 3.0,
        "count": 2
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
        "range": 7.5,
        "speed": 0.8
    }
}

# Observation vector representation for troop types
TROOP_TYPE_MAP = {
    "Knight": 0.1,
    "Archer": 0.2
}
