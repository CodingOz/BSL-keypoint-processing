import math


def vectorMagnitude(v):
    return math.sqrt(v[0]**2 + v[1]**2)


def dotProduct(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


def angleBetweenVectors(v1, v2):
    """Returns angle in radians between two vectors"""
    mag1 = vectorMagnitude(v1)
    mag2 = vectorMagnitude(v2)
    if mag1 == 0 or mag2 == 0:
        return 0
    cos_angle = dotProduct(v1, v2) / (mag1 * mag2)
    cos_angle = max(-1, min(1, cos_angle))
    return math.acos(cos_angle)


def normalizeVector(dx, dy):
    """Returns (direction_x, direction_y, magnitude)"""
    magnitude = (dx**2 + dy**2)**0.5
    if magnitude == 0:
        return (0, 0, 0)
    return (dx / magnitude, dy / magnitude, magnitude)
