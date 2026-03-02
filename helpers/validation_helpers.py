import math

def vector_magnitude(v):
    return math.sqrt(v[0]**2 + v[1]**2)

def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

def angle_between_vectors(v1, v2):
    """Returns angle in radians between two vectors"""
    mag1 = vector_magnitude(v1)
    mag2 = vector_magnitude(v2)
    if mag1 == 0 or mag2 == 0:
        return 0
    cos_angle = dot_product(v1, v2) / (mag1 * mag2)
    cos_angle = max(-1, min(1, cos_angle))
    return math.acos(cos_angle)

def normalize_vector(dx, dy):
    """Returns (direction_x, direction_y, magnitude)"""
    magnitude = (dx**2 + dy**2)**0.5
    if magnitude == 0:
        return (0, 0, 0)
    return (dx / magnitude, dy / magnitude, magnitude)

