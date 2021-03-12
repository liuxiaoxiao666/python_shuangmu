import math

def standardRad(t):
    TWOPI = 2.0 * math.pi
    if t >= 0.:
        t = math.fmod(t + math.pi, TWOPI) - math.pi
    else:
        t = math.fmod(t - math.pi, -TWOPI) + math.pi
    return t


def wRo_to_euler(wRo):
    yaw = standardRad(math.atan2(wRo[1, 0], wRo[0, 0]))
    c = math.cos(yaw)
    s = math.sin(yaw)
    pitch = standardRad(math.atan2(-wRo[2, 0], wRo[0, 0] * c + wRo[1, 0] * s))
    roll = standardRad(math.atan2(wRo[0, 2] * s - wRo[1, 2] * c, -wRo[0, 1] * s + wRo[1, 1] * c))
    return yaw, pitch, roll