import math
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def point_projected_toserface(points,point):
    """
    find the point is projected from  original point to surface
    :param points: points np array in clude 3 point (0,3,3)
    :param point: point position
    :return: list() is position oc new point
    """

    A,B,C,D = surface_from_three_points(points)
    x0, y0,z0 = point
    k = -(A*x0 + B*y0 + C*z0 + D)/(A**2 + B**2 + C**2)
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    x_new = x0 + A*k
    y_new = y0 + B*k
    z_new = z0 + C*k

    return [x_new,y_new,z_new]

def surface_from_three_points(points):
    """

    :param points: numpy array (0,3,3)
    :return: list in clude 4 parateter [A,B,C,D] Ax + By + Cz + D
    """
    p0, p1, p2 = points
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    ux, uy, uz = u = [x1 - x0, y1 - y0, z1 - z0]
    vx, vy, vz = v = [x2 - x0, y2 - y0, z2 - z0]

    u_cross_v = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]
    point = np.array(p0)
    normal = np.array(u_cross_v)
    d = -point.dot(normal)

    return [normal[0], normal[1], normal[2],d]

def distance_point_to_line(p, a,b):

    """

    :param point:
    :param a: start line
    :param b: end line
    :return: floate distance
    """
    p = np.asarray(p,dtype=np.float32)
    a = np.asarray(a,dtype=np.float32)
    b = np.asarray(b,dtype=np.float32)
    # normalizes tangent vector
    d = np.divide(b-a, np.linalg.norm(b-a))
    #signed parallel distance components
    s = np.dot(a-p,d.T)
    t = np.dot(p-b,d.T)

    # clamped parallel distance
    h = np.maximum.reduce([s,t,0.])
    # perpendicular distance component
    c = np.cross(p-a,d)

    return np.hypot(h, np.linalg.norm(c))

def distance_between_tow_points(point1,point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)
