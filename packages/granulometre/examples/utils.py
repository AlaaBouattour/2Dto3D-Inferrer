import numpy as np

def create_ellipsoid(a, b, c, R, center=[0, 0, 0], resolution=100):
    """Crée un maillage d'ellipsoïde 3D avec rotation et translation."""
    u = np.linspace(0, 2*np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones_like(u), np.cos(v))
    
    # Appliquer la rotation
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    rotated_points = np.dot(points, R)
    
    # Appliquer la translation
    rotated_points[:, 0] += center[0]
    rotated_points[:, 1] += center[1]
    rotated_points[:, 2] += center[2]
    
    # Reformer les grilles
    x_rot = rotated_points[:, 0].reshape(x.shape)
    y_rot = rotated_points[:, 1].reshape(y.shape)
    z_rot = rotated_points[:, 2].reshape(z.shape)
    
    return x_rot, y_rot, z_rot