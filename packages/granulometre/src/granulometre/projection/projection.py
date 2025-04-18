import numpy as np
from ..rotation.random_rotation import Generator

def ellipsoid_projection_axes(a, b, c, R=None, method= 'quaternion'):
    """
    Calcule les axes de l'ellipse projetée sur le plan XY après application 
    d'une rotation R à l'ellipsoïde (X'^2/a^2 + Y'^2/b^2 + Z'^2/c^2 = 1).

    Paramètres
    ----------
    a, b, c : floats
        Demi-axes de l'ellipsoïde.
    R : array (3x3) ou None
        Matrice de rotation. Si None, une rotation aléatoire est générée.
        
    
    Calculs
    ----------
    Équation de l’ellipsoïde après rotation :
    X^T Q X = 1, avec X=(X,Y,Z). avec Q = R^T diag(1/a², 1/b², 1/c²) R
    Forme : Q_{11}X² + Q_{22}Y² + Q_{33}Z² + 2Q_{12}XY + 2Q_{13}XZ + 2Q_{23}YZ = 1.

    Pour projeter sur XY, on regarde l'équation en Z :
    Q_{33} Z² + 2(Q_{13}X + Q_{23}Y)Z + (Q_{11}X² + 2Q_{12}XY + Q_{22}Y² - 1) = 0.
    Discriminant D = [2(Q_{13}X+Q_{23}Y)]² - 4 Q_{33}(Q_{11}X²+2Q_{12}XY+Q_{22}Y² -1)
    On pose D=0 pour le contour:
    (Q_{13}X + Q_{23}Y)² - Q_{33}(Q_{11}X² + 2Q_{12}XY + Q_{22}Y² -1) = 0

    Développons pour identifier A,B,C dans AX² + BXY + CY² = 1 :
    (Q_{13}²)X² + 2(Q_{13}Q_{23})XY + (Q_{23}²)Y² 
    - Q_{33}(Q_{11}X² + 2Q_{12}XY + Q_{22}Y² -1) = 0
    
    Regroupons en X², XY, Y² et le terme constant :
    [Q_{13}² - Q_{33}Q_{11}] X² + [2(Q_{13}Q_{23}) - 2Q_{33}Q_{12}] XY + [Q_{23}² - Q_{33}Q_{22}] Y² + Q_{33} = 0
    
    On veut une forme normalisée = 1, donc on divise par Q_{33} :
    => A X² + B XY + C Y² = 1, avec
    (Q_{13}² - Q_{33}Q_{11}) X² + [2(Q_{13}Q_{23}) - 2Q_{33}Q_{12}] XY + (Q_{23}² - Q_{33}Q_{22}) Y² + Q_{33} = 0.
    On doit avoir la forme = 1, donc:
    Divisons toute l'équation par Q_{33}:
    ((Q_{13}² - Q_{33}Q_{11})/Q_{33}) X² + ((2(Q_{13}Q_{23}) - 2Q_{33}Q_{12})/Q_{33}) XY + ((Q_{23}² - Q_{33}Q_{22})/Q_{33}) Y² + 1 = 0
    
    =>[-(Q_{13}² - Q_{33}Q_{11})/Q_{33}] X² + [-(2(Q_{13}Q_{23}) - 2Q_{33}Q_{12})/Q_{33}] XY + [-(Q_{23}² - Q_{33}Q_{22})/Q_{33}] Y² = 1
    
    Retour
    ------
    a_ellipse, b_ellipse : floats
        Demi-axes de l'ellipse projetée.
    angle : float
        Angle d'orientation de l'ellipse (en radians) par rapport à l'axe X.
    """
    if R is None:
        rot_gen = Generator(method=method)
        R = rot_gen.random_rotation_matrix()
    
    
    D = np.diag([1/a**2, 1/b**2, 1/c**2])
    Q = R.T @ D @ R

    A = -(Q[0,2]**2 - Q[2,2]*Q[0,0])/Q[2,2]
    B = -(2*(Q[0,2]*Q[1,2]) - 2*Q[2,2]*Q[0,1])/Q[2,2]
    C = -(Q[1,2]**2 - Q[2,2]*Q[1,1])/Q[2,2]



    M = np.array([[A, B/2],
                  [B/2, C]])
    
    
    vals, vecs = np.linalg.eig(M)
    
    
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]


    a_ellipse = 1/np.sqrt(vals[0])  # demi-grand axe
    b_ellipse = 1/np.sqrt(vals[1])  # demi-petit axe

    vx, vy = vecs[:,0]
    angle = np.arctan2(vy, vx)

    return a_ellipse, b_ellipse, angle