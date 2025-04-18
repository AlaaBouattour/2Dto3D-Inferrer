import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, uniform

class Generator:
    """
    Générateur de matrices de rotation 3D aléatoires (dans SO(3))
    selon trois méthodes distinctes :

    - 'qr'         : Tirage par décomposition QR d'une matrice gaussienne
    - 'axis_angle' : Tirage d'un axe uniforme sur la sphère et angle sur [0, pi]
    - 'quaternion' : Tirage d'un quaternion unitaire (4 gaussiennes normalisées)

    Paramètres
    ----------
    method : str
        Méthode à utiliser : 'qr', 'axis_angle' ou 'quaternion'.

    Exemple d'utilisation
    ---------------------
    >>> rot_gen = Generator(method='quaternion')
    >>> R = rot_gen.random_rotation_matrix()
    >>> print(R)  # Matrice de rotation 3x3
    """

    def __init__(self, method='quaternion'):
        self.method = method

    def random_rotation_matrix(self):
        """
        Génère et retourne une matrice de rotation 3x3 aléatoire,
        selon la méthode choisie lors de l'initialisation.
        """
        if self.method == 'qr':
            return self._rotation_qr()
        elif self.method == 'axis_angle':
            return self._rotation_axis_angle()
        elif self.method == 'quaternion':
            return self._rotation_quaternion()
        else:
            raise ValueError(f"Méthode inconnue : {self.method}")

    @staticmethod
    def _rotation_qr():
        """
        1) Méthode QR :
           - On tire une matrice 3x3 dont les coefficients sont gaussiens N(0,1).
           - On effectue une décomposition QR.
           - Q est alors une matrice orthonormée (dans O(3)).
           - On s'assure de l'unicité de la decomposion en forcant R a avoir une diagonale positive.
           - On corrige le signe si det(Q) < 0 pour le ramener dans SO(3)).
           - pour plus de detail voir http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
        """
        H = np.random.randn(3,3)
        Q, R = np.linalg.qr(H)
        Q = Q@np.diag(np.sign(np.diag(R))) # Pour que la decomposition soit unique il faut que diag(R) > 0
        if np.linalg.det(Q) < 0: # On ramene Q dans SO(3) si detQ = -1
            Q[:,0] = -Q[:,0]
        return Q

    @staticmethod
    def _rotation_axis_angle():
        """
        2) Méthode Axe–Angle :
           - On tire un axe uniforme sur la sphère (u).
           - On tire un angle theta dans [0, pi] avec densité theta sur [0, pi].
           - Puis on construit la matrice de rotation via la formule de Rodrigues.
        """
        
        u = np.random.randn(3)
        u /= np.linalg.norm(u) #axe de rotation normalisé

        
        x = np.random.rand()
        theta = 2.0 * np.arcsin(np.sqrt(x))

        K = np.array([[0,     -u[2],  u[1]],
                      [u[2],   0,    -u[0]],
                      [-u[1],  u[0],  0   ]])
        I = np.eye(3)

        R = I + np.sin(theta)*K + (1 - np.cos(theta)) * (K @ K)
        return R

    @staticmethod
    def _rotation_quaternion():
        """
        3) Méthode Quaternions :
           - On tire 4 variables gaussiennes indépendantes.
           - On normalise le vecteur obtenu (sur la sphère S^3).
           - On convertit ce quaternion unitaire en matrice de rotation 3x3.
        """
        # Tirage gaussien 4D
        q = np.random.randn(4)
        q /= np.linalg.norm(q)  # Normalisation

        w, x, y, z = q
        # Conversion quaternion -> rotation 3x3
        # (avec convention w = q0, x = q1, y = q2, z = q3)
        R = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - w*z),         2*(x*z + w*y)],
            [2*(x*y + w*z),           1 - 2*(x**2 + z**2),   2*(y*z - w*x)],
            [2*(x*z - w*y),           2*(y*z + w*x),         1 - 2*(x**2 + y**2)]
        ])
        return R


