import numpy as np
from copy import deepcopy
from scipy.optimize import minimize

__all__ = ['circular_mean', 'angle_in_range', 'rotation_matrix', 'jet_distance', 'jet_distance_rotate', 'JRM']

def circular_mean(angles, weights=None):
    """Compute the circular mean for a set of angles.

    Parameters
    ----------
    angles : array-like
        Angles in radians.
    weights : array-like, optional
        Weights for each angle. If None, all angles are assumed to have
        equal weight.

    Returns
    -------
    mean_angle : float
        Circular mean of the input angles.

    Notes
    -----
    The circular mean is computed using the formula given in [1]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mean_of_circular_quantities

    """
    if weights is None:
        weights = np.ones_like(angles)
    else:
        weights = np.asarray(weights)

    x = np.sum(weights * np.cos(angles))
    y = np.sum(weights * np.sin(angles))

    return np.arctan2(y, x)  


def angle_in_range(angle):
    """Return the angle in the range [-pi, pi].
    This range is preferred to center the jets around (0,0).

    Parameters
    ----------
    angle : float
        Angle in radians.

    Returns
    -------
    angle : float
        Angle in radians in the range [-pi, pi].

    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rotation_matrix(angle):
    """Return the rotation matrix for the given angle.

    Parameters
    ----------
    angle : float
        Angle in radians.

    Returns
    -------
    rot_matrix : array-like
        Rotation matrix.

    """
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def jet_distance(jet, jet_symm, jet_radius, kappa=1, beta=1):
    """Compute the distance between 'jet' and 'jet_symm.'
    Only the pT of 'jet' is used to weigh the distances.

    Parameters
    ----------
    jet : array-like
        Jet as an array with columns (pt, eta, phi).
    jet_symm : array-like
        Reference jet as an array with columns (pt, eta, phi).
    jet_radius : float
        Radius of the jet.
    kappa : float, default=1
        Power of the pT weighting.
    beta : float, default=1
        Power of the distance weighting.

    Returns
    -------
    distance : float
        Distance between the two jets.

    """
    # Find the nearest neighbors between jet and jet_symm
    # For must events brute force should be fine 
    # For a more efficient nn search, use scipy.spatial.cKDTree
    dEta = jet[:, 1].reshape(-1, 1) - jet_symm[:, 1].reshape(-1, 1).T
    dPhi = angle_in_range(jet[:, 2].reshape(-1, 1) - jet_symm[:, 2].reshape(-1, 1).T)
    distance = np.sqrt(dEta**2 + dPhi**2) / jet_radius
    mindist = np.min(distance, axis=1)**beta
    return np.sum(jet[:, 0]**kappa * mindist)


def jet_distance_rotate(theta, jet, jet_symm, jet_radius, kappa=1, beta=1):
    """Compute the distance between 'jet' and 'jet_symm' rotated by 'theta'.
    See jet_distance for more details.
    """
    eta0, phi0 = jet_symm[:, 1], jet_symm[:, 2]
    jet_symm_rotated = deepcopy(jet_symm)
    jet_symm_rotated[:, 1] = eta0*np.cos(theta) - phi0*np.sin(theta)
    jet_symm_rotated[:, 2] = eta0*np.sin(theta) + phi0*np.cos(theta)
    return jet_distance(jet, jet_symm_rotated, jet_radius, kappa=kappa, beta=beta)


def JRM(jet, n, kappa=1, beta=1, jet_radius=1, optimizer='L-BFGS-B', opt_restarts=5, return_angle=False):
    """Jet Rotational Metric (JRM) computation.

    Parameters
    ----------
    jet : array-like
        Jet as an array with columns (pt, eta, phi).
    n : int
        n-foldedness of the rotational symmetry of the reference jet.
    kappa : float, default=1
        Power of the pT weighting.
    beta : float, default=1
        Power of the distance weighting.
    jet_radius : float, default=1
        Radius of the jet.
    optimizer : str, 'L-BFGS-B' or 'grid', default='L-BFGS-B'
        Optimizer to use for the minimization. See scipy.optimize.minimize
        for details.
    opt_restarts : int, default=5
        Number of optimizations to run with random initial angles.
        Only used if optimizer='L-BFGS-B'.
    return_angle : bool, default=False
        If True, return the optimal angle as well.

    Returns
    -------
    JRM : float
        JRM for the jet.

    """
    jet = deepcopy(jet)
    # Remove particles with pT=0
    jet = jet[jet[:, 0] > 0]
    # Center the jet
    jet[:, 1] -= np.average(jet[:, 1], weights=jet[:, 0])
    jet[:, 2] = angle_in_range(jet[:, 2] - circular_mean(jet[:, 2], weights=jet[:, 0]))
    
    # Calculate the mean constituent radius 
    consts_radii = np.sqrt(jet[:, 1]**2 + jet[:, 2]**2)
    mean_radius = np.average(consts_radii, weights=jet[:, 0])

    # Construct the reference jet
    jet_symm = np.zeros((n, 3))
    jet_symm[:, 0] = np.sum(jet[:, 0])/n
    jet_symm[:, 1] = mean_radius
    rotation_matrices = np.array([rotation_matrix(2*np.pi*i/n) for i in range(n)])
    jet_symm[:, 1:] = np.einsum('ijk,ij->ik', rotation_matrices, jet_symm[:, 1:], optimize=True)

    # Rotate the reference jet to find the optimal angle
    mindist = 999999999 
    opt_angle = 0.
    if optimizer == 'grid':
        # Grid optimization with ~1 degree resolution.
        # Lower resolution could also be used with okay results.  
        ngrid = int(360./n)
        for i in range(ngrid): 
            theta = 2*np.pi/n * (i/ngrid)
            dist = jet_distance_rotate(theta, jet, jet_symm, jet_radius, kappa=kappa, beta=beta)
            if dist < mindist:
                mindist = dist
                opt_angle = theta
    elif optimizer == 'L-BFGS-B':
        # L-BFGS-B optimization using scipy.optimize.minimize
        # Run multiple optimizations with random initial angles and keep the best result
        for i in range(opt_restarts):
            theta = np.random.uniform(0, 2*np.pi/n)
            res = minimize(jet_distance_rotate, theta, 
                           method='L-BFGS-B',
                           args=(jet, jet_symm, jet_radius, kappa, beta),
                           bounds=[(0, 2*np.pi/n)])
            if res.fun < mindist:
                mindist = res.fun
                opt_angle = res.x[0]
    else:
        raise ValueError("Unknown optimizer '{}'.".format(optimizer))

    if return_angle:
        return mindist, opt_angle
    return mindist
