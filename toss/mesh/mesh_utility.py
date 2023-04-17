# core stuff
import pickle as pk
import numpy as np
import pathlib
from typing import Union

# meshing
import tetgen


def read_pk_file(filename):
    """
    Reads in a .pk file and returns the vertices and triangles (faces)
    Args:
        filename (str): The filename of the .pk file

    Returns:
        mesh_points, mesh_triangles (tuple): list of mesh points and list of mesh triangles

    Notes:
        Adapted from
        https://github.com/darioizzo/geodesyNets/blob/master/archive/Modelling%20Bennu%20with%20mascons.ipynb

    """
    with open(filename, "rb") as f:
        mesh_points, mesh_triangles = pk.load(f)
    mesh_points = np.array(mesh_points)
    mesh_triangles = np.array(mesh_triangles)
    # Characteristic dimension
    L = max(mesh_points[:, 0]) - min(mesh_points[:, 0])

    # Non dimensional units
    mesh_points = mesh_points / L * 2 * 0.8
    return mesh_points, mesh_triangles


def create_mesh() -> Union[tetgen.pytetgen.TetGen, np.ndarray, np.ndarray, float]:
    """
    Creates a tetrahedralized mesh object representing the celestial body of interest.

    Returns:
        tgen (tetgen.pytetgen.TetGen): Tetgen mesh object of celestial body.
        mesh_points (np.ndarray): Array of all points on mesh.
        mesh_triangles (np.ndarray): Array of all triangles on mesh.
        largest_protuberant (float): Length of largest protuberant mass of the celestial body. (Computed from body centered at origin)
    """

    path = str(pathlib.Path("TOSS").parent.resolve())
    corrected_path = path.split('TOSS', 1)[0]

    # Read the input .pk file
    mesh_points, mesh_triangles = read_pk_file(corrected_path + "/toss/3dmeshes/churyumov-gerasimenko_lp.pk")

    # Un-normalizing the scale
    #   Conversion factor [to metric meters]: 3126.6064453124995
    mesh_points = mesh_points*float(3126.6064453124995)
    #print("Physical dimension along x (UN-normalized): ", max(mesh_points[:,0]) - min(mesh_points[:,0]), "Km")
    largest_protuberant = max(max(mesh_points[:,0]), max(mesh_points[:,1]), max(mesh_points[:,2]))

    tgen = tetgen.TetGen(mesh_points, mesh_triangles)
    _, _ = tgen.tetrahedralize()

    return tgen, mesh_points, mesh_triangles, largest_protuberant


def is_outside(points, mesh_vertices, mesh_triangles):
    """Detects if points are outside a 3D mesh
    Args:
        points ((N,3)) np.array): points to test.
        mesh_vertices ((M,3) np.array): vertices pf the mesh
        mesh_triangles ((M,3) np.array): ids of each triangle
    Returns:
        np.array of boolean values determining whether the points are inside
    """
    counter = np.array([0]*len(points))
    direction = np.array([0, 0, 1])
    if len(points.shape) == 1:
        for t in mesh_triangles:
            counter += ray_triangle_intersect(
                points, direction, mesh_vertices[t[0]], mesh_vertices[t[1]], mesh_vertices[t[2]])   
    else:
        for t in mesh_triangles:
            counter += rays_triangle_intersect(
                points, direction, mesh_vertices[t[0]], mesh_vertices[t[1]], mesh_vertices[t[2]])    
    return (counter % 2) == 0


def ray_triangle_intersect(ray_o, ray_d, v0, v1, v2):
    """Möller–Trumbore intersection algorithm
    Computes whether a ray intersect a triangle
    Args:
        ray_o (3D np.array): origin of the ray.
        ray_d (3D np.array): direction of the ray.
        v0, v1, v2 (3D np.array): triangle vertices
    Returns:
        boolean value if the intersection exist (includes the edges)
    See: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    """
    if ray_o.shape != (3,):
        raise ValueError("Shape f ray_o input should be (3,)")
    edge1 = v1-v0
    edge2 = v2-v0
    h = np.cross(ray_d, edge2)

    a = np.dot(edge1, h)

    if a < 0.000001 and a > -0.000001:
        return False

    f = 1.0 / a
    s = ray_o-v0
    u = np.dot(s, h) * f

    if u < 0 or u > 1:
        return False

    q = np.cross(s, edge1)
    v = np.dot(ray_d, q) * f

    if v < 0 or u + v > 1:
        return False

    t = f * np.dot(edge2, q)

    return t > 0


def rays_triangle_intersect(ray_o, ray_d, v0, v1, v2):
    """Möller–Trumbore intersection algorithm (vectorized)
    Computes whether a ray intersect a triangle
    Args:
        ray_o ((N, 3) np.array): origins for the ray.
        ray_d (3D np.array): direction of the ray.
        v0, v1, v2 (3D np.array): triangle vertices
    Returns:
        boolean value if the intersection exist (includes the edges)
    See: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    """
    if ray_o.shape[1] != 3:
        raise ValueError(
            "Shape f ray_o input should be (N, 3) in this vectorized version")
    edge1 = v1-v0
    edge2 = v2-v0
    h = np.cross(ray_d, edge2)

    a = np.dot(edge1, h)

    if a < 0.000001 and a > -0.000001:
        return [False]*len(ray_o)

    f = 1.0 / a
    s = ray_o-v0
    u = np.dot(s, h) * f

    crit1 = np.logical_not(np.logical_or(u < 0, u > 1))
    q = np.cross(s, edge1)
    v = np.dot(q, ray_d) * f
    crit2 = np.logical_not(np.logical_or(v < 0, u+v > 1))
    t = f * np.dot(q, edge2)
    crit3 = t > 0

    return np.logical_and(np.logical_and(crit1, crit2), crit3)