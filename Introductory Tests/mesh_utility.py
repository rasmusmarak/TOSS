# core stuff
import pickle as pk
import numpy as np
import pathlib

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


def create_mesh():

    path = str(pathlib.Path("Introductory Tests").parent.resolve())

    # Read the input .pk file
    mesh_points, mesh_triangles = read_pk_file(path + "/3dmeshes/churyumov-gerasimenko_lp.pk")

    # Un-normalizing the scale
    #   Conversion factor [to metric meters]: 3126.6064453124995
    mesh_points = mesh_points*float(3126.6064453124995)
    print("Physical dimension along x (UN-normalized): ", max(mesh_points[:,0]) - min(mesh_points[:,0]), "Km")


    tgen = tetgen.TetGen(mesh_points, mesh_triangles)
    nodes, elem = tgen.tetrahedralize()

    return tgen, mesh_points, mesh_triangles

