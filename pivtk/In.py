import numpy as np
from pivtk import geom
import sys


def read(filename:str):
    with open(filename, "r") as file:
        lines = file.readlines()[3:]
    
    data_type = lines[0].split(" ")[-1][:-1]
    if data_type == "UNSTRUCTURED_GRID":
        return read_UnstructuredGrid(lines[1:])
    else:
        raise NotImplementedError

def read_UnstructuredGrid(lines:list[str]):
    current_idx = 0
    point_num = int(lines[current_idx].split(" ")[1])
    current_idx += 1

    points = []
    for _ in range(point_num):
        points.append(np.array([float(s) for s in lines[current_idx].split(" ")]))
        current_idx += 1
    points = np.stack(points)

    cell_num = int(lines[current_idx].split(" ")[1])
    current_idx += 1

    cells = []
    for _ in range(cell_num):
        indice = np.array([int(l) for l in lines[current_idx].split(" ")[1:]])
        cells.append({"indice":indice})
        current_idx += 1
    current_idx += 1 #skip "CELL_TYPES ** line"
    for i in range(cell_num):
        cells[i]["type"] = int(lines[current_idx])
        current_idx += 1
    
    return geom.unstructured_grid(points, cells)
    ###TODO read point data and cell data


def Graph2UnstructuredGrid(V:np.ndarray, E:np.ndarray)->geom.unstructured_grid:
    cells = []
    for i in range(E.shape[0]-1):
        for j in range(i+1, E.shape[1]):
            if E[i,j] == 1:
                cells.append({"type" : 3, "indice" : np.array([i,j])})
    
    return geom.unstructured_grid(points = V, cells = tuple(cells))