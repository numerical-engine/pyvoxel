import numpy as np
from copy import deepcopy

from pyvoxel.core.Geometry import Surface
from pyvoxel.core.Algorithm import get_BoundingBox, in_Polygon

import sys


class Field:
    def __init__(self, dx:float, surface_list:list[Surface], surface_props:dict[dict], bc_line_props:dict[dict])->None:
        self.dx = dx
        self.surface_list = deepcopy(surface_list)
        self.surface_props = deepcopy(surface_props)
        self.bc_line_props = deepcopy(bc_line_props)
        self.bc_line_list = []
        for surface in surface_list:
            for line in surface.lines:
                if line.name in set(bc_line_props.keys()):
                    self.bc_line_list.append(line)

        self.cells = self.select_Cells()

    
    def select_Cells(self)->dict[dict]:
        pointset = np.concatenate([surface.get_pointset() for surface in self.surface_list])
        coordmin, coordmax = get_BoundingBox(pointset, 3.*self.dx)

        x_num = int(np.ceil((coordmax[0] - coordmin[0])/self.dx))
        y_num  = int(np.ceil((coordmax[1] - coordmin[1])/self.dx))

        x_center = coordmin[0] + self.dx/2. + self.dx*np.arange(x_num)
        y_center = coordmin[1] + self.dx/2. + self.dx*np.arange(y_num)

        yy, xx = np.meshgrid(y_center, x_center)
        cell_centers = np.stack((xx, yy), axis = -1)
        cell_tags = (np.arange(cell_centers.shape[0]*cell_centers.shape[1]).astype(int)).reshape((cell_centers.shape[0], cell_centers.shape[1]))
        in_flags = [surface.is_in(self.get_Vertice(cell_centers)) for surface in self.surface_list]


        cells = {}
        for i in range(1, cell_centers.shape[0]-1):
            for j in range(1, cell_centers.shape[1]-1):
                for surface, flag in zip(self.surface_list, in_flags):
                    if flag[i,j] == False: continue

                    etag = None if flag[i+1,j] == False else cell_tags[i+1,j]
                    wtag = None if flag[i-1,j] == False else cell_tags[i-1,j]
                    ntag = None if flag[i,j+1] == False else cell_tags[i,j+1]
                    stag = None if flag[i,j-1] == False else cell_tags[i,j-1]

                    cells[cell_tags[i,j]] = {"prop" : self.surface_props[surface.name], "E" : etag, "W" : wtag, "N" : ntag, "S" : stag}

        return cells

    def get_Vertice(self, cell_center:np.ndarray)->tuple[np.ndarray]:
        point0 = cell_center + 0.5*self.dx*np.array([-1., -1.])
        point1 = cell_center + 0.5*self.dx*np.array([1., -1.])
        point2 = cell_center + 0.5*self.dx*np.array([1., 1.])
        point3 = cell_center + 0.5*self.dx*np.array([-1., 1.])

        return point0, point1, point2, point3