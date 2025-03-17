import numpy as np
from copy import deepcopy
from pyvoxel.core.Algorithm import in_Polygon
import pivtk
import sys

class Point:
    def __init__(self, coord:np.ndarray, name:str)->None:
        assert len(coord.shape) == 1
        assert 0 < coord.shape[0] < 4

        self.dim = len(coord)
        self.coord = coord

        self.name = name
    
    def __call__(self)->np.ndarray:
        return self.coord


class Line:
    def __init__(self, point1:Point, point2:Point, name:str)->None:
        self.dim = point1.dim
        self.point1 = point1
        self.point2 = point2
        self.name = name

    def get_pointset(self)->np.ndarray:
        return np.stack([self.point1(), self.point2()])
    
    def length(self)->float:
        return np.linalg.norm(self.point2()-self.point1())

class ClosedLoop:
    def __init__(self, lines:list[Line], name:str)->None:
        self.name = name
        self.dim = lines[0].dim

        for i in range(len(lines)-1):
            assert np.all(lines[i].point2() == lines[i+1].point1())
        assert np.all(lines[-1].point2() == lines[0].point1())

        self.lines = deepcopy(lines)
    
    def get_pointset(self, closed:bool = True)->np.ndarray:
        if closed:
            return np.stack([self.lines[0].point1()] + [line.point2() for line in self.lines])
        else:
            return np.stack([line.point1() for line in self.lines])
    
    def get_vtk(self)->pivtk.unstructured_grid:
        points = self.get_pointset(False)
        cells = [{"type" : 3, "indice" : np.array([i, i+1])} for i in range(len(points)-1)]
        cells.append({"type" : 3, "indice" : np.array([len(points)-1, 0])})
        
        return pivtk.unstructured_grid(points, cells)

class Surface:
    def __init__(self, in_loops:list[ClosedLoop], out_loops:list[ClosedLoop], name:str)->None:
        self.name = name
        self.dim = in_loops[0].dim

        self.in_loops = deepcopy(in_loops)
        self.out_loops = deepcopy(out_loops)
        
        self.lines = []
        for in_loop in in_loops:
            for line in in_loop.lines:
                self.lines.append(line)

        for out_loop in out_loops:
            for line in out_loop.lines:
                self.lines.append(line)
    
    def get_pointset(self)->np.ndarray:
        pointset = np.concatenate([loop.get_pointset() for loop in self.in_loops + self.out_loops])
        return np.unique(pointset, axis = 0)
    
    def is_in(self, points:tuple[np.ndarray])->np.ndarray:
        in_points = [in_loop.get_pointset(True) for in_loop in self.in_loops]
        out_points = [out_loop.get_pointset(True) for out_loop in self.out_loops]

        flag = np.zeros((points[0].shape[0], points[0].shape[1])).astype(bool)

        for in_point in in_points:
            flag_in = False
            for point in points:
                flag_in += in_Polygon(point.reshape((-1, 2)), in_point)
        
            flag += flag_in.reshape(flag.shape)
        

        for out_point in out_points:
            flag_out = True
            for point in points:
                flag_out *= in_Polygon(point.reshape((-1, 2)), out_point)
            flag *= (flag_out == False).reshape(flag.shape)
        
        return flag