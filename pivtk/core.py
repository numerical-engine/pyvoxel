import numpy as np
import copy


class version2:
    """Object for VTK file which format is version 2

    Args:
        point_data (list[dict]): Point (or node) data. Dict keys are "name", "values" and "type" where,
            "name" (str): Name of this data
            "values" (np.ndarray): values of each point. This shape is (n, ) in case of scalar field or (n, D) in case of vector field
            "type" (str): "scalar" or "vector"
        cell_data (list[dict]): Cell data which the meanings of each element is similar to point_data
    
    Attributes:
        point_data (list[dict]): Point (or node) data. Dict keys are "name", "values" and "type" where,
            "name" (str): Name of this data
            "values" (np.ndarray): values of each point. This shape is (n, ) in case of scalar field or (n, D) in case of vector field
            "type" (str): "scalar" or "vector"
        cell_data (list[dict]): Cell data which the meanings of each element is similar to point_data
    """
    geom_type = None
    def __init__(self, point_data:list[dict] = [], cell_data:list[dict] = [])->None:
        self.point_data = copy.deepcopy(point_data)
        self.cell_data = copy.deepcopy(cell_data)

    @property
    def dim(self) -> int: raise NotImplementedError
    @property
    def num_points(self) -> int: raise NotImplementedError
    @property
    def num_cells(self) -> int: raise NotImplementedError

    def add_pointdata(self, name:str, values:np.ndarray, theresold:float = 1e-10)->None:
        """Add point data

        Args:
            name (str): Name of point data
            values (np.ndarray): Values of each points
        """
        assert len(values) == self.num_points
        v = copy.deepcopy(values)
        v[np.abs(v) < theresold] = 0.
        if len(values.shape) == 1:
            self.point_data.append({"name" : name, "values" : v, "type" : "scalar"})
        else:
            self.point_data.append({"name" : name, "values" : v, "type" : "vector"})
    
    def add_celldata(self, name:str, values:np.ndarray)->None:
        """Add point data

        Args:
            name (str): Name of point data
            values (np.ndarray): Values of each points
        """
        assert len(values) == self.num_cells
        if len(values.shape) == 1:
            self.cell_data.append({"name" : name, "values" : values, "type" : "scalar"})
        else:
            self.cell_data.append({"name" : name, "values" : values, "type" : "vector"})

    def write_dataset(self, filename:str)->None:
        raise NotImplementedError
    
    def write_scalar(self, name : str, values : np.ndarray, filename : str)->None:
        with open(filename, "a") as file:
            file.write("SCALARS {} float 1\n".format(name))
            file.write("LOOKUP_TABLE default\n")
            for v in values:
                file.write("{}\n".format(v))
    
    def np2str(self, L : np.ndarray)->str:
        s = str(L[0])
        for l in L[1:]:
            s += " " + str(l)
        
        return s + "\n"
    
    def write_vector(self, name : str, values : np.ndarray, filename : str)->None:
        _values = np.concatenate((values, np.zeros((len(values), 1))), axis = 1) if self.dim == 2 else values
        
        with open(filename, "a") as file:
            file.write("VECTORS {} float\n".format(name))
            for v in _values:
                file.write(self.np2str(v))

    def write_pointdata(self, filename : str)->None:
        if not self.point_data: return
        with open(filename, "a") as file:
            file.write("POINT_DATA {}\n".format(self.num_points))
        
        for point_data in self.point_data:
            if point_data["type"] == "scalar":
                self.write_scalar(point_data["name"], point_data["values"], filename)
            else:
                self.write_vector(point_data["name"], point_data["values"], filename)

    def write_celldata(self, filename : str)->None:
        if not self.cell_data: return
        with open(filename, "a") as file:
            file.write("CELL_DATA {}\n".format(self.num_cells))

        for cell_data in self.cell_data:
            if cell_data["type"] == "scalar":
                self.write_scalar(cell_data["name"], cell_data["values"], filename)
            else:
                self.write_vector(cell_data["name"], cell_data["values"], filename)

    def write(self, filename : str)->None:
        """Write VTK file

        Args:
            filename (str): File name
        """
        with open(filename, "w") as file:
            file.write("# vtk DataFile Version 2.0\n")
            file.write("VTKio\n")
            file.write("ASCII\n")
            file.write("DATASET {}\n".format(self.geom_type))
        self.write_dataset(filename)
        self.write_pointdata(filename)
        self.write_celldata(filename)