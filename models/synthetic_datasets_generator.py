# Model for generating synthetic datasets without FEMM

from datetime import datetime
import femm
import numpy as np


def load_yolo(filepath):
    """
    load tissues borders from yolo dataset into dict, where 
        keys - tissues classes,
        vals - list of lists with coordinates.
    Ignores repeats.
    Args:
        filepath - path to file
    Returns:
        borders = {tissue_type : [[x, y], ..., [x, y]]}
    """
    borders = {}
    with open(filepath) as file:
        for line in file:
            x = []
            y = []
            for idx, val in enumerate(line.strip().split(' ')):
                if idx:
                    if idx % 2:
                        x.append(int(val))
                    else:
                        y.append(int(val))
                        try:
                            if (x[-2], y[-2]) == (x[-1], y[-1]):
                                del(x[-1])
                                del(y[-1])
                        except:
                            pass
                else:
                    tissue_type = val
            if len(x) != len(y):
                raise ValueError(f'len(x) != len(y): {len(x)} != {len(y)}')
            if len(x)>=3:
                if not tissue_type in borders:
                    borders[tissue_type] = []
                borders[tissue_type].append([x, y])
    return borders


def femm_create_problem(units = 'millimeters', problemtype = 'planar', freq = 50000, precision = 1e-8, depth = 10):
    '''
    create new femm current flow problem
    Args:
        units - "inches", "millimeters", "centimeters", "mils", "meters", "micrometers"
        problemtype - "planar" for a 2-D planar problem, or to "axi" for an axisymmetric problem
        freq - current frequency in Hz
        precision - solver precission (RMS of the residual)
        depth - depth of the problem in the into-the-page direction for 2-D planar problems
    Returns:
    '''
    femm.openfemm()
    femm.newdocument(3)#3 - current flow problem
    femm.ci_probdef(units, problemtype, freq, precision, depth)

def femm_add_contour(coords_list):
    '''add closed countour by points coordinates
    Args:
        coords - [x : list, y : list]
    Returns:
        
    TODO: if multiple points lie on the same line - 
    ignore all points ignore all points except the first and last
    '''
    coords = np.transpose(np.array([coords_list[0],coords_list[1]]))
    x0 = coords[0, 0]
    y0 = coords[0, 1]
    for i in range(0,len(coords)):
        x1 = coords[i-1, 0]
        y1 = coords[i-1, 1]
        x2 = coords[i, 0]
        y2 = coords[i, 1]
        femm.ci_addnode(x2, y2)
        femm.ci_addsegment(x1, y1, x2, y2)
    femm.ci_addsegment(x0, y0, x2, y2)


def test():
    ''''''
    start = datetime.now()
    print(f'Started at {start}')
    borders = load_yolo(r'./data/IOP_870/IOP_870.txt')
    femm_create_problem()
    femm_add_contour(borders['0'][0])
    print(f'Elapsed: {datetime.now() - start}')