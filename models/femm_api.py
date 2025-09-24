### Interacting with FEMM
import femm
import numpy as np

def femm_prepare_problem(units='millimeters', problem_type='planar', freq=50000, precision=1e-8, depth=10, fname = ''):
    """
    create new femm current flow problem or open creared (if fname passed to kwargs)
    Args:
        units - "inches", "millimeters", "centimeters", "mils", "meters", "micrometers"
        problem_type - "planar" for a 2-D planar problem, or to "axi" for an axisymmetric problem
        freq - current frequency in Hz
        precision - solver precision (RMS of the residual)
        depth - depth of the problem in the into-the-page direction for 2-D planar problems
        fname - full file name
    Returns:
    """
    femm.openfemm(1)
    #femm.main_minimize()
    if fname:
        femm.opendocument(fname)
    else:
        femm.newdocument(3)  # 3 - current flow problem
        femm.ci_probdef(units, problem_type, freq, precision, depth)

def femm_add_contour(coords):
    """add closed contour by points coordinates
    Args:
        coords - np.array([x : list, y : list])
    Returns:
    """
    x0, y0 = coords[0]
    femm.ci_addnode(x0,y0)
    for i in range(coords.shape[0]-1):
        x1, y1 = coords[i]
        x2, y2 = coords[i+1]
        femm.ci_addnode(x2,y2)
        femm.ci_addsegment(x1,y1,x2,y2)
    femm.ci_addsegment(x2,y2,x0,y0)

def get_material_data_freq(data, Freq):
    '''
    returns interpolated or extrapolated value within given 
    frequency value
    '''
    #if extrapolate
    if Freq < data[0, 0] or Freq > data[-1, 0]:
        if Freq < data[0, 0]:
            #if value on 10 Hz unkmown
            if data[0, 1] == 11111:
                idx1 = 1
                idx2 = 2
            else:
                idx1 = 0
                idx2 = 1
        else:
            idx1 = -2
            idx1 = -1
    else:
        idx1 = np.where(data[:, 0] >= Freq)[0][0]
        idx2 = idx1 + 1
    x1, y1 = data[idx1]
    x2, y2 = data[idx2]
    y = (y2 - y1) * Freq / (x2 - x1) - (y2 - y1) * x1 / (x2 - x1) + y1
    return y

def femm_add_materials(materials, freq):
    '''
    adds materials to opened femm problem from dict of dicts
    Args:
        materials - dict ([tissue][cond or perm] = 2d array cond(f))
        freq - problem frequency
    '''
    for mat, param in materials.items():
        c = get_material_data_freq(param['cond'], freq)
        p = get_material_data_freq(param['perm'], freq)
        femm.ci_addmaterial(mat, c, c, p, p, 0, 0)

def femm_modify_material(name, prop, val):
    '''
    Modify material conductivity, permitivity or dielectric loss tangent
    by name
    '''
    if prop == 'cond':
        idx = [1,2]
    elif prop == 'perm':
        idx = [3,4]
    elif prop == 'tang':
        idx = [5,6]
    else:
        raise ValueError('unknown material property')
    for i in idx:
        femm.ci_modifymaterial(name, i, val)

def femm_add_label(coords, material, pos):
    '''
    add label to opened femm problem and
    set its material
    Args:
        coords - 2d np.array with polygone coords
        material - string with material name
        pos - type of label positioning:
            edge1 - 0.1 mm righter first point
            edgel - 0.1 mm righter leftest point
            edgeu - 0.1 mm below uppest point
            center - polygone center of mass
            cutted - cutted polygone center of mass
    '''
    if pos == 'edge1':
        label = coords[0] + [0.1, 0]
    elif pos == 'edgel':
        label = coords[np.argmin(coords[:, 0])] + [0.1, 0]
    elif pos == 'edgeu':
        label = coords[np.argmax(coords[:, 1])] + [0, -0.1]
    elif pos == 'center':
        label = np.mean(coords, axis = 0)
    elif pos == 'cutted':
        label = np.mean(coords[[-1, 0, 1]], axis = 0)
    else:
        raise ValueError(f'Unknown type {pos} of label positioning')
    femm.ci_addblocklabel(label[0],label[1])
    femm.ci_selectlabel(label[0],label[1])
    femm.ci_setblockprop(material, 0, 0, 0)
    femm.ci_clearselected()

def femm_add_conductors(current):
    '''
    add all conductors for problem
    '''
    femm.ci_addconductorprop('INJ', 0, current, 0)
    femm.ci_addconductorprop('GND', 0, 0, 1)

def femm_set_elec_state(typ, coords):
    '''
    select electrodes segments by its centers and set 
    its conductors
    Args:
        typ - conductor name ('INJ', 'GND', 'None')
        coords - [x, y] for preprocessor selectsegment function
    '''
    femm.ci_selectsegment(coords[0], coords[1])
    femm.ci_setsegmentprop('None', 0, 1, 0, 0, typ)
    femm.ci_clearselected()

def femm_close():
    '''
    close FEMM post and preprocessor
    '''
    femm.ci_close()
    femm.closefemm()