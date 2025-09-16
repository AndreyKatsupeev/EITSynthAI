### Interacting with FEMM
import femm
import numpy as np

def femm_create_problem(units='millimeters', problem_type='planar', freq=50000, precision=1e-8, depth=10):
    """
    create new femm current flow problem
    Args:
        units - "inches", "millimeters", "centimeters", "mils", "meters", "micrometers"
        problem_type - "planar" for a 2-D planar problem, or to "axi" for an axisymmetric problem
        freq - current frequency in Hz
        precision - solver precision (RMS of the residual)
        depth - depth of the problem in the into-the-page direction for 2-D planar problems
    Returns:
    """
    femm.openfemm()
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

def GetMaterialDataFreq(data, Freq):
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
        c = GetMaterialDataFreq(param['cond'], freq)
        p = GetMaterialDataFreq(param['perm'], freq)
        femm.ci_addmaterial(mat, c, c, p, p, 0, 0)

def femm_add_label(coords, material):
    '''
    add label to opened femm problem and
    set its material
    '''
    label = coords[0] + [0.1, 0]
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

def femm_set_injecting(injN, gndN, centers):
    '''
    select electrodes segments by its centers and set 
    its conductors
    '''
    femm.ci_selectsegment(centers[injN, 0], centers[injN, 1])
    femm.ci_setsegmentprop('None', 0, 1, 0, 0,'INJ')
    femm.ci_clearselected()
    femm.ci_selectsegment(centers[gndN, 0], centers[gndN, 1])
    femm.ci_setsegmentprop('None', 0, 1, 0, 0,'GND')
    femm.ci_clearselected()

def femm_save_calculate(fname, not_visible):
    '''
    save problem, analyze and show solution
    '''
    femm.ci_saveas(fname)
    femm.ci_createmesh()
    femm.ci_analyze(not_visible)
    femm.ci_loadsolution() 