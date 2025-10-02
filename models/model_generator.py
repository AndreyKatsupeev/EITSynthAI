#generation model in FEMM by coords
from .filters import *
from .femm_api import *
import os
import collections
from pyeit.mesh.wrapper import PyEITMesh
from pyeit.mesh.external import place_electrodes_equal_spacing

Settings = collections.namedtuple('Settings',
    ['Nelec', 'Relec', 'accuracy', 'min_area', 'polydeg',
    'skinthick', 'I', 'Freq', 'thin_coeff'])
classes_list = {'0': 'bone', '1':'muscles', '2':'fat', '3':'lung', '4':'skin'}

def load_yolo(filepath, classes_list):
    """
    load tissues borders from yolo dataset into dict, where 
        keys - tissues classes,
        vals - list of lists with coordinates.
    Ignores repeats.
    Args:
        filepath - path to file
    Returns:
        borders = {tissue_type : np.array[[x],[y]]}
    """
    borders = {}
    with open(filepath) as file:
        for line in file:
            x = []
            y = []
            for idx, val in enumerate(line.strip().split(' ')):
                if idx:
                    if idx % 2:
                        x.append(float(val))
                    else:
                        y.append(float(val))
                        try:
                            if (x[-2], y[-2]) == (x[-1], y[-1]):
                                del(x[-1])
                                del(y[-1])
                        except:
                            pass
                else:
                    if val in classes_list:
                        tissue_type = classes_list[val]
                    else:
                        raise ValueError(f'Unknown tissue type {val}')
            if len(x) != len(y):
                raise ValueError(f'len(x) != len(y): {len(x)} != {len(y)}')
            if len(x)>=3:
                if not tissue_type in borders:
                    borders[tissue_type] = []
                borders[tissue_type].append(np.transpose(np.array([x, y])))
    return borders

def load_mesh(fpath, classes_list):
    '''
    load mesh from mesh service, saved in txt
    Args:
        fpath - path to txt
    Returns:
        dict with nodes, elems and conductivity
    '''
    classes_elems_idxs = {}
    for _, class_name in classes_list.items():
        classes_elems_idxs[class_name] = []
    dic = {'NODES' : [], 'TRIANGLES': [], 'CLASS': []}
    key = ''
    i = 0
    with open(fpath,'r') as file:
        for line in file:
            if line.strip():
                s = line.strip().split(' ')
                if '#' in line:
                    key = line.strip()[2:]
                elif key == 'NODES':
                    dic[key].append([float(s[1]), float(s[2])])
                elif key == 'TRIANGLES':
                    dic[key].append([int(s[i]) - 1 for i in range(3)])
                    clasidx = int(float(s[-1]))
                    dic['CLASS'].append(clasidx)
                    class_name = classes_list[str(clasidx)]
                    classes_elems_idxs[class_name].append(i)
                    i += 1
    return {'element' : np.array(dic['TRIANGLES']),
            'node' : np.array(dic['NODES']),
            'cond' : np.array(dic['CLASS']),
            'classes_gr':classes_elems_idxs}

def check_mesh_nodes(meshinfo):
    '''
    check that all nodes used by elements, if not - delete unused nodes
    and change elements nodes indexes
    Args:
        dict with meshinfo
    '''
    all_used_nodes = list(set(meshinfo['element'].ravel()))
    newmeshinfo = dict(meshinfo)
    if len(all_used_nodes) < meshinfo['node'].shape[0]:    
        not_used_nodes = [i for i in range(meshinfo['node'].shape[0]) if i not in all_used_nodes]
        newmeshinfo['node'] = np.delete(meshinfo['node'], not_used_nodes, axis=0)
        node_number_bias = [all_used_nodes[i] - i for i in range(len(all_used_nodes))]
        newmeshinfo['element'] = np.empty([0,3], dtype='i')
        for triangle in meshinfo['element']:
            tmp = []
            for node in triangle:
                if node in all_used_nodes:
                    bias_idx = all_used_nodes.index(node)
                    tmp.append(node - node_number_bias[bias_idx])
            if len(tmp) != 3:
                raise ValueError('triangle with unused noode impossible')
            newmeshinfo['element'] = np.vstack((newmeshinfo['element'],np.array(tmp)))    
    return newmeshinfo

def prepare_mesh(fpath, classes_list):
    badmesh = load_mesh(fpath, classes_list)
    mesh = check_mesh_nodes(badmesh)
    return mesh

def create_pyeit_model(meshinfo, Nelec):
    '''
    create mesh object for pyeit from loaded meshinfo with equal spaced
    elctrodes
    Args:
        dict with shape nodes and elements
        int - number of electrodes
    Returns:
        PyEITMesh object
    '''
    mesh_obj = PyEITMesh(element=meshinfo['element'], node=meshinfo['node'])
    electrode_nodes = place_electrodes_equal_spacing(mesh_obj, 
                                                     n_electrodes=Nelec,
                                                     starting_angle = math.radians(180),
                                                     starting_offset = 0)
    mesh_obj.el_pos = np.array(electrode_nodes)
    return mesh_obj

def prepare_data(borders, settings):
    bordersf = {}
    maxArea = 0
    for tissue, elements in borders.items():
        bordersf[tissue] = {'coords' : [], 'pos' : 'cutted'}
        idx = 0
        for data in elements:
            dataf = filter_inline_points(data, accuracy = settings.accuracy)
            adataf = сut_min_area_close_points(dataf, settings.min_area, settings.accuracy)
            area = PolyArea(adataf[:, 0], adataf[:, 1])
            if adataf.shape[0] >= 3 and area >= settings.min_area:
                bordersf[tissue]['coords'].append(adataf)
                if area > maxArea:
                    maxArea = area
                    maxAreaTissue = tissue
                    maxAreaIdx = idx
                idx += 1
    #move to center
    bias = np.mean(bordersf[maxAreaTissue]['coords'][maxAreaIdx], axis = 0)
    bordersf[maxAreaTissue]['pos'] = 'edge1'
    for tissue, tisinfo in bordersf.items():
        for i in range(len(tisinfo['coords'])):
            bordersf[tissue]['coords'][i] = bordersf[tissue]['coords'][i] - bias
            if not (tissue == maxAreaTissue and i == maxAreaIdx):
                bordersf[tissue]['coords'][i]  = bordersf[tissue]['coords'][i][::settings.thin_coeff]
    data = filter_degr_polyfit(bordersf[maxAreaTissue]['coords'][maxAreaIdx], 90, 3)
    data = interpolate_surface_step(data, settings.polydeg, 2, 0.9, 3)
    data = interpolate_big_vert_breaks_poly(data, 10, 5)
    bordersf[maxAreaTissue]['coords'][maxAreaIdx] = data
    skin = add_skin(data, settings.skinthick)
    elecs = get_electrodes_coords(skin, settings.Nelec, settings.Relec)
    #move centers out of model center at distansce of elec radius
    #for selection right segment in femm preprocessor
    elecs[:, 2, :] = add_skin(elecs[:, 2, :], settings.Relec)
    bordersf['skin'] = {'coords' : [insert_electordes_to_polygone(skin, elecs)],
                        'pos' : 'edge1'}
    return (bordersf, elecs)

def get_materials(path):
    '''
    returns dictionary with tissues conductivity and permitivity within frequency
    '''
    materials = {}
    freq = [10, 1e2, 1e3, 1e4, 1e5, 1e6]
    materials['lung'] = {'cond':{}, 'perm':[], 'infl':[]}
    materials['lung']['infl'] = np.transpose(np.array([freq, [11111,0.0416,0.04335,0.0497,0.06424,0.0647]]))
    materials['lung']['cond'] = np.transpose(np.array([freq, [11111,0.1387,0.1231,0.1422,0.1821,0.2017]]))
    materials['lung']['perm'] = np.transpose(np.array([freq, [3.195e7,5.426e5,1.088e5,30606,11513,1567]]))
    materials['skin'] = {'cond' : np.transpose(np.array([freq, [0.3347,0.365374,0.3817,0.43529,0.566,0.839]])),
                         'perm' : np.transpose(np.array([freq, [1.116e5,55953.3,41437.3,28898.1,14925,2118.79]]))}
    materials['bone'] = {'cond' : np.transpose(np.array([freq, [0.00585,0.00586,0.00587,0.00589,0.006,0.007]])),
                         'perm' : np.transpose(np.array([freq, [40140,3824,892,303,103,30.4]]))}
    for mat in ('muscles', 'fat'):
        materials[mat] = {}
        for param in ('cond', 'perm'):
            data = []
            fpath = os.path.join(path, 'data', f"{mat}_{param[0]}.csv")
            with open(fpath) as file:
                for line in file:
                    s = line.split(',')
                    data.append([float(s[0]), float(s[1])])
            materials[mat][param] = np.array(data)
    return materials

def add_skin(data, width):
    '''
    creates new polygone on distance 'width' from given
    https://math.stackexchange.com/questions/175896
    '''
    cent = np.mean(data, axis = 0)
    skin = np.empty([0,2])
    for point in data:
        dist = calc_dist(cent, point, typ = 'dist')
        t = - width/dist
        xt = (1-t)*point[0] + t*cent[0]
        yt = (1-t)*point[1] + t*cent[1]
        skin = np.vstack([skin, [xt, yt]])
    return skin

def get_electrodes_coords(data, Nelec, Relec):
    '''
    find centeres and edges of flat electrodes
    Args:
        data - 2d np.array of skin polygone
        Nelec - number of electrodes
        Relec - radius of electrodes
    Returns:
        elecs - 3d np. array. first dimension - number of elctrodes, 
                second - rigth(0), left(1) and center(2) points, third - x and y
        cents - 2d array of centres coordinates
    '''
    #array of distances from right point
    ds = []
    #find nearest to zero Ox 
    #points clockwise
    idx = np.where(np.logical_and(data[:,1] < 0, data[:,0] >= 0))[0][-1]
    #calculate distance from first right pointto zero Ox
    k, b = calc_lin_coef(data[idx], data[idx + 1])
    ds.append(calc_dist(data[idx], [0, b]))
    #calculate perimeter
    perim = calc_dist(data[0], data[-1])
    for i in range(data.shape[0] - 1):
        perim += calc_dist(data[i], data[i + 1])
    #calculate distance between centres of elctrodes
    distbetwelec = perim / Nelec
    #idx of points starting from zero and back
    distidx = np.r_[idx:data.shape[0], 0 : idx ]
    #add first value to array of nearest to center of electrode points
    nearidx = [(idx, idx + 1)]
    #current distance from center
    s = - ds[0]
    for i in range(data.shape[0] - 1):
        s += calc_dist(data[distidx[i]], data[distidx[i + 1]])
        #if new distance bigger than calculated
        if s >= distbetwelec:
            s -= distbetwelec
            ds.append(s)
            nearidx.append((distidx[i], distidx[i + 1]))
    #electrodes coordinates [right edge left edge center]
    elecs = []
    for i in range(len(nearidx)):
        pr = data[nearidx[i][0]]
        pl = data[nearidx[i][1]]
        k, b = calc_lin_coef(pr, pl)
        d = calc_dist(pr, pl)
        x0 = pr[0] - (pr[0] - pl[0]) * ds[i] / d
        dx = (pr[0] - pl[0]) * Relec / d
        temp = np.empty([3, 2])
        for j in range(2):
            a = -1 if j else 1
            temp[j] = np.array([x0 + a * dx, k * (x0 + a * dx) + b], ndmin = 2)
        temp[2] = np.array([x0, k * x0 + b], ndmin = 2)#center
        elecs.append(temp)
    elecs = np.array(elecs)
    return elecs

def insert_electordes_to_polygone(polygone, elecs):
    '''
    insert electrodes to skin polygone
    '''
    out = polygone.copy()
    insidx = 0
    for i in range(elecs.shape[0]):
        elecr = max(elecs[i, 0:2, 0])
        elecl = min(elecs[i, 0:2, 0])
        elecu = max(elecs[i, 0:2, 1])
        elecd = min(elecs[i, 0:2, 1])
        xand = np.logical_and(elecl <= out[:, 0], out[:, 0] <= elecr)
        yand = np.logical_and(elecd <= out[:, 1], out[:, 1] <= elecu)
        idx = np.where(np.logical_and(xand, yand))[0]
        if idx.size == 0:
            for j in range(out.shape[0] - 1):
                polyr = max(out[j:j+2, 0])
                polyl = min(out[j:j+2, 0])
                polyu = max(out[j:j+2, 1])
                polyd = min(out[j:j+2, 1])
                #print(polyr, polyl, polyu, polyd)
                if polyl <= elecs[i, 0, 0] <= polyr and polyd <= elecs[i, 0, 1] <= polyu:
                    insidx = j + 1
                    break
            else:
                print(i)
                raise ValueError('electrode not found in polygone')
        else:
            out = np.delete(out, idx, axis = 0)
            insidx = idx[0]
        out = np.insert(out, insidx, elecs[i, 0:2, :], axis = 0)
    return out

def save_model(fname, Nprojections = 0, dirpath = ''):
    """
    if Nprojections != 0 - save currnet opened problem Nprojections times
    with unique names (projection number in name)
    Args:
        fname - problem file name without file extension
        Nprojections (optional) - number of projections for parallel computations
    Returns:
        list of files paths
    """
    fpaths = []
    if not dirpath:
        dirpath = './models/temp/'
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    if Nprojections:
        for i in range(Nprojections):
            fpaths.append(dirpath + fname + str(i) + '.fec')
            femm.ci_saveas(fpaths[-1])
    else:
        fpaths.append(dirpath + fname + '.fec')
        femm.ci_saveas(fpaths[-1])
    return fpaths

def create_femm_model(borders, settings, materials):
    '''
    Open FEMM and create new current flow FEMM problem, add countours, 
    add conductors and materials
    Args:
        borders - dict with raw borders
        settings - named tuple
        materials - dict with materials properties
    '''
    bordersf, elecs = prepare_data(borders, settings)
    femm_prepare_problem()
    femm_add_conductors(settings.I)
    femm_add_materials(materials, settings.Freq)
    for tissue, tisinfo in bordersf.items():
        for data in tisinfo['coords']:
            femm_add_contour(data)
            femm_add_label(data, tissue, tisinfo['pos'])
    return elecs

def test_module():
    """
    :return:
    """
    import matplotlib.pyplot as plt
    from pyeit.visual.plot import create_mesh_plot
    
    borders = load_yolo(r'./models/data/test_data.txt', classes_list)
    materials = get_materials('./models')
    testborders = {'muscles': [borders['fat'][2]],
                   'lung':borders['lung'],
                   'bone':borders['bone']}
    settings = Settings(Nelec = 16, Relec = 10, accuracy = 0.5,
                        min_area = 100, polydeg = 5, skinthick = 1,
                        I = 0.005, Freq = 50000, thin_coeff = 5)
    elecs = create_femm_model(testborders, settings, materials)
    save_model('test', Nprojections = settings.Nelec)
    meshinfo = prepare_mesh('./models/data/tmp.txt', classes_list)
    mesh_obj = create_pyeit_model(meshinfo, settings.Nelec)
    mesh_obj.perm = meshinfo['cond']
    fig, ax = plt.subplots()
    create_mesh_plot(ax, mesh_obj, electrodes = mesh_obj.el_pos,coordinate_labels="radiological")
    plt.savefig('./models/temp/img.png')


if __name__ == "__main__":
    import timeit
    print(timeit.timeit('test_module()', globals=globals(), number = 1))