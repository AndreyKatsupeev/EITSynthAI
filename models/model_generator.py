#generation model in FEMM by coords
from .filters import *
from .femm_api import *
import os

def load_yolo(filepath):
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
    classes_list = {'0': 'bone', '1':'muscles', '2':'fat', '3':'lung'}
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

def prepare_data(borders, settings):
    accuracy = settings['accuracy']
    Nelec = settings['Nelec']
    Relec = settings['Relec']
    min_area = settings['min_area']
    polydeg = settings['polydeg']
    skinthick = settings['skinthick']
    bordersf = {}
    maxArea = 0
    for tissue, elements in borders.items():
        bordersf[tissue] = []
        idx = 0
        for data in elements:
            dataf = filter_inline_points(data, accuracy = accuracy)
            adataf = сut_min_area_close_points(dataf, min_area, accuracy)
            area = PolyArea(adataf[:, 0], adataf[:, 1])
            if adataf.shape[0] >= 3 and area >=min_area:
                bordersf[tissue].append(adataf)
                if area > maxArea:
                    maxArea = area
                    maxAreaTissue = tissue
                    maxAreaIdx = idx
                idx += 1
    #move to center
    bias = np.mean(bordersf[maxAreaTissue][maxAreaIdx], axis = 0)
    for tissue, elements in bordersf.items():
        for i in range(len(elements)):
            bordersf[tissue][i] = bordersf[tissue][i] - bias
    data = filter_degr_polyfit(bordersf[maxAreaTissue][maxAreaIdx], 90, 3)
    data = interpolate_surface_step(data, polydeg, 2, 0.9, 3)
    data = interpolate_big_vert_breaks_poly(data, 10, 5)
    bordersf[maxAreaTissue][maxAreaIdx] = data
    skin = add_skin(data, skinthick)
    elecs, cents = get_electrodes_coords(skin, Nelec, Relec)
    bordersf['skin'] = [insert_electordes_to_polygone(skin, elecs)]
    return (bordersf, cents)

def get_materials(path):
    '''
    returns dictionary with tissues conductivity and permitivity within frequency
    '''
    materials = {}
    freq = [10, 1e2, 1e3, 1e4, 1e5, 1e6]
    materials['lung'] = {'cond':{}, 'perm':[]}
    #materials['lung']['cond']['inf'] = np.transpose(np.array([freq, [11111,0.0416,0.04335,0.0497,0.06424,0.0647]]))
    #materials['lung']['cond']['def'] = np.transpose(np.array([freq, [11111,0.1387,0.1231,0.1422,0.1821,0.2017]]))
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
                second - rigth(0) and left(1) points, third - x and y
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
    #electrodes edges coordinates
    elecs = []
    cent = np.empty([Nelec, 2])
    for i in range(len(nearidx)):
        pr = data[nearidx[i][0]]
        pl = data[nearidx[i][1]]
        k, b = calc_lin_coef(pr, pl)
        d = calc_dist(pr, pl)
        x0 = pr[0] - (pr[0] - pl[0]) * ds[i] / d
        cent[i] = np.array([x0, k * x0 + b], ndmin = 2)
        dx = (pr[0] - pl[0]) * Relec / d
        temp = np.empty([2, 2])
        for j in range(2):
            a = -1 if j else 1
            temp[j] = np.array([x0 + a * dx, k * (x0 + a * dx) + b], ndmin = 2)
        elecs.append(temp)
    elecs = np.array(elecs)
    return (elecs, cent)

def insert_electordes_to_polygone(polygone, elecs):
    '''
    insert electrodes to skin polygone
    '''
    out = polygone.copy()
    insidx = 0
    for i in range(elecs.shape[0]):
        elecr = max(elecs[i, :, 0])
        elecl = min(elecs[i, :, 0])
        elecu = max(elecs[i, :, 1])
        elecd = min(elecs[i, :, 1])
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
        out = np.insert(out, insidx, elecs[i], axis = 0)
    return out

def test_module():
    """
    :return:
    """
    borders = load_yolo(r'./models/data/test_data.txt')
    settings = {'Nelec' : 16, 'Relec' : 10, 'accuracy' : 0.5,
                'min_area' : 100, 'polydeg' : 5, 'skinthick' : 1}
    bordersf, centers = prepare_data(borders, settings)
    materials = get_materials('./models')
    femm_create_problem()
    femm_add_conductors(0.005)
    femm_add_materials(materials, 50000)
    #for tissue, elements in bordersf.items():
    #    for data in elements:
    #        femm_add_contour(data)
    data = bordersf['skin'][0]
    femm_add_contour(data)
    femm_add_label(data, "skin")
    data = bordersf['fat'][0]
    femm_add_contour(data)
    femm_add_label(data, "muscles")
    femm_set_injecting(0, 1, centers)
    femm_save_calculate("./models/test.fec", 0)

if __name__ == "__main__":
    test_module()