###Data filters for sh**** input data
import numpy as np
import math

def calc_lin_coef(point1, point2):
    """
    Calculate y = k * x * b coefficients by 2 points coordinate
    Args:
        point1 : [x, y]
        point2 : [x, y]
    Returns:
        (k, b)
    """
    x1, y1 = point1
    x2, y2 = point2
    if x1 == x2:
        raise ValueError('vertical lines not supported')
    k = - (y2 - y1)/(x1 - x2)
    b = - (x2*y1 - x1*y2)/(x1 - x2)
    return (k, b)
    
def calc_dist(point1, point2, typ=None):
    """
    Calculate distance between 2 points by different methods:
    dist - linear distance between 2 points
    max_coord_dif - maximal difference between points coordinates
    Default method - dist
    Args:
        point1 : [x, y]
        point2 : [x, y]
        typ : method name (str)
    Returns:
        calculated distance (float)
    """
    if typ is None:
        #TODO log default used
        typ = 'dist'
    if typ == 'max_coord_dif':
        dist = max(abs(point1 - point2))
    elif typ == 'dist':
        x1, y1 = point1
        x2, y2 = point2
        dist = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    else:
        raise ValueError(f'Unknown distance calculation method {typ}')
    return dist

def check_point_in_line(filtered_data, point, accuracy):
    """
    check is new point inline last 2 points in filtered data
    Args:
        filtered_data : np.array([x : list of floats, y : list of floats])
        point2 = [x, y]
        accuracy : threshold for check (float)
    Returns:
        bool : True if new point inline last 2 points in filtered data
    """
    x = point[0]
    y = point[1]
    #last 2 filtered points
    x1, y1 = filtered_data[-2]
    x2, y2 = filtered_data[-1]
    if x1 != x2: #if not vertical line
        k, b = calc_lin_coef(filtered_data[-1, :], filtered_data[-2, :])
        lin_y = k*x + b
    else:
        if x == x1: #if vertical and same x
            return True
        else:
            return False
    if calc_dist((x, lin_y), (x, y)) > accuracy: # if current not filtered not on the line within given accuracy
        return False
    else:
        return True

def filter_degr_polyfit(data, min_deg, N_points):
    """
    calculate angle of inclination of line for group of points of specified length, 
    if difference between angle of new group greater than threshold - ignore all point
    after this group including this group (for external border only)
    Args:
        data : np.array([x : list of floats, y : list of floats])
        min_deg : threshold for check
        N_points : size of new group
    Returns:
        filtered np.array
    """
    dataf = data[:N_points]
    for i in range(N_points, math.ceil(data.shape[0]/N_points) * N_points + 1, N_points):
        if i > data.shape[0]:
            i = data.shape[0] - 1
        next_points = data[(i-N_points):i]
        k_new, _ = np.polyfit(next_points[:, 0], next_points[:, 1], 1)
        x = next_points[-1, 0] - next_points[0, 0]
        deg_new = math.degrees(math.atan2(k_new*x, x))
        k_old, _ = np.polyfit(dataf[-N_points:, 0], dataf[-N_points:, 1], 1)
        x = dataf[-1, 0] - dataf[-N_points, 0]
        deg_old = math.degrees(math.atan2(k_old*x, x))
        if abs(deg_new - deg_old) <= min_deg:
            dataf = np.append(dataf, next_points, axis = 0)
        else:
            break
    return dataf
    
def filter_inline_points(data, *args, **kwargs):
    """deletes multiple inline points and appendixes
    Args:
        data : np.array([x : list of floats, y : list of floats])
        accuracy : threshold for check
    Returns:
        filtered np.array
    """
    if not 'accuracy' in kwargs:
        #TODO - log using default accuracy
        accuracy = 1E-9
    else:
        accuracy = kwargs['accuracy']
    data_filt = data[:2]
    for i in range(2, data.shape[0]):
        #current point
        x, y = data[i]
        #skip already added points
        #if x in data_filt[0] and y in data_filt[1]:
        #    continue
        if check_point_in_line(data_filt, (x, y), accuracy):
            data_filt[-1, :] = [x, y]
        else:
            data_filt = np.append(data_filt, data[i:i+1], axis = 0)        
        # filter appendixes
        try:
            #if last point near third last point - delete last two points
            if calc_dist(data_filt[-1], data_filt[-3])<= accuracy:
                data_filt = np.delete(data_filt, (-1, -2), axis = 0)
            #if last point near second last point - delete last point
            if calc_dist(data_filt[-1], data_filt[-2])<= accuracy:
                data_filt = np.delete(data_filt, (-1), axis = 0)
        except IndexError:
            pass
    if data_filt.shape[0] > 1:
        #check first and last
        x, y = data_filt[0]
        if check_point_in_line(data_filt, (x, y), accuracy):
            data_filt = np.delete(data_filt, (-1), axis = 0)
    return data_filt
    
def PolyArea(x,y):
    """Implementation of Shoelace formula from stackoverflow for
    calculation polygon area
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def сut_min_area_close_points(data, min_area, accuracy):
    """
    if polygon area after 2 close points less than threshold - delete smaller polygone
    continues while all polygones will be greater threshold
    Args:
        data : np.array([x : list of floats, y : list of floats])
        min_area : threshold for check (minimal polygone area)
    Returns:
        filtered np.array
    """
    i = 0
    while i < data.shape[0]:
        idx = np.where(np.array([calc_dist(data[i, :], data[j, :]) for j in range(data.shape[0])]) <= accuracy)[0]
        if idx.size > 1:
            after_idx = [x for x in range(idx[0],idx[-1])]
            before_idx = [x for x in range(data.shape[0]) if x not in after_idx]
            after_area = PolyArea(data[after_idx, 0], data[after_idx, 1])
            before_area = PolyArea(data[before_idx, 0], data[before_idx, 1])
            if after_area <= min_area and before_area > min_area:
                data = np.delete(data, after_idx, axis = 0)
                i = 0
            elif after_area > min_area and before_area <= min_area:
                data = np.delete(data, before_idx, axis = 0)
                i = 0
            elif after_area <= min_area and before_area <= min_area:
                data = np.empty([0,2])
                break
        i += 1
    return data

def interpolate_surface_step(d, por, dx, borderc, thinN):
    '''
    interpolate polygone d with polynome of degreee por
    new x coordinates - arange from min x to max with step dx
    in range +- borderc*max(abs(maxx),abs(minx)) - only every
    thinN point used
    '''
    assert borderc < 1, 'thin out coefficient must be less than 1'
    dataf = np.empty([0,2])
    #find first most left point
    i1 = np.where(d[:,0] == np.min(d[:,0]))[0][0] + 1
    #find first most right point
    i2 = np.where(d[:,0] == np.max(d[:,0]))[0][0] + 1
    idx1 = [i for i in range(i1, i2)]
    idx = [idx1, [i for i in range(0, d.shape[0]) if i not in idx1]]
    maxx = max(d[:, 0])
    minx = min(d[:, 0])
    largestx = max((maxx, abs(minx)))
    N = int((largestx - largestx*borderc)/dx)
    for i in (0,1):
        data = d[idx[i], :]
        coefficients = np.polyfit(data[:, 0], data[:, 1], por)
        f = np.poly1d(coefficients)
        #change the order from left->right to right->left
        if i:
            x = np.arange(maxx, minx, -dx)
        else:
            x = np.arange(minx, maxx, dx)
        N2 = x.shape[0] - N
        newidx = np.r_[0:N, N : N2 : thinN, N2:x.shape[0]]
        x = x[newidx]
        polypoints = np.transpose(np.array([x, f(x)]))
        dataf = np.append(dataf, polypoints, axis = 0)
    return dataf

def interpolate_big_vert_breaks_lin(data, Nmax):
    '''
    if given polygone have big vertical brakes (distance between neighboring points
    more than 4 median distance over all neighboring points)
    add new points in the middle between neighboring points by lenear interpolation
    Args:
        data - np.array 2d of floats (polygone coordinates)
        Nmax - int number of interpolations
    Returns:
        np.array 2d of floats (polygone coordinates) with interpolated data
    '''
    newdata = data.copy()
    N = 0
    while N < Nmax:
        N += 1
        tempdata = np.vstack((newdata, newdata[0]))
        dist = np.array([calc_dist(tempdata[i], tempdata[i+1]) for i in range(newdata.shape[0])])
        threshold = np.median(dist) * 4
        idxs = np.where(dist > threshold)[0]
        if idxs.size != 0:
            idx = idxs[0]
            point1 = newdata[idx, :]
            if idx + 1 != newdata.shape[0]:
                point2 = newdata[idx + 1, :]
            else:
                point2 = newdata[0, :]
            if point1[0] != point2[0]:
                k, b = calc_lin_coef(point1, point2)
                x = (point2[0] - point1[0])/2 + point1[0]
                y = k * x + b
            else:
                x = data[idx, 0]
                y = (point2[1] - point1[1])/2 + point1[1]
            newpoint = np.array([x, y], ndmin = 2)
            if idx + 1 != newdata.shape[0]:
                newdata = np.insert(newdata, idx + 1, newpoint, 0)
            else:
                newdata = np.append(newdata, newpoint, 0)
        else:
            break
    return newdata

def interpolate_big_vert_breaks_poly(data, por, N):
    '''
    adds new points by polynomial interpolation to the left and right of polygone
    Args:
        data - np.array 2d of floats (polygone coordinates)
        por - polynome degree
        N - number of points up and down for polynome determination
    Returns:
        np.array 2d of floats (polygone coordinates) with interpolated data
    '''
    newdata = data
    #find first most left point
    i1 = np.where(data[:,0] == np.min(data[:,0]))[0][0] + 1
    #find first most right point
    i2 = np.where(data[:,0] == np.max(data[:,0]))[0][0]
    idxs = [i1, i2]
    for i in idxs:
        idx = [a for a in range(i - N, i + N)]
        coefficients = np.polyfit(data[idx, 1], data[idx, 0], por)
        f = np.poly1d(coefficients)
        y = data[idx, 1]
        j = 0
        threshold = np.mean([abs(y[i+1] - y[i]) for i in range(y.size - 1)])
        while j < len(y) - 1:
            dy = y[j + 1] - y[j]
            if abs(dy) > threshold:
                if y[j + 1] > y[j]:
                    nwp = y[j] + abs(dy) / 2
                else:
                    nwp = y[j] - abs(dy) / 2
                y = np.insert(y, j + 1, nwp)
            else:
                j += 1
        x = f(y)
        for j in range(len(x)):
            if y[j] not in newdata[:, 1]:
                oldidx = np.where(newdata[:, 1] == y[j - 1])[0][0]
                newdata = np.insert(newdata, oldidx + 1, np.array([x[j], y[j]], ndmin = 2), 0)
    return newdata