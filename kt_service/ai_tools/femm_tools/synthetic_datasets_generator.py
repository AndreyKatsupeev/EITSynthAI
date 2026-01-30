# Modeling for generating synthetic datasets with FEMM
import femm
import time
from kt_service.ai_tools.femm_tools.model_generator import get_materials, create_pyeit_model, prepare_mesh, classes_list, prepare_mesh_from_femm_generator
from kt_service.ai_tools.femm_tools.femm_api import *
import numpy as np
import numpy.typing as npt
import os
from concurrent.futures import ProcessPoolExecutor
#import pythoncom
import re
import multiprocessing

import pyeit.eit.protocol as protocol
import math
from pyeit.eit.fem import EITForward

def get_spirometry_ref(fname:str)->npt.NDArray:
    '''
    load ventilation reference from file
    Data recorded by custom EIT device at
    Platov South-Russian Polytechnical University (Novocherkassk)
    Args:
        fname:str - path to file
    Returns:
        2d np.array vent(t)
    '''
    data = []
    with open(fname) as file:
        for line in file:
            s = line.split(',')
            data.append([float(s[0]), float(s[1])])
    ventref = np.array(data)
    return ventref

def make_spirometry(N_resp:float, N_points:int)->npt.NDArray:
    '''
    make referent spirometry signal with N_points per one inspiration
    Args:
        N_resp:float - number of inspirations per minute
        N_points:int - number of points per one respiration
    Returns:
        numpy.array([time,spiro])
    '''
    T=60
    t = np.linspace(0,T/N_resp,N_points)
    x = 0.5*np.sin(2*3.1415*1/(60/N_resp)*t+math.radians(270))+0.5
    return np.transpose(np.array([t, x]))

def filt_FFT(typ:str, FPS:float, FC, data:npt.NDArray)->npt.NDArray:
    '''
    make spectrum, zero some frequencys ampliude, rebuild signal from
    spectrum
    Args:
        typ - type of filer (highpass, lowpass, bypass, rejecting)
        FPS - sampling frequency
        FC - cut frequency (tuple if bypass or reject)
        data - input 1d array
    Reurns:
        dataf - output 1d array
    '''
    N = data.shape[0]
    f = np.r_[-N/2 : N/2-1] * FPS / N
    Y = np.fft.fft(data)
    Ys = np.fft.fftshift(Y)
    if typ == 'low':
        idx = np.where(np.logical_or(f <= -FC, f >= FC))
    elif typ == 'high':
        idx = np.where(np.logical_and(f >= -FC, f <= FC))
    elif typ == 'bypass':
        a = np.logical_and(np.logical_or(f >= FC[1], f <= FC[0]), f >=0)
        b = np.logical_and(np.logical_or(f <= -FC[1], f >= FC[0]), f <0)
        idx = np.where(np.logical_or(a, b))
    elif typ == 'reject':
        a = np.logical_and(np.logical_or(f <= FC[1], f >= FC[0]), f >=0)
        b = np.logical_and(np.logical_or(f >= -FC[1], f <= FC[0]), f <0)
        idx = np.where(np.logical_or(a, b))
    else:
        raise ValueError(f'Unknown filter type: {typ}')
    Ys[idx] = 0 + 0j
    Yi = np.fft.ifftshift(Ys)
    dataf = np.fft.ifft(Yi)
    return dataf.real

def spirometry_to_conuctivity(sample:npt.NDArray, Freq:float, materials:dict, spir:npt.NDArray):
    '''
    calculate lungs conductivity change from spirometry data
    Args:
        sample - spirometry part for modeling (2d np.array air vol from time)
        Freq - problem frequency
        material - dict with conductivity(frequency)
        spir - full spirometry sample
    Returns:
        2d np.array conductivity(time) [time, conductivity]
    '''
    if len(spir.shape) == 1:
        sp = spir
    elif len(spir.shape) == 2:
        sp = spir[:,1]
    else:
        raise ValueError('unsupported spirometry data shape')
    def_c = get_material_data_freq(materials['lung']['cond'],Freq)
    inf_c = get_material_data_freq(materials['lung']['infl'],Freq)
    spiramp = max(sp) - min(sp)
    condamp = def_c - inf_c
    condspir = sample.copy()
    condspir[:,1] = (-sample[:, 1] + max(sp))*(condamp/spiramp) + inf_c
    return condspir

def class_to_cond(materials:dict, freq:float, classes_list:dict)->dict:
    '''
    prepare dict {class_name : conductivity}
    Args:
        materials:dict - class_name:np.2darray with conductivity and permitivity from frequency
        freq:float - problem frequency
        clases_list:dict - {class_code:class_name}
    '''
    classes_vals = {}
    for _, name in classes_list.items():
        classes_vals[name] = get_material_data_freq(materials[name]['cond'], freq)
        #classes_vals[name] = complex(get_material_data_freq(data['cond'], freq),
        #                             get_material_data_freq(data['perm'], freq))
    return classes_vals

def meas_voltages_slice(elecs:npt.NDArray)->npt.NDArray:
    '''
    get all voltages on electrodes in one slice
    Args:
        3d array of electrodes edges and centers coordinates
    Returns:
        list of complex Voltages
    '''
    Nelec = elecs.shape[0]
    V = np.empty(Nelec)
    femm.co_seteditmode('contour')
    for i in range(Nelec):
        femm.co_selectpoint(elecs[i,0,0], elecs[i,0,1])
        femm.co_selectpoint(elecs[i,1,0], elecs[i,1,1])
        V[i] = femm.co_lineintegral(3)[0].real
        femm.co_clearcontour()
    dV = abs_to_diff(V, Nelec)
    return dV

def abs_to_diff(v:npt.NDArray, Nelec:int) -> npt.NDArray:
    '''
    calculate neighbours voltage differences from absolute
    voltages. Can be applied to single slice or to full scan
    Current injection voltages must present in input array
    Args:
        v:np.array - absolute voltage
        Nelec:int - number of electrodes
    Returns:
        np.array - voltage differences between neighbour
        electrodes for every projection
    '''
    diff_v = np.empty(v.shape)
    for i in range(v.shape[0]):
        if (i + 1) % Nelec:
            diff_v[i] = v[i] - v[i + 1]
        else:
            diff_v[i] = v[i] - v[i - (Nelec - 1)]
    return diff_v

def calculate_EIT_projection_femm(idx:int, elecs:npt.NDArray)->npt.NDArray:
    '''
    simulate EIT current injection and measurment
    in early opened FEMM problem - seletcs 2 neighbour
    injection electrodes and measures all
    electrodes voltages (Sheffield protocol)
    Args:
        idx - index of projection
        elecs - 3d array with coords of elecs edges and centers
    '''
    Nelec = elecs.shape[0]
    inj = 0 if idx == Nelec - 1 else idx + 1
    femm_set_elec_state('INJ', elecs[inj, 2])
    femm_set_elec_state('GND', elecs[idx, 2])
    not_visible = 1
    femm.ci_analyze(not_visible)
    femm.ci_loadsolution()
    elec_volts = meas_voltages_slice(elecs)
    femm_set_elec_state('None', elecs[inj, 2])
    femm_set_elec_state('None', elecs[idx, 2])
    return elec_volts

def calculate_EIT_slice_femm_fast(fullfpath:str, elecs:npt.NDArray, tissue_props:dict, V:npt.NDArray):
    '''
    open FEMM problem and
    simulate EIT current injection and measurment
    - seletcs all neighbour electrodes
    as current injection and zero voltage and meas all
    electrodes voltages
    '''
    pythoncom.CoInitialize()
    fname = os.path.basename(fullfpath)
    t = re.findall(r'\d+',fname)
    if t:
        idx = int(t[0])
    else:
        raise ValueError(f'no projection number in problem file path ({fullfpath})')
    femm_prepare_problem(fname = fullfpath)
    femm.smartmesh(0)
    Nelec = elecs.shape[0]
    inj = 0 if idx == Nelec - 1 else idx + 1
    femm_set_elec_state('INJ', elecs[inj, 2])
    femm_set_elec_state('GND', elecs[idx, 2])
    femm.ci_createmesh()
    Nelems = V.shape[2]
    for i in range(Nelems):
        for tisue_name, tissue_info in tissue_props.items():
            for tissue_param, vals in tissue_info.items():
                femm_modify_material(tisue_name,tissue_param, vals[i])
        femm.ci_analyze(1)
        femm.ci_loadsolution()
        V[idx,:,i] = meas_voltages_slice(elecs)
        #femm.co_close() #MORE SPEEEED!!11
    femm_set_elec_state('None', elecs[inj, 2])
    femm_set_elec_state('None', elecs[idx, 2])
    femm.closefemm()

def calculate_EIT_projection_pyeit(meshinfo:dict, classes_vals:dict, fwd)->npt.NDArray:
    """
        Compute the EIT voltage projection for a given conductivity distribution.

        The function creates a PyEIT forward model, assigns conductivities to mesh
        elements based on material classes, constructs a standard adjacent
        stimulation/measurement protocol, and solves the forward EIT problem.

        :param meshinfo: dict, mesh description prepared by
                         prepare_mesh_from_femm_generator
        :param classes_vals: dict[str, float], conductivity values per class
        :param Nelec: int, number of electrodes
        :return: np.ndarray, simulated EIT voltage measurements
        """
    cond = meshinfo['cond'].astype(float)
    for class_name, class_elements in meshinfo['classes_gr'].items():
        for class_idx in class_elements:
            cond[class_idx] = float(classes_vals[class_name])
    v = fwd.solve_eit(perm=cond)
    return v

def process_EIT_projection(line, classes_vals, meshinfo, fwd):
    """
        Wrapper function for multiprocessing-based EIT projection computation.

        This function updates lung conductivity for a single time step and
        computes the corresponding EIT projection. Intended to be used with
        multiprocessing.Pool.

        :param line: array-like, one row of conductivity evolution data
        :param classes_vals: dict[str, float], base conductivity values per class
        :param meshinfo: dict, mesh description
        :param N_elec: int, number of electrodes
        :return: np.ndarray, simulated EIT voltage measurements
        """
    classes_vals_local = classes_vals.copy()
    classes_vals_local['lung'] = line[1]
    return calculate_EIT_projection_pyeit(meshinfo, classes_vals_local, fwd)

def simulate_EIT_femm(fpath:list[str], elecs:npt.NDArray, tissue_props:dict, V:npt.NDArray)->npt.NDArray:
    '''
    calculate each EIT projection in different processes
    Args:
        fpath - list of full path to problems
        elecs - 3d array with elec info
        tissue_props - dict with tissues properties for monitoring modeling
    Returns:
        2d array with voltages for every point in tissues properties
    '''
    Nelec = elecs.shape[0]
    Nelems = 0
    for tisue_name, tissue_info in tissue_props.items():
        for tissue_param, vals in tissue_info.items():
            if not Nelems:
                Nelems = len(vals)
            else:
                if Nelems != len(vals):
                    raise ValueError((f'bad len of {tissue_param} values for'
                                      f'{tisue_name}'))
    volts = np.zeros([Nelec, Nelec, Nelems])
    with ProcessPoolExecutor(max_workers = Nelec) as executor:
        [executor.submit(calculate_EIT_slice_femm_fast, fpath[i], elecs, tissue_props, V) for i in range(Nelec)]
    volts = np.reshape(volts,(Nelec ** 2, Nelems))
    return volts

def simulate_EIT_monitoring(fpath:list[str], condspir:npt.NDArray, elecs:npt.NDArray)->npt.NDArray:
    '''
    simulate EIT monitoring with changing lungs conductivity
    in time
    Args:
        fpath - list of problems full path
        condspir - 2d np array with change of lungs conductivity in time
    '''
    Nelec = elecs.shape[0]
    V = np.zeros([Nelec, Nelec, condspir.shape[0]])
    tissue_props = {'lung' : {'cond' : condspir[:, 1]}}
    V = simulate_EIT_femm(fpath, elecs, tissue_props, V)
    return V

def simulate_EIT_monitoring_pyeit(meshdata, N_elec=16, N_spir=12, N_points=100, N_minutes=1, isSaveToFile=False, filename=None, materials_location="../femm_tools"):
    """
    Simulate EIT monitoring with time-varying lung conductivity.

    The function generates a spirometry-based conductivity evolution,
    computes EIT projections for each time step using multiprocessing,
    and optionally saves the results to a text file.

    :param meshdata: dict, FEMM-generated mesh data
    :param N_elec: int, number of electrodes
    :param N_spir: int, number of spirometry cycles
    :param N_points: int, number of points per cycle
    :param N_minutes: int, total duration in minutes
    :param isSaveToFile: bool, whether to save results to file
    :param filename: str or None, output file path
    :param materials_location: str or None, path to materials location
    :return: tuple:
        - v: list[np.ndarray], EIT voltage projections over time
        - dataset_generation_time: float, total simulation time in seconds
    """
    t1 = time.time()
    meshinfo = prepare_mesh_from_femm_generator(meshdata)
    materials = get_materials(materials_location)
    Freq = 50000
    dataf = make_spirometry(N_spir, N_points)
    spir = dataf[:, 1] * 1.5
    condspir = spirometry_to_conuctivity(dataf, Freq, materials, spir)
    classes_vals = class_to_cond(materials, Freq, classes_list)
    mesh_obj = create_pyeit_model(meshinfo, N_elec)
    protocol_obj = protocol.create(N_elec, dist_exc=1, step_meas=1, parser_meas="std")
    fwd = EITForward(mesh_obj, protocol_obj)
    task_args = [(line, classes_vals, meshinfo, fwd) for line in condspir]
    with multiprocessing.Pool() as pool:
        v = pool.starmap(process_EIT_projection, task_args)
    if isSaveToFile is True and filename is not None:
        with open(filename, "w") as f:
            for i in range(N_spir*N_minutes):
                for arr in v:
                    arr = np.asarray(arr).ravel()
                    np.savetxt(f, arr[None, :])
    dataset_generation_time = time.time() - t1
    return v, dataset_generation_time

def test_module():
    import time
    elecs = [[[ 9.97650385e+00, -1.17559591e+02],
              [-9.97650385e+00, -1.18929804e+02],
              [ 1.15395082e-01, -1.28244031e+02]],
             [[-5.40184444e+01, -1.24913068e+02],
              [-7.39642982e+01, -1.26383754e+02],
              [-6.81704177e+01, -1.34733320e+02]],
             [[-1.24842855e+02, -1.14597243e+02],
              [-1.40980978e+02, -1.02783649e+02],
              [-1.40326752e+02, -1.15400157e+02]],
             [[-1.66113957e+02, -7.08158160e+01],
              [-1.75907660e+02, -5.33778293e+01],
              [-1.80208490e+02, -6.60214448e+01]],
             [[-1.92151392e+02, -9.15145286e+00],
              [-1.87491895e+02,  1.02982024e+01],
              [-1.99808497e+02,  6.07779113e-02]],
             [[-1.70590247e+02,  5.71760559e+01],
              [-1.62227162e+02,  7.53435764e+01],
              [-1.75877399e+02,  6.94759984e+01]],
             [[-1.38500936e+02,  1.12466346e+02],
              [-1.23955330e+02,  1.26193157e+02],
              [-1.38882178e+02,  1.25765246e+02]],
             [[-8.04195243e+01,  1.46471619e+02],
              [-6.07331180e+01,  1.49999424e+02],
              [-7.50534301e+01,  1.57177305e+02]],
             [[-1.42012137e+01,  1.54770621e+02],
              [ 5.73767222e+00,  1.56332936e+02],
              [-4.42091798e+00,  1.65549989e+02]],
             [[ 5.20179128e+01,  1.56935270e+02],
              [ 7.17318237e+01,  1.53564555e+02],
              [ 6.58785956e+01,  1.64413436e+02]],
             [[ 1.16064755e+02,  1.31726519e+02],
              [ 1.30882324e+02,  1.18293736e+02],
              [ 1.30838425e+02,  1.31774627e+02]],
             [[ 1.61707080e+02,  7.70796851e+01],
              [ 1.71815107e+02,  5.98219970e+01],
              [ 1.76211435e+02,  7.17205578e+01]],
             [[ 1.89725565e+02,  2.68899821e+01],
              [ 1.90555706e+02,  6.90721790e+00],
              [ 2.00134604e+02,  1.72458598e+01]],
             [[ 1.74712581e+02, -4.02488232e+01],
              [ 1.66427577e+02, -5.84520841e+01],
              [ 1.80019363e+02, -5.26232257e+01]],
             [[ 1.33461374e+02, -1.02958664e+02],
              [ 1.16817517e+02, -1.14048390e+02],
              [ 1.32433797e+02, -1.15344026e+02]],
             [[ 7.84894063e+01, -1.19569970e+02],
              [ 5.85045450e+01, -1.18791948e+02],
              [ 7.32533191e+01, -1.27977390e+02]]]
    elecs = np.array(elecs)
    fpath = ['./models/temp/test' + str(i) + '.fec' for i in range(16)]
    #t = []
    #for i in range(1,15):
    #    start = time.time()
    #    tissue_props = {'lung' : {'cond' : [0.03]*i}}
    #    v = simulate_EIT_femm(fpath, elecs, tissue_props)
     #   #v = calculate_EIT_slice_femm_fast(fpath[0], elecs, tissue_props)
     #   end = time.time()
    #    t.append(end - start)
    #    print(f't({i}): {t[-1]}')
    #for i in range(1,len(t)):
    #    print(f'dt({i}): {t[i] - t[-1]}')
    materials = get_materials('./models')
    Freq = 50000
    #spir = get_spirometry_ref('./models/data/vent.csv')
    #dT = spir[1,0]
    #t1 = 1.5;
    #t2 = 11.5;
    #idx = np.where(np.logical_and(t1 <= spir[:,0], spir[:,0] <= t2))[0]
    #sample = spir[idx[0]:idx[-1]:12]
    #dataf = sample.copy()
    #dataf[:, 1] = filt_FFT('bypass', 1/dT, (0.05, 0.5), sample[:,1] - np.mean(sample[:,1]))
    Nspir = 12
    dataf = make_spirometry(Nspir, 15)
    spir = dataf[:,1]*1.5
    condspir = spirometry_to_conuctivity(dataf, Freq, materials, spir)
    v_spir = simulate_EIT_monitoring(fpath, condspir, elecs)
    v = np.tile(v_spir, Nspir)
    print(v.shape)
    #meshinfo = prepare_mesh('./models/data/tmp.txt', classes_list)
    #classes_vals = class_to_cond(materials, Freq, classes_list)
    #v = calculate_EIT_projection_pyeit(meshinfo, classes_vals, elecs.shape[0])

if __name__ == "__main__":
    #test_module()
    import timeit
    print(timeit.timeit('test_module()', globals=globals(), number = 1))