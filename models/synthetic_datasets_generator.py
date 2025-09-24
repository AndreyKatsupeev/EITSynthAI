# Modeling for generating synthetic datasets with FEMM
import femm
from .model_generator import create_model, get_materials
from .femm_api import *
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import pythoncom
import re

def get_spirometry_ref(fname):
    '''
    load ventilation reference from file
    Data recorded by custom EIT device at 
    Platov South-Russian Polytechnical University (Novocherkassk)
    Args:
        fname - path to file
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

def filt_FFT(typ, FPS, FC, data):
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

def spirometry_to_conuctivity(sample, Freq, materials, spir):
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
    def_c = get_material_data_freq(materials['lung']['cond'],Freq)
    inf_c = get_material_data_freq(materials['lung']['infl'],Freq)
    spiramp = max(spir[:,1]) - min(spir[:,1])
    condamp = def_c - inf_c
    condspir = sample.copy()
    condspir[:,1] = (-sample[:, 1] + max(spir[:, 1]))*(condamp/spiramp) + inf_c
    return condspir

def meas_voltages_slice(elecs):
    '''
    get all voltages on electrodes in one slice
    Args:
        3d array of electrodes edges and centers coordinates
    Returns:
        list of complex Voltages
    '''
    Nelec = elecs.shape[0]
    V = [0] * Nelec
    femm.co_seteditmode('contour')
    for i in range(Nelec):
        femm.co_selectpoint(elecs[i,0,0], elecs[i,0,1])
        femm.co_selectpoint(elecs[i,1,0], elecs[i,1,1])
        V[i] = femm.co_lineintegral(3)[0].real
        femm.co_clearcontour()
    return V

def simulate_EIT_projection(idx, elecs):
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
    gnd = 0 if idx == Nelec - 1 else idx + 1
    femm_set_elec_state('INJ', elecs[idx, 2])
    femm_set_elec_state('GND', elecs[gnd, 2])
    not_visible = 1
    femm.ci_analyze(not_visible)
    femm.ci_loadsolution()
    elec_volts = meas_voltages_slice(elecs)
    femm_set_elec_state('None', elecs[idx, 2])
    femm_set_elec_state('None', elecs[gnd, 2])
    return elec_volts

def calculate_EIT_slice_fast(fullfpath, elecs, tissue_props):
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
        raise ValueError(f'no projection number in problem file path ({fpath})')
    femm_prepare_problem(fname = fullfpath)
    femm.smartmesh(0)
    Nelec = elecs.shape[0]
    gnd = 0 if idx == Nelec - 1 else idx + 1
    femm_set_elec_state('INJ', elecs[idx, 2])
    femm_set_elec_state('GND', elecs[gnd, 2])
    femm.ci_createmesh()
    Nelems = 0
    V = []
    for tisue_name, tissue_info in tissue_props.items():
        for tissue_param, vals in tissue_info.items():
            if not Nelems:
                Nelems = len(vals)
            else:
                if Nelems != len(vals):
                    raise ValueError((f'bad len of {tissue_param} values for'
                                      f'{tisue_name}'))
    for i in range(Nelems):
        for tisue_name, tissue_info in tissue_props.items():
            for tissue_param, vals in tissue_info.items():
                femm_modify_material(tisue_name,tissue_param, vals[i])
        femm.ci_analyze(1)
        femm.ci_loadsolution()
        V.append(meas_voltages_slice(elecs))
        femm.co_close()
    femm_set_elec_state('None', elecs[idx, 2])
    femm_set_elec_state('None', elecs[gnd, 2])
    femm.closefemm()
    return V

def simulate_EIT_slice(fpath, elecs, tissue_props):
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
    iterelecs = [elecs] * Nelec
    ittertissues = [tissue_props] * Nelec
    with ProcessPoolExecutor(max_workers = Nelec) as executor:
        results = np.array(list(executor.map(calculate_EIT_slice_fast, fpath, iterelecs, ittertissues)))
        volts = np.reshape(np.transpose(results, (0, 2, 1)), (Nelec ** 2, results.shape[1]))
    return volts

def simulate_EIT_monitoring(fpath, condspir, elecs):
    '''
    simulate EIT monitoring with changing lungs conductivity
    in time
    Args:
        fpath - list of problems full path
        condspir - 2d np array with change of lungs conductivity in time
    '''
    Nelec = elecs.shape[0]
    #V = np.zeros([Nelec, , Nelec, condspir.shape[0]])
    tissue_props = {'lung' : {'cond' : condspir[:, 1]}}
    V = simulate_EIT_slice(fpath, elecs, tissue_props)
    return V

def test_module():
    elecs = [[[ 9.97650385e+00, -1.17559591e+02],
              [-9.97650385e+00, -1.18929804e+02],
              [ 1.89091337e-01, -1.28242909e+02]],
             [[-1.24842855e+02, -1.14597243e+02],
              [-1.40980978e+02, -1.02783649e+02],
              [-1.40302540e+02, -1.15426817e+02]],
             [[-1.92151392e+02, -9.15145286e+00],
              [-1.87491895e+02,  1.02982024e+01],
              [-1.99808363e+02,  5.81753658e-02]],
             [[-1.38500936e+02,  1.12466346e+02],
              [-1.23955330e+02,  1.26193157e+02],
              [-1.38858896e+02,  1.25792835e+02]],
             [[-1.42012137e+01,  1.54770621e+02],
              [ 5.73767222e+00,  1.56332936e+02],
              [-4.35575653e+00,  1.65551010e+02]],
             [[ 1.16064755e+02,  1.31726519e+02],
              [ 1.30882324e+02,  1.18293736e+02],
              [ 1.30863825e+02,  1.31746869e+02]],
             [[ 1.89725565e+02,  2.68899821e+01],
              [ 1.90555706e+02,  6.90721790e+00],
              [ 2.00134664e+02,  1.72441358e+01]],
             [[ 1.33461374e+02, -1.02958664e+02],
              [ 1.16817517e+02, -1.14048390e+02],
              [ 1.32459169e+02, -1.15316870e+02]]]
    elecs = np.array(elecs)
    fpath = ['./models/temp/test' + str(i) + '.fec' for i in range(16)]
    #tissue_props = {'lung' : {'cond' : [0.03]*20}}
    #v = simulate_EIT_slice(fpath, elecs, tissue_props)
    materials = get_materials('./models')
    Freq = 50000
    spir = get_spirometry_ref('./models/data/vent.csv')
    dT = spir[1,0]
    t1 = 1.5;
    t2 = 61.5;
    idx = np.where(np.logical_and(t1 <= spir[:,0], spir[:,0] <= t2))[0]
    sample = spir[idx[0]:idx[-1]:5]
    dataf = sample.copy()
    dataf[:, 1] = filt_FFT('bypass', 1/dT, (0.05, 0.5), sample[:,1] - np.mean(sample[:,1]))
    condspir = spirometry_to_conuctivity(dataf, Freq, materials, spir)
    v = simulate_EIT_monitoring(fpath, condspir, elecs)
    print(v.shape)

if __name__ == "__main__":
    import timeit
    print(timeit.timeit('test_module()', globals=globals(), number = 1))