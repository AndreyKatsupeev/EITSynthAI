# Modeling for generating synthetic datasets with FEMM
import femm
from .model_generator import create_model, get_materials
from .femm_api import get_material_data_freq, femm_create_problem, femm_modify_material, femm_set_elec_state, femm_close
import numpy as np
import os

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
        3d array of electrodes edges coordinates
    Returns:
        list of complex Voltages
    '''
    V = []
    femm.co_seteditmode('contour')
    Nelec = elecs.shape[0]
    for i in range(Nelec):
        femm.co_selectpoint(elecs[i,0,0], elecs[i,0,1])
        femm.co_selectpoint(elecs[i,1,0], elecs[i,1,1])
        V.append(femm.co_lineintegral(3)[0].real)
        femm.co_clearcontour()
    return V

def simulate_EIT_slice(elecs, cents):
    '''
    simulate EIT current injection and measurment
    in created FEMM problem - seletcs all neighbour electrodes
    as current injection and zero voltage and meas all 
    electrodes voltages
    '''
    V = []
    Nelec = elecs.shape[0]
    for i in range(Nelec):
        gnd = 0 if i == 15 else i + 1
        femm_set_elec_state('INJ', cents[i])
        femm_set_elec_state('GND', cents[gnd])
        femm.ci_createmesh()
        not_visible = 1
        femm.ci_analyze(not_visible)
        femm.ci_loadsolution()
        V.extend(meas_voltages_slice(elecs))
        femm.co_close()
        femm_set_elec_state('None', cents[i])
        femm_set_elec_state('None', cents[gnd])
    return V

def simulate_EIT_monitoring(fname, condspir, centers, elecs):
    '''
    simulate EIT monitoring with changing lungs conductivity
    in time
    Args:
        fname - problem full file name with extension
        condspir - 2d np array with change of lungs conductivity in time
    '''
    femm_create_problem(fname = fname)
    Nelec = elecs.shape[0]
    V = np.zeros([condspir.shape[0], Nelec ** 2])
    i = 0
    for t, cond in condspir:
        femm_modify_material('lung','cond', cond)
        V[i, :] = simulate_EIT_slice(elecs, centers)
        i += 1
    femm_close()
    return V

def test_module():
    centers = [[ 1.15395082e-01, -1.28244031e+02],
               [-6.81704177e+01, -1.34733320e+02],
               [-1.40326752e+02, -1.15400157e+02],
               [-1.80208490e+02, -6.60214448e+01],
               [-1.99808497e+02,  6.07779113e-02],
               [-1.75877399e+02,  6.94759984e+01],
               [-1.38882178e+02,  1.25765246e+02],
               [-7.50534301e+01,  1.57177305e+02],
               [-4.42091798e+00,  1.65549989e+02],
               [ 6.58785956e+01,  1.64413436e+02],
               [ 1.30838425e+02,  1.31774627e+02],
               [ 1.76211435e+02,  7.17205578e+01],
               [ 2.00134604e+02,  1.72458598e+01],
               [ 1.80019363e+02, -5.26232257e+01],
               [ 1.32433797e+02, -1.15344026e+02],
               [ 7.32533191e+01, -1.27977390e+02]]
    elecs = [[[   9.97650385, -117.55959067],
              [  -9.97650385, -118.92980356]],
             [[ -54.0184444,  -124.91306847],
              [ -73.96429823, -126.3837536 ]],
             [[-124.84285543, -114.59724278],
              [-140.9809784,  -102.78364946]],
             [[-166.11395741,  -70.81581597],
              [-175.90766048,  -53.37782927]],
             [[-192.15139181,   -9.15145286],
              [-187.49189494,   10.29820237]],
             [[-170.59024742,   57.17605595],
              [-162.22716189,   75.34357643]],
             [[-138.50093632,  112.46634602],
              [-123.95532971,  126.1931566 ]],
             [[ -80.41952433,  146.47161907],
              [ -60.73311799,  149.9994237 ]],
             [[ -14.20121372,  154.770621  ],
              [   5.73767222,  156.33293581]],
             [[  52.01791283,  156.93527008],
              [  71.73182368,  153.56455518]],
             [[ 116.06475457,  131.72651863],
              [ 130.88232372,  118.29373618]],
             [[ 161.70708008,   77.07968507],
              [ 171.81510674,   59.82199702]],
             [[ 189.7255649,    26.88998212],
              [ 190.55570599,    6.9072179 ]],
             [[ 174.712581,    -40.24882322],
              [ 166.4275768,   -58.45208406]],
             [[ 133.46137384, -102.95866378],
              [ 116.81751691, -114.04838995]],
             [[  78.4894063,  -119.56997042],
              [  58.50454499, -118.79194823]]]
    materials = get_materials('./models')
    Freq = 50000
    spir = get_spirometry_ref('./models/data/vent.csv')
    dT = spir[1,0]
    t1 = 1.5;
    t2 = 2;
    idx = np.where(np.logical_and(t1 <= spir[:,0], spir[:,0] <= t2))[0]
    sample = spir[idx]
    dataf = sample.copy()
    dataf[:, 1] = filt_FFT('bypass', 1/dT, (0.05, 0.5), sample[:,1] - np.mean(sample[:,1]))
    condspir = spirometry_to_conuctivity(dataf, Freq, materials, spir)
    simulate_EIT_monitoring('./models/temp/test.fec', condspir, centers, np.array(elecs))

if __name__ == "__main__":
    import timeit
    print(timeit.timeit('test_module()', globals=globals(), number = 1))