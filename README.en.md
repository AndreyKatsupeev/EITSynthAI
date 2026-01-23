# EITSynthAI - Library for Image Segmentation and Synthetic Dataset Generation

This library is designed for processing tomographic images from medical research with anatomical structure segmentation, finite element mesh generation, and creating synthetic datasets for electrical impedance tomography (EIT).

## Project Overview

This library can be used in medical research for analyzing measurement data obtained from electrical impedance tomography and generating synthetic datasets for EIT research. The developed approach for obtaining finite element meshes based on tomographic images can be applied to any tasks using finite elements: in particular, in industrial device design, model development for game engines. The applied use cases are as follows:
- Automatic segmentation of tomographic images into human thoracic cavity organs.
- Generation of geometry and finite element meshes with predefined conductivity for FEMM software.
- Creation of synthetic EIT datasets for scientific research and experiments.

## Library Requirements

- Python 3.8+
- Pytorch 2.4+
- Additional dependencies listed in `requirements.txt`.

## System Requirements:

- Hardware platform capable of running 64-bit Ubuntu version 22.04 LTS or higher, or Windows version 10 or higher;
- At least 16 GB of RAM;
- Octa-core x86 processor 2.5 GHz or higher, or equivalent;
- Graphics card with cuda 12.+ support, and at least 8 GB of video memory;
- At least 100 GB of free hard disk space;
- Keyboard and mouse (or touchpad).

## Installation
In the command line, enter the appropriate commands:
```bash
git clone https://github.com/AndreyKatsupeev/EITSynthAI.git`
cd EITSynthAI`
```

## Usage

1. **Download model weights from [link](https://disk.yandex.ru/d/KWZ-lDjv8seAfQ)**
2. **Place weights in the weights/ directory**
3. **Install FEMM software. Can be downloaded from [link](https://www.femm.info/wiki/Download)**
4. **From the project root, run the command**
   ```bash
   docker compose up --build -d
5. **After building, the frontend service will be available at [address](http://0.0.0.0:8601/)**
6. **Test data can be downloaded from [link](https://disk.yandex.ru/d/umV5bwXXuZrciw)**

## Examples of usage 

Examples of usage are given in [link](EITSynthAI/mesh_service/examples/README.md)

## Licence

The library can be used free by the GNU General Public License.