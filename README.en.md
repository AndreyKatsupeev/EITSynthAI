# EITSynthAI - Library for Image Segmentation and Synthetic Dataset Generation

This library is designed for processing tomographic medical imaging data with anatomical structure segmentation, finite element mesh generation, and creating synthetic datasets for Electrical Impedance Tomography (EIT).

# Project Overview

The EITSynthAI library is intended for automated processing of tomographic medical images, segmentation of anatomical structures, generation of synthetic datasets, and modeling of the Electrical Impedance Tomography (EIT) measurement process.

The library can be applied in medical imaging and diagnostics for automating organ segmentation in the thoracic cavity on computed tomography, magnetic resonance imaging, and EIT scans; in scientific research for generating synthetic data to train machine learning algorithms, including segmentation and reconstruction models; for building finite element meshes for biomechanics, electrophysiology, and medical device design problems; for creating educational datasets for courses in medical informatics, image processing, and numerical methods; and for modeling measurement processes, optimizing injection and measurement protocols, and validating EIT reconstruction methods.

The library is implemented with a modular architecture, ensuring flexibility, scalability, and integration capabilities with external tools. The module composition is listed below:

- Tomographic image preprocessing and segmentation module (kt_service). The module supports DICOM, NIfTI, JPEG, and PNG formats, performs automatic detection of the slice between the 6th and 7th ribs using a neural network model for rib detection, performs tissue segmentation (bones, muscles, fat, lungs), and image normalization and filtering. Average processing time per image is ~5 ms.

- Finite element mesh generation module (mesh_tools). The module converts segmentation results into a two-dimensional triangular mesh using Gmsh and classifies finite elements by anatomical structure affiliation. The module eliminates segmentation artifacts (cropped edges, noise, nested contours) and exports the mesh in formats compatible with FEMM and PyEIT.

- EIT measurement process modeling module (femm_tools). The module formulates the forward EIT problem based on the finite element mesh, assigns tissue electrical properties (conductivity, dielectric permittivity) depending on current frequency, generates synthetic measurement data, and models dynamic processes such as breathing. The module integrates with FEMM via API for solving current distribution problems.

- Synthetic dataset generation module (synthetic_datasets_generator). The module is designed for creating realistic datasets for AI model training and data augmentation based on physically grounded tissue models.

## Library Requirements

- Python 3.8+
- PyTorch 1.10+

## PC Technical Specifications:

- Hardware platform capable of running 64-bit Ubuntu version 22.04 LTS or higher or Windows version 10 or higher;
- At least 8 GB of RAM;
- Eight-core x86 processor 2.5 GHz and above or equivalent;
- At least 50 GB of free hard disk space;
- Keyboard and mouse (or touchpad).

# Project Launch Instructions

## For Windows

PC must have [Docker](https://www.docker.com/products/docker-desktop/) and [git](https://git-scm.com/install/windows) preinstalled.

1. **Copy the project**  
   Execute the command:  
   ```bash
   git clone git@github.com:AndreyKatsupeev/EITSynthAI.git
   ```

2. **Download model weights from [link](https://disk.yandex.ru/d/KWZ-lDjv8seAfQ)** and place them in the weights/ directory
   ```bash
   mkdir -p weights
   
   # Download the file
   echo "Downloading model weights..."
   wget "https://disk.yandex.ru/d/KWZ-lDjv8seAfQ" -O weights/model_weights.pth
   
   # Check download success
   if [ $? -eq 0 ]; then
       echo "Weights successfully downloaded to weights/model_weights.pth"
   else
       echo "Error downloading weights"
       exit 1
   fi
   ```

3. **Install FEMM software. Download from [link](https://www.femm.info/wiki/Download)**  
4. **From the project root, run the command**  
   ```bash
   wsl --update
   ```
5. **Start Docker on PC**  
6. **From the project root, run the command**  
   ```bash
   docker compose up --build -d
   ```
7. **After build, the frontend service will be available at [address](http://0.0.0.0:8601/) or [address](http://localhost:8601/)**  
8. **Test data can be downloaded from [link](https://disk.yandex.ru/d/z0EADQ_DNz15UQ)**  

## For Linux

PC must have [Docker](https://www.docker.com/products/docker-desktop/) and [git](https://git-scm.com/) preinstalled.

1. **Copy the project**  
   Execute the command:  
   ```bash
   git clone git@github.com:AndreyKatsupeev/EITSynthAI.git
   ```

2. **Download model weights from [link](https://disk.yandex.ru/d/KWZ-lDjv8seAfQ)** and place them in the weights/ directory
   ```bash
   mkdir -p weights
   
   # Download the file
   echo "Downloading model weights..."
   wget "https://disk.yandex.ru/d/KWZ-lDjv8seAfQ" -O weights/model_weights.pth
   
   # Check download success
   if [ $? -eq 0 ]; then
       echo "Weights successfully downloaded to weights/model_weights.pth"
   else
       echo "Error downloading weights"
       exit 1
   fi
   ```

3. **From the project root, run the command**  
   ```bash
   docker compose up --build -d
   ```
4. **After build, the frontend service will be available at [address](http://0.0.0.0:8601/) or [address](http://localhost:8601/)**  
5. **Test data can be downloaded from [link](https://disk.yandex.ru/d/z0EADQ_DNz15UQ)**  

## Usage Examples

Usage examples are available via [link](https://github.com/AndreyKatsupeev/EITSynthAI/blob/master/kt_service/ai_tools/mesh_tools/examples/README.md).