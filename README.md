# Image Segmentation and Synthetic Dataset Generation Library  

This library is designed to process tomographic images for medical research, enabling segmentation of anatomical structures, finite element mesh generation, and synthetic dataset creation for Electrical Impedance Tomography (EIT).  

## Features  
- **Segmentation**: Automatically segment tomographic images into human thoracic cavity organs.  
- **Finite Element Mesh Generation**: Generate geometry and finite element meshes with predefined conductivity for FEMM software.  
- **Synthetic Dataset Generation**: Create synthetic EIT datasets for scientific research and experimentation.  

## Use Cases  
- Medical research in EIT measurement analysis and synthetic dataset generation.  
- Finite element model generation for industrial design, game engines, and simulation tasks.  

## Requirements  
- Python 3.8+  
- Required libraries are listed in `requirements.txt`.

# Now library is in early stages of development


## Contribution

Contributions are welcome! Please create a pull request or submit an issue if you encounter problems or have ideas for improvement.

This project is funded by Foundation for Assistance to Small Innovative Enterprises in the Scientific and Technical Sphere, Russia, Moscow.

# Инструкция по запуску проекта

1. **Скопировать проект**  
   Выполните команду:  
   ```bash
   git clone git@github.com:AndreyKatsupeev/EITSynthAI.git
   
2. **Скачать веса модели по [ссылке](https://github.com/user/repo/blob/branch/other_file.md)**
3. **Поместить веса в директорию weights/**
4. **Из корня проекта запустить команду**
   ```bash
   docker compose up --build
5. **После сборки вставить в поисковую строку браузера URL**
   ```bash
   http://0.0.0.0:8601/

6. **Данные для тестов можно скачать по [ссылке](https://disk.yandex.ru/d/umV5bwXXuZrciw)**