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

## Под системой Windows

На ПК должны быть предустановлены [Docker](https://www.docker.com/products/docker-desktop//d/KWZ-lDjv8seAfQ), [git](https://git-scm.com/install/windows)

1. **Скопировать проект**  
   Выполните команду:  
   ```bash
   git clone git@github.com:AndreyKatsupeev/EITSynthAI.git
   
2. **Скачать веса модели по [ссылке](https://disk.yandex.ru/d/KWZ-lDjv8seAfQ)**
3. **Поместить веса в директорию weights/**
4. **Установить программное обеспечение FEMM. Скачать можно по [ссылке](https://www.femm.info/wiki/Download)**
5. **Из корня проекта запустить команду**
   ```bash
   wsl --update
6. **Запустить Docker на ПК**
7. **Из корня проекта запустить команду**
   ```bash
   docker compose up --build -d
8. **После сборки фронт сервиса будет доступен по [адресу](http://0.0.0.0:8601/) или [адресу](http://localhost:8601/)**

9. **Данные для тестов можно скачать по [ссылке](https://disk.yandex.ru/d/umV5bwXXuZrciw)**


## Под системой Linux

На ПК должны быть предустановлены [Docker](https://www.docker.com/products/docker-desktop//d/KWZ-lDjv8seAfQ), [git](https://git-scm.com/install/linux)

1. **Скопировать проект**  
   Выполните команду:  
   ```bash
   git clone git@github.com:AndreyKatsupeev/EITSynthAI.git
   
2. **Скачать веса модели по [ссылке](https://disk.yandex.ru/d/KWZ-lDjv8seAfQ)**
3. **Поместить веса в директорию weights/**
4. **Из корня проекта запустить команду**
   ```bash
   docker compose up --build -d
5. **После сборки фронт сервиса будет доступен по [адресу](http://0.0.0.0:8601/) или [адресу](http://localhost:8601/)**
6. **Данные для тестов можно скачать по [ссылке](https://disk.yandex.ru/d/umV5bwXXuZrciw)**