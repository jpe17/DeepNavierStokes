# DeepRANSProject

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.6%2B-blue.svg)
![GitHub Issues](https://img.shields.io/github/issues/jpe17/DeepRANSProject.svg)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Tutorial](#tutorial)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

**DeepRANSProject** is a cutting-edge project aimed at solving the Reynolds-Averaged Navier-Stokes (RANS) equations for 1D channel flow using neural networks. By leveraging various closure models, this project explores the integration of deep learning techniques with traditional fluid dynamics to achieve accurate and efficient simulations.

Welcome to the project! Below is an overview video that explains the key features and functionalities.

<video width="600" controls>
  <source src="../assets/videos/overview.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Features

- **Neural Network Integration**: Utilizes the [NeuroDiffEq](https://github.com/odegym/neurodiffeq) library to solve differential equations with neural networks.
- **Multiple Closure Models**: Implements several closure models including:
  - Prandtl Mixing Length
  - Van Driest Mixing Length
  - Rui FCNN
  - Ocariz CNN
- **Flexible Boundary Conditions**: Supports mixed boundary value problems tailored to specific simulation requirements.
- **Extensible Framework**: Easily add new models and methodologies to expand the project's capabilities.

## Installation

### Prerequisites

- Python 3.6 or higher
- [PyTorch](https://pytorch.org/) library

### Install via Pip

DeepRANSProject relies on the `NeuroDiffEq` library, which can be installed using pip:

```bash
pip install neurodiffeq
```

### Manual Installation

For developers aiming to contribute or access the latest features:

1. **Create a Virtual Environment**:
   - Using `conda`:
     ```sh
     conda create --name deeprans_env python=3.7
     conda activate deeprans_env
     ```
   - Using `venv`:
     ```sh
     python3 -m venv deeprans_env
     source deeprans_env/bin/activate
     ```
2. **Clone the Repository**:
   ```sh
   git clone https://github.com/jpe17/DeepRANSProject.git
   cd DeepRANSProject
   ```
3. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   pip install .
   ```
4. **Run Tests** (Optional):
   ```sh
   cd tests
   pytest
   ```

## Usage

### Basic Example

Here's a simple example to get you started with solving the RANS equation using DeepRANSProject:

```python
import neurodiffeq as nd
```

For more detailed examples and advanced usage, refer to the [documentation](https://neurodiffeq.readthedocs.io/en/latest/).

## Tutorial

To help you get up and running quickly, we've created a [Colab notebook](https://colab.research.google.com/github/jpe17/DeepRANSProject/blob/main/DeepRANS_JoaoEsteves.ipynb) that walks you through setting up the environment, running simulations, and analyzing results.

### Steps:

1. **Set Up Working Space**:
   - Import the `DeepRANSProject_JoaoEsteves.zip` folder to your Google Drive.
   - Mount Google Drive to Google Colab and unzip the folder:
     ```python
     from google.colab import drive
     drive.mount('/content/gdrive', force_remount=True)
     !unzip /content/gdrive/MyDrive/DeepRANSProject_JoaoEsteves.zip -d /content/gdrive/MyDrive/
     ```
2. **Run Simulations**:
   - Navigate to the project directory and execute your simulation scripts.
3. **Analyze Results**:
   - Utilize built-in tools and libraries to visualize and interpret the simulation data.

## Contributing

We welcome contributions to DeepRANSProject! Whether you're reporting bugs, suggesting features, or submitting pull requests, your help is invaluable.

### Guidelines

1. **Fork the Repository**: Create your own fork to work on.
2. **Open an Issue**: Discuss the changes you plan to make.
3. **Create a Feature Branch**:
   ```sh
   git checkout -b feature/YourFeatureName
   ```
4. **Implement Your Changes**: Follow the [style guidelines](neurodiffeq/CONTRIBUTING.md#style).
5. **Write Tests**: Ensure your changes are well-tested.
6. **Submit a Pull Request**: Start the discussion by submitting a pull request.

For detailed contribution guidelines, please refer to the [CONTRIBUTING.md](neurodiffeq/CONTRIBUTING.md) file.

## License

This project is licensed under the [MIT License](neurodiffeq/LICENSE). You are free to use, modify, and distribute the software as per the terms of the license.

## Acknowledgements

- [NeuroDiffEq](https://github.com/odegym/neurodiffeq) for providing a robust framework for solving differential equations with neural networks.
- The open-source community for their continuous support and contributions.
- [Deep Learning](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) communities for their exceptional tools and libraries.

---

## Quick Walkthrough Demo Video

<p align="center">
  <a href="https://youtu.be/VDLwyFD-sXQ" target="_blank">
    <img src="https://img.youtube.com/vi/VDLwyFD-sXQ/maxresdefault.jpg" alt="Quick Walkthrough Demo Video" width="80%" style="border: none;">
  </a>
</p>

*Click the image above to watch the Quick Walkthrough Demo Video on YouTube.*

For more information, please visit our [documentation](https://neurodiffeq.readthedocs.io/en/latest/) or watch our [quick walkthrough demo video](https://youtu.be/VDLwyFD-sXQ).
