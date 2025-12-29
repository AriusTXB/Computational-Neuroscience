# Exploring Neuron Models and Neural Networks 🧠

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Course-AIT3012-orange)

> **🎓 Acknowledgments**
>
> I would like to express my sincere gratitude to my supervisors, **Prof. Nguyen Linh Trung** and **Prof. Guy Nagels**, for their expert mentorship and valuable insights throughout this project.
>
> I also extend my heartfelt thanks to the **Teaching Assistants** of the *Computational Neuroscience and Applications* course for their constant support and technical guidance.

---

## 📌 Overview

This repository contains the source code and simulation reports for the course **AIT3012: Computational Neuroscience and Applications** at **VNU University of Engineering and Technology**.

The project investigates classic biological neuron models and brain-inspired neural network architectures. It implements simulations for:
1.  **Biophysical Modeling:** The Hodgkin-Huxley (HH) Model.
2.  **Simplified Spiking:** The Leaky Integrate-and-Fire (LIF) Model.
3.  **Reservoir Computing:** Echo State Networks (ESN) for chaotic time-series prediction.

The code is structured using **Object-Oriented Programming (OOP)** principles to ensure modularity between the physical modeling logic and the scientific visualization.

---

## 📂 Project Structure

```text
neural_network_assignment/
│
├── models/                  # Core Logic for Neural Models
│   ├── __init__.py
│   ├── hodgkin_huxley.py    # HH Model (ODEs and Ion Channels)
│   ├── lif.py               # LIF Model (Euler Integration)
│   └── reservoir.py         # Echo State Network (Ridge Regression)
│
├── plotting/                # Visualization Tools
│   ├── __init__.py
│   └── visualizer.py        # Seaborn/Matplotlib styling functions
│
├── output/                  # Generated Graphs and Plots
│   ├── 1_hodgkin_huxley_dashboard.png
│   ├── 2_lif_simulation.png
│   └── 3_reservoir_prediction.png
│
├── main.py                  # Entry point to run all simulations
├── requirements.txt         # Python dependencies
└── report.pdf               # Full LaTeX Report
```
---

## 🚀 Installation & Usage
1. Clone the Repository
code
Bash
git clone https://github.com/AriusTXB/Computational-Neuroscience.git
cd neuron-models-simulation
2. Install Dependencies
Ensure you have Python installed, then run:
code
Bash
pip install -r requirements.txt
3. Run Simulations
Execute the main script to generate all simulations and save plots to the output/ directory:
code
Bash
python main.py

---

## 🎓 Academic Info
```Bullet
Course: AIT3012 - Computational Neuroscience and Applications
Institution: Vietnam National University, Hanoi - University of Engineering and Technology (UET)
Student: Tran Xuan Bao
Student ID: 23020332
Supervisor: Prof. Nguyen Linh Trung & Prof. Guy Nagels
```
---

## 📖 References
```Bullet
Trappenberg, T. P. (2023). Fundamentals of computational neuroscience. Oxford University Press.
Dayan, P., & Abbott, L. F. (2001). Theoretical neuroscience. MIT Press.
Tavanaei, A., Ghodrati, M., Kheradpisheh, S. R., Masquelier, T., & Maida, A. S. (2019). Deep
learning in spiking neural networks. Neural Networks
Lukoševičius, M., & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training. Computer Science Review.
```