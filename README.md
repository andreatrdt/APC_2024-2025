This repository contains three projects developed as part of the APC 2024 assignments. Each project addresses a distinct topic and algorithmic challenge:

Assignment 1 – Power Iteration Methods
Linear Algebra / Eigenvalue Computation
Implements the Power Iteration method along with its inverse and shifted variants for approximating eigenvalues.

Assignment 2 – Neural Networks & CNNs
Introduction to Neural Networks and Convolutional Neural Networks (CNNs)
Implements the LeNet architecture for digit recognition using convolutional, pooling, and fully connected layers in C++.

Assignment 3 – Parallel Stable Marriage Problem (SMP)
Parallel Programming with MPI
Parallelizes the Gale-Shapley algorithm to solve the Stable Marriage Problem by splitting the computation of proposals and match updates across multiple MPI processes.


APC_2024_Assignments/
├── Assignment1/
│   ├── src/
│   │   ├── power_iteration.cpp
│   │   ├── inverse_power_iteration.cpp
│   │   ├── shift_inverse_power_iteration.cpp
│   │   └── other supporting source/header files
│   ├── inputs/
│   │   ├── input_10.txt
│   │   ├── ... (other test input files)
│   └── README.md  (assignment-specific instructions)
│
├── Assignment2/
│   ├── src/
│   │   ├── convolutional_layer.cpp
│   │   ├── max_pooling_layer.cpp
│   │   ├── fc_layer.cpp
│   │   └── other supporting source/header files
│   ├── dataset/
│   │   └── (input images and weights)
│   └── README.md  (assignment-specific instructions)
│
├── Assignment3/
│   ├── src/
│   │   ├── simulator.cpp
│   │   ├── dense_matrix.cpp
│   │   ├── logger.cpp
│   │   └── other supporting source/header files
│   ├── input/
│   │   ├── apps4.txt
│   │   ├── devices4.txt
│   │   └── (other input files)
│   └── README.md  (assignment-specific instructions)
│
└── README.md  (this file)
