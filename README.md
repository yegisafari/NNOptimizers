# NNOptimizers
Neural Network Optimizers Study
# Neural Network Optimizers Study (Part 2)

## Project Overview

This project is a study on neural network optimizers using Keras to analyze their behavior on complex functions. The primary focus is on visualizing the optimization paths and evaluating the performance of different optimizers. The functions explored include the Six-Hump Camel function, the Michalewicz function, and at least two other suitable functions from [this list](https://www.sfu.ca/~ssurjano/optimization.html).

## Tasks and Objectives

### 1. Visualizing Functions
- Visualize some of the functions, including:
  - Six-Hump Camel function
  - Michalewicz function
  - At least two additional functions from the provided list

### 2. Optimizer Implementation
Use the following Keras optimizers to find the minimum of these functions:
- SGD optimizer
- Vanilla momentum optimizer
- Nesterov accelerated momentum optimizer
- AdaGrad
- RMSProp
- Adam

### 3. Visualization of Optimization Paths
- Visualize the trajectory taken by each optimizer from a starting point \((x_{\text{start}}, y_{\text{start}})\) to the minimum \((x_{\text{min}}, y_{\text{min}})\) of the function \(z = f(x, y)\).
- Visualization can be done in 2D (projected onto the plane spanned by \(p1\) and \(p2\)) or in 3D (visualizing the path along the loss surface).

### 4. Analysis of Optimizers
- Indicate which optimizers fail to reach a minimum and which get stuck in a local rather than the global minimum.
- Keep track of the total number of steps each optimizer required to find a minimum of the function.
  - Use an absolute (target) error of \(10^{-13}\) as a stopping criterion: stop the minimization procedure when \(|f(x_i, y_i) - f(x_{i+1}, y_{i+1})| \leq 10^{-13}\) for two consecutive steps.
  - Limit the total number of optimization steps to a large number (e.g., 100,000 or 1,000,000) if necessary.

### 5. Custom Keras Training Loop
- Use TensorFlow’s GradientTape method to compute the two-dimensional gradient vector \(\nabla f(x_i, y_i)\) at each iteration.
- Implement a custom Keras training loop where the variables are \(x\) and \(y\).

## Repository Structure
- `src/`: Contains the implementation of the function visualizations, optimizer applications, and custom training loops.
- `notebooks/`: Jupyter notebooks for step-by-step exploration and visualization.
- `results/`: Logs and visualizations of optimization paths and performance metrics.

## Usage
1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/neural-network-optimizers-study.git
    cd neural-network-optimizers-study
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the visualization scripts:
    ```sh
    python src/visualize_functions.py
    ```

4. Run the optimization experiments:
    ```sh
    python src/optimize_functions.py
    ```

5. Explore the results in the `results/` directory:
    ```sh
    ls results/
    ```

## Results
The results of the experiments, including visualizations of the optimization paths and performance metrics, are documented in the `results/` directory.

## References
- Original Functions List: [https://www.sfu.ca/~ssurjano/optimization.html](https://www.sfu.ca/~ssurjano/optimization.html)

