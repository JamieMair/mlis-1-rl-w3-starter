# MLiS I - RL Week 3 Workshop Q2

Starter project for implementing a basic REINFORCE algorithm

## Task

Complete Q2 of the workshop sheet by modifying code inside blocks similar to
```python
#### TODO:
#
#### END
```


## Getting Started

### Downloading the code
Download the code from this project either as a `.zip` using the `<> Code` button **OR** click the green  `Use this template`  button to create your own GitHub repository, and then clone your repository.

### Python Setup

You must have access to Python and install the following packages:
- `numpy`
- `matplotlib`
- `tqdm`

You can install all of these with `pip` in the command line:
```bash
pip install numpy matplotlib tqdm
```
**OR** run the following in a terminal opened to the same folder as this README:
```bash
pip install -r requirements.txt
```

### Running the code

#### Terminal
The easiest way to run the code is to open the terminal to this folder and run
```bash
python main.py
```
#### VS Code

You can run the script using the "play" button on the top right of the `main.py` script.

#### Jupyter Notebook
If you prefer Jupyter Notebook, create a new notebook and copy in the source code, splitting into different cells if you prefer.

### Final Checks

Running the code in the initial state should produce a graph similar to:

![Avg Return Learning Curve - No Learning](/no_learning_curve.png)


## Verification

You should make sure your implementation is working by seeing the learning curve approach the maximum return.

![Avg Return Learning Curve - REINFORCE](/learning_curve_reinforce.png)