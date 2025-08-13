# Multi-output-shrunken-regression-trees


This repository contains the implementation of various multi-output shrunken regression tree methods and their applications to real-world and synthetic datasets. 


## Setup

To set up the project, follow these steps:

1. Clone the repository:

   ```sh
   git clone https://github.com/SimonTianPolyU/Multi-output-shrunken-regression-trees.git
   cd MOSRT

2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3.	Install the required libraries:
   
      Ensure you have the following dependencies installed in your environment:

    - Python: 3.11 and above
    - Scikit-learn: 1.6.0
    - NumPy: 2.2.0
    - Pandas: 2.2.3
    
## Usage
1. To run the experiments for real-world datasets, use the following command:
   ```sh
   python RealWorldExperiments.py

2. To run the experiments for synthetic datasets, use the following command:
   ```sh
   python SyntheticExperiments.py

## Citation
If you use this code, please cite the following paper:

Tian, X., Wang, S., Laporte, G. Multi-output shrunken regression trees.

## License
This project is licensed under the MIT License.
