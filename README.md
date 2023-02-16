# THERMAL AND MECHANICAL ANALYSIS

The idea of this repository is to do a thermal analysis, then these results are going to be induced into a mechanical analysis in the form of initial strains. With this you can have a loaded beam for example, and subject it to a gradient of temperatures and see how the beam behaves. Here goes an example:
![image](https://user-images.githubusercontent.com/111939223/219373375-a878003a-47cb-42b5-91c0-685888e69531.png)


# How to use:
1. First, you have to compile OpenSees from this branch I have https://github.com/j0selarenas/OpenSees/tree/develop-larenas
2. You download the thermal folder, ```functions.py``` & ```runAnalysis.py```
3. You install the required libraries (```pip install tabulate, numpy, scipy, h5py```)
4. Then you only need to open ```runAnalysis.py``` and run the code.

# Observations:
- In ```runAnalysis.py``` there are some parameters you can modify depending on what you need.
- If you don't know how to compile OpenSees, then you can contact me directly for help.
