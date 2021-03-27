# KMML Data Challenge

## How to recreate the submission
Extract the challenge data into a 'data/' directory in the repositorys top-level.

Install required packages with `pip install -r requirements.txt`.

Run `python start.py`. This creates a `Yte.csv` file in the `data/` directory that
is a replica of my selected submission.

NOTE #1: The `start.py` script runs a seeded cross-validation experiment
to recreate the best model. It's deterministic because a seed is used, 
but it takes some time to execute (about 10 minutes on my machine).

NOTE #2: The `start.py` script uses precomputed matrices for the substring kernel
whenever available. These are included in the repository so should be present
by default, but read the next section if you don't have or don't want to use
the precomputed matrices.

## How to recompute the substring kernel
This kernel is computed by a C program, but it's invoked with the right setup
by the Python script. You just need to build the executable with `gcc substring.cpp -o substring`.

## How to conduct more experiments
The gateway to more experiments is the `train.py` script. It contains
the definitions of kernels, models and hyperparameters that will be tested
in cross-validation. For more control you can directly look at the `KernelCrossValidation`
object in `utils.py`.

## Troubleshooting
Managing the precomputed substring kernel files can sometimes be tricky,
so if something goes wrong you can first try deleting the `kernels/` directory
at the top-level (that's where the matrices are kept). They can be then
recreated by the scripts.