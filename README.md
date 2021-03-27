# KMML Data Challenge

## How to recreate the submission

### Prerequisites
Extract the challenge data into a `data/` directory in the repositorys top-level.

Install required packages with `pip install -r requirements.txt`.

### Obtain the substring kernel files
The substring kernel is quite slow to compute, so the result
is only produced once and stored in a `.csv` file.
These files are required to recreate the submission.
You can either download the files (58MB compressed) that I precomputed 
from my [Google drive](https://drive.google.com/file/d/1oZJCbO9gTwuYUuCniTtYTlA4agQcpVGr/view?usp=sharing) 
(download the archive and extract it in the top-level) or let
the script create them for you. If you choose the second option,
you need to compile the C program that actually computes this kernel
with `make`. You don't have to run the program,
it will be automatically called with the right parameters by the python script.

WARNING: This may take some time to compute. To speed up the process
the script supports multithreading, you can select the number of workers
by changing the global variable `WORKERS` in the `kernels.py` file.
Computing one kernel with 4 workers takes about 10 minutes on my machine
(we need 6 such kernels). This script may also produce output
of up to 2GB (more than required) as it computes several kernels at one go
(suboptimal for testing, but very useful for cross-validation).
Because of all the reasons above it's strongly advised to just download the precomputed files.

### Run the script

Run `python start.py`. This creates a `Yte.csv` file in the `data/` directory that
is a replica of my selected submission.

NOTE: The `start.py` script runs a seeded cross-validation experiment
to recreate the best model. It's deterministic because a seed is used, 
but it takes some time to execute (about 10 minutes on my machine).

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