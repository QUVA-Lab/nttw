# Research virus spread and contact tracing

This repo contains accompanying code for the AISTATS submission, 'no time to waste'.

Note: we are working on a new version, which already runs faster and compares additional inference methods. Please contact romijndersrob@gmail.com for more information.

## Typical usage

Starting point for experiments will be the following command:

```
python3 nttw/experiments/compare_stats.py \
    --inference_method "dummy" \
    --experiment_setup "prequential" \
    --config_data intermediate_graph_abm_02 \
    --config_model model_IG02
```

Experiments take two configs: one for the model and one for the simulator (data).
Whenever 'abm' is in the data config, the ABM simulator will be used.

Experimental setup could be 'single', where inference will be performed on a single, static graph, or 'prequential',
where an experiment with conditional testing and quarantining will be performed (similar to related research like CRISP and SIB).

## Typical data generation
For 'single' inference, a static graph must be created and dumped. As this generation can take time, this code is
multiprocessed:

```
python3 nttw/data/generate_graph.py \
    --config intermediate_graph_02 --sample_contact
```

'sample_contact' graph indicates that contacts should be sampled (in addition to running a single realisation of the states).
The corresponding config should also be used in the call to 'nttw/experiments/compare_stats.py'. These configs contain
information about the dynamics, transmission/transition probabilities etcetera.

## Code convention

Code convention: We care deeply about good code and scientific reproducibility. As of september 2022, the code contains
57 unittests, spanning more than one thousands line of code (`make test` or `nose2 -v`).

The code includes abundant type hints (`make hint` or `pytype nttw`).

Code is styled with included '.pylintrc' and pycodestyle (`make lint` or `pylint nttw`)

## Installation

For GSL, follow [these instructions](https://coral.ise.lehigh.edu/jild13/2016/07/11/hello/)

```
# get the installation file
wget ftp://ftp.gnu.org/gnu/gsl/gsl-latest.tar.gz

# Unpack archive
tar -zxvf gsl-latest.tar.gz

# make a directory for the gsl installation
mkdir /var/scratch/${USER}/projects/gsl

# installation
./configure --prefix=/var/scratch/${USER}/projects/gsl
make
make check
make install
```

[SWIG](https://www.swig.org/) install
```
sudo apt-get update
sudo apt-get -y install swig
```

ABM install
```
# Get the ABM code
cd ../

mkdir abm
cd abm

git clone https://github.com/aleingrosso/OpenABM-Covid19.git .

cd src
make all

make swig-all
```

Insights from debugging:
  * 'gsl/gsl_rng.h: No such file or directory' -> make sure the includes are set correctly. Like -I/var/scratch/${USER}/projects/gsl/include to the compiler
  * 'cannot find -lgsl' -> Make sure the libraries are set correctly. Like -L/var/scratch/${USER}/projects/gsl/lib to the linker


## Run a sweep with WandB
To run a sweep with WandB, run the following command

```
$ wandb sweep sweep/stale_abm.yaml
```

Copy the sweepid. Then on the cluster, or another computer, start up an agent with

```
$ export SWEEP=sweepid
$ wandb agent "$USERNAME/nttw-nttw_experiments/$SWEEP"
```


## Attribution

Any questions may go to 'romijndersrob [at] gmail [dot] com'.

Please use the following for citations:

```
R. Romijnders, Y.M. Asano, C. Louizos, and M. Welling, 'No time to waste: practical statistical contact tracing with few low-bit messages', AISTATS 2023
```

```
@article{2023notimetowaste,
  title={No time to waste: practical statistical contact tracing with few low-bit messages},
  author={Romijnders, Rob and Asano, Yuki and Louizos, Christos and Welling, Max},
  journal={Accepted for AISTATS},
  year={2023},
}
```