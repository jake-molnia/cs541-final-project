First log into turing.

Then

```
cd scratch # we use scratch for faster IO
```

From here

```
git clone https://github.com/jake-molnia/cs541-final-project
git checkout simCSE
```

Then you can

```
cd cs541-final-project/01b-model-simCSE
```

And add the two slurm jobs

```
sbatch 00sc_run_model.slurm
sbatch 00sd_run_eval.slurm
```

They should generate a log file called `slurm_error_%j.log` which is expected.

To check they are running use:

```
squeue -u $USER
```
