# ASC-idealised

An idealised Antarctic Slope Current configuration in [Oceananigans.jl](http://github.com/CliMA/Oceananigans.jl) similar to the one used by [Stewart and Thompson, *GRL*, 2014](https://doi.org/10.1002/2014GL062281).

Please open an [issue](https://github.com/navidcy/ASC-idealised/issues) or start a new [discussion](https://github.com/navidcy/ASC-idealised/discussions/17) for any ideas, concerns, suggestions, or questions you may have!

To run this on NCI's HPC first from a login node (that has internet access) we clone the repository and instantiate
the project (i.e., download all necessary dependencies).

```
git clone ...
cd ASC-idealised
julia --project
```

```julia
julia> using Pkg; Pkg.instantiate();
```

Then we can either ask for an interactive GPU/CPU queue via, e.g.,

```
$ qsub -q gpuvolta -P x77 -l walltime=6:00:00 -l ncpus=12 -l ngpus=1 -l mem=128GB -l jobfs=10GB -N gpuineract -W umask=027 -l storage=gdata/v45+gdata/hh5+gdata/x77+scratch/v45+scratch/x77 -l wd -j n -I -X
```

or

```
$ qsub -q normal -P x77 -l walltime=6:00:00 -l ncpus=48 -l -l mem=190GB -l jobfs=10GB -N cpuineract -W umask=027 -l storage=gdata/v45+gdata/hh5+gdata/x77+scratch/v45+scratch/x77 -l wd -j n -I -X
```

and then 

```
julia --project
```

```julia
julia> include("asc.jl")
```

Alternatively we can submit a job to the usual non-interactive queue via

```
$ qsub asc.sh
```
