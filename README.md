# CSDesign
<!-- <img src="https://github.com/anonymized/cs_design/blob/master/data/figs/title_fig_csd.png?raw=true" alt="drawing" width="700"/> -->

CSDesign is an algorithm for designing proteins with high conformational specificity.

![title_fig_csd (3)-1](https://github.com/user-attachments/assets/5cea7c62-e200-4c68-98e6-71f8cf1fab4c)

## Requirements
- micromamba

## Installation
```
# Micromamba install command (run and follow prompts)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
# Clone CSDesign repo
git clone https://github.com/anonymized/cs_design.git
# Create environment
cd cs_design
micromamba env create -f cs_design.yaml
```


## Examples
- Redesign a protein to prefer one conformation (CSDesign)
```
python3 -m cs_design.design --model_name cs_design --protein_id 4GSB --protein_id_anti 2ERK --decode_order n_to_c --decode_algorithm greedy --fixed_positions 1 168 187 358
```
- Generate MotifDiv dataset and run a benchmark
```
python3 -m cs_design.experiments.scrmsd_experiment
````
