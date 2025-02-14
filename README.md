# BayesDesign
<img src="https://github.com/jacobastern/cs_design/blob/master/data/figs/title_fig_csd.pdf?raw=true" alt="drawing" width="700"/>

CSDesign is an algorithm for designing proteins with high conformational specificity.

Try out the CSDesign model here:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jacobastern/cs_design/blob/master/examples/CSDesign.ipynb)

## Requirements
- micromamba

## Installation
```
# Micromamba install command (run and follow prompts)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
# Clone BayesDesign repo
git clone https://github.com/jacobastern/cs_design.git
# Create environment
cd cs_design
micromamba env create -f cs_design.yaml
```

## One-line sequence design
To design a protein sequence to fit a protein backbone:
```
micromamba activate bayes_design
python3 design.py --model_name bayes_design --protein_id 6MRR --decode_order n_to_c --decode_algorithm beam --n_beams 128 --fixed_positions 67 68
```

## Other examples
- Redesign a protein to prefer one conformation (CSDesign)
```
python3 -m cs_design.design --model_name cs_design --protein_id 4GSB --protein_id_anti 2ERK --decode_order n_to_c --decode_algorithm beam --n_beams 128 --fixed_positions 16 16 31 34 52 52 62 62 65 65 68 69 147 165 183 184
```
- CSDesign with a spatial mask while preserving all residues within an 8-angstrom radius of fixed positions
```
python3 -m bayes_design.design --model_name cs_design --protein_id 4GSB --protein_id_anti 2ERK --decode_order n_to_c --decode_algorithm greedy --fixed_positions 1 168 187 358 --ball_mask
```
- Generate MotifDiv dataset and run a benchmark
```
python3 -m bayes_design.experiments.scrmsd_experiment
````