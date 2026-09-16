<div align="center">

# Drum Gesture Mapping Training Code


[![website](https://img.shields.io/badge/website-Project_Page-03b49b.svg)](https://jordieshier.com/projects/aimc2026/)
[![website](https://img.shields.io/badge/toolkit-Max4Live_Package-03b49b.svg)](https://jordieshier.com/projects/aimc2026/toolkit/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.08587-b31b1b.svg)](https://arxiv.org/abs//2609.08587)


[Jordie Shier](https://jordieshier.com), [Teresa Pelinski](https://teresapelinski.com/), [Charalampos Saitis](http://eecs.qmul.ac.uk/people/profiles/saitischaralampos.html), Andrew Robertson, and [Andrew McPherson](https://www.imperial.ac.uk/people/andrew.mcpherson)

</div>

This repository contains training code for our paper *Rescuing Performance from the Demo: Co-Designing Drum Gesture Mappings with a Percussionist*, which was presented at AIMC 2026.
The paper presented neural network models for real-time gesture mapping from drum performances to synthesizers.
This repo contains Python training code to produce models that run in the [Max4Live toolkit](https://jordieshier.com/projects/aimc2026/toolkit/).


## Install
Clone the repo and then install the `drum_gesture` package. Download the package:

```bash
git clone https://github.com/jorshi/aimc26-drum-gesture.git
cd aimc26-drum-gesture
```

install with uv:
```bash
uv sync
```

or with pip (if you use pip you can leave out `uv run` from commands below):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

## Example: Training a Continuous Model

Download an example dataset if you want to try running this example. Or, checkout `DATASET.md` for info on how to create your own training datasets.

```bash
wget https://pub-b51dee6d83db41bf888af08afc8dc988.r2.dev/aimc26-continuous-drum-dataset.tar.gz
tar xzf aimc26-continuous-drum-dataset.tar.gz
```

Train a continuous brushing model with recordings from a low tom:
```bash
uv run python scripts/train_continuous.py --config-name continuous_lowtom
```

or a snare buzz stroke roll model:
```bash
uv run python scripts/train_continuous.py --config-name continuous_buzz
```

Once training is complete (time depends on dataset size and your machine, but should ~10min) the results will be saved in a new folder `outputs/train_continuous/{date}/{time}`. 

The two files you need are `model.json` and `standardize.json`. These both need to be loaded into `JEM_InputBrush` or `JEM_InputBuzz` M4L objects, depending on the type of model you trained. They differ in input audio representation.
