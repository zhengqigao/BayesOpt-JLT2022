# Bayesian Optimization - JLT2022

This repo contains code for our paper titled 'Automatic Synthesis of Broadband Silicon Photonic Devices via Bayesian Optimization' submitted to IEEE Journal of Lightwave Technology (IEEE JLT). 

Bayesian optimziation is a generic optimization approach suitable to medium dimensional problems (e.g., <20 design variables) in many application areas. If you are looking for a bare-bone implementation of Bayesian optimization and want to adapt it for your purpose, please see the folder SimpleBO. If you are interested in reproducing results of our paper, please see the folder Directional-Coupler, Y-Splitter, and Ring-Resonator.

For now, only SimpleBO is not empty. The other folders will be updated as soon as our JLT paper is accepted. If you find our code to be useful, please cite our [conference paper](https://opg.optica.org/viewmedia.cfm?r=1&uri=CLEO_AT-2022-JW3B.156&seq=0) at this time.

```
@inproceedings{Gao2022Automatic,
author = {Zhengqi Gao and Zhengxing Zhang and Duane S. Boning},
booktitle = {Conference on Lasers and Electro-Optics},
journal = {Conference on Lasers and Electro-Optics},
pages = {JW3B.156},
publisher = {Optica Publishing Group},
title = {Automatic Design of a Broadband Directional Coupler via Bayesian Optimization},
year = {2022},
url = {http://opg.optica.org/abstract.cfm?URI=CLEO_AT-2022-JW3B.156}
}
```

Our Bayesian optimization code has been extensively used by researchers at many top universities, such as MIT, Columbia University, University of Notre Dame, UCSB. 

<img src="https://libraries.mit.edu/mithistory/wp-content/files/mit-seal_400x400-300x300.gif" width = "100" height = "100"/> <img src="http://www.columbiamedicinemagazine.org/sites/default/files/images/fall2017-psNews-columbiaSeal568.jpg" width = "100" height = "100"/> <img src="https://upload.wikimedia.org/wikipedia/commons/e/e2/University_of_Notre_Dame_seal_%282%29.svg" width = "100" height = "100"/> <img src="http://web.physics.ucsb.edu/~hepjc/ucsbseal.png" width = "100" height = "100"/>


Please ping me if you achieve your goal with the help of our code: I would like to hear your successful story! If you have any other problems, please feel free to contact me zhengqi@mit.edu. See my homepage for more details about me: https://zhengqigao.github.io/.
