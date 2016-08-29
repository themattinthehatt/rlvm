Rectified Latent Variable Model (RLVM) 
===

The RLVM is a latent variable model developed to study large populations of simultaneously recorded neurons. 

For more information regarding the mathematical formulation of the model, see the preprint on [bioRxiv](https://github.com/themattinthehatt/rlvm). The *doc* directory contains a script that shows how to use the model on several synthetic datasets. 

The RLVM optimizes model parameters using [Mark Schmidt's](http://www.cs.ubc.ca/~schmidtm/) minFunc package, which is located in the *lib* directory and should work out of the box. If not you may need to run the mexAll.m file from the *lib/minFunc_2012* directory.
