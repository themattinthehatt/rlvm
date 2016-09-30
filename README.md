Rectified Latent Variable Model (RLVM) 
===

The RLVM is a latent variable model developed to study large populations of simultaneously recorded neurons. 

For more information regarding the mathematical formulation of the model, see the preprint on [bioRxiv](http://biorxiv.org/content/early/2016/08/29/072173). The `doc` directory contains scripts that show how to use the model on several simulated datasets. 

The RLVM optimizes model parameters using [Mark Schmidt's](http://www.cs.ubc.ca/~schmidtm/) minFunc package, which is located in the `lib` directory and should work out of the box. If not you may need to run the mexAll.m file from the `lib/minFunc_2012` directory.
