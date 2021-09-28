Based on [Air Quality Predictions with a Semi-Supervised Bidirectional LSTM Neural Network](https://www.doi.org/10.1016/j.apr.2020.09.003) by Zhang et al.
Cite as: Zhang L., Liu P., Zhao L., Wang G., and Liu J. (2020). Air quality predictions with a semi-supervised bidirectional lstm neural network. Atmospheric Pollution Research, 12(1).\
@article{2020Air,\
  title={Air quality predictions with a semi-supervised bidirectional LSTM neural network},\
  author={ Zhang, L.  and  Liu, P.  and  Zhao, L.  and  Wang, G.  and  Liu, J. },\
  journal={Atmospheric Pollution Research},\
  volume={12},\
  number={1},\
  year={2020},\
}

our experiment envs (environment.yml) mainly contains：\
CUDA 9.0 \
tensorflow 1.13.1 or later\
keras 2.2.4 or later\
emd-signal 1.0.0 (pip install EMD-signal)

all original PM2.5 data and EMD decomposed data are in folder ./data/ ;\
all model results are in folder ./result/xxx

experiment step:\
Step 1. Run EMD_decompose.py for original PM2.5 data decomposing.\
   Obtain EMD decompositions. (imf)
   
Step 2. Modify the config param in Config.py\
   VERSION: different experiment control.\
   IMF_NUM: experimental data selection in step 1
   
Step 3. Run Train.py for training the selected imf data.

Step 4. Repeat step 2 and 3 until all decompositions are trained.

Step 5. Modify the config param in Config.py\
    Run Predict.py for different imf.\
    Modify line 23 in Predicted.py (choose model param files) 
    
Step 6. Aggregate all imf results to obtain final PM2.5 predictions.
