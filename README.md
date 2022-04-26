# 2022 - NRI

Reimplementation of the **N**eural **R**elation **I**nference proposed in the following paper: Kipf, Thomas, et al. "Neural relational inference for interacting systems." *International Conference on Machine Learning*. PMLR, 2018.
<p align="center">
<figure>
<img src="figures/paper_fig.PNG" alt="Drawing" style="width: 600px;"/>
<figcaption align = "center"><b>Results figures in Neural Relational Inference for Interacting Systems</b></figcaption>
</figure>
</p>

## Prerequisites

Recommend using conda virtual environment. A `environment.yml` file has been set up. Simply run the following command to setup the required environment.

```
conda env create --name recoveredenv --file environment.yml
```

Next, create a local package (named `src`). Notice that `-e` indicates that the package is editable (no need to reinstall ) and `.` indicates the current folder. This approach takes the advantage of python package system. 

```
pip install -e.
```

## Model training and testing

Run the following code to train the encoder and the decoder respectively. 
When the best model (selected through validation) is obtained, test the model.

```
/scripts$ python train_enc.py
/scripts$ python train_dec.py
```

You can further adjust training arguments. For details, use `python train_enc.py  -h`.

Notice that GPU is not necessary for training. You can train the model in a short time on a CPU platform. 

## Run demo

We provide `run_decoder.py` and `run_encoder.py` for generating trajectory based on trained model. The generated trajectory will be save in the `saved_results` folder. 
Then use the notebook `traj_plot.ipynb` to plot a gif.

The visualization part of `run_encoder.py` is still under consideration. 

<table><tr>
<td> <figure>
<img src="figures/demo_grand_truth.gif" alt="Drawing" style="width: 300px;"/> 
</figure></td>
<td> <figure>
  <img src="figures/demo_model_out.gif" alt="Drawing" style="width: 300px;"/>
</figure></td>
</tr>
<tr>
  <td> <center>Ground truth trajectory</center></td>
  <td> <center>Prediction trajectory</center></td>
</tr>  
</table>


## TODOs



