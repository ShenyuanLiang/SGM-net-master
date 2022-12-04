 # Shape-Graph Matching Network (SGM-net): Registration of Shape Graphs Using Deep Networks
#### This repo is the official implementation of the paper [Shape-Graph Matching Network](https://link-url-here.org).

### Getting Started
#### Step 1 Download project:
- Clone into the repo by running following command. The password will be: **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**
```sh
    git clone https://ShenyuanLiang@github.com/ShenyuanLiang/SGM-net-master.git
```


#### Step 2 Pre-configuration of system distributions and python environment:
- The code is tested on Ubuntu 20.04 machine. In order to run the code, one needs an 24GB Nvidia GPU with CUDA support. The code does not support CPU execution currently.
- Set up system distributions and python packages needed. For system distritions, please check **environment.yaml** for system distribution configurations and for pthon packages, please check **requirements.txt**.  


#### Step 3 Checking datasets and arguments
- Download datasets (if needed) from [Training sets](https://www.dropbox.com/sh/ap2fy0560usui1c/AACnExI501GkZGYGpC2-GU9Da?dl=0) and [Testing sets](https://www.dropbox.com/scl/fo/d316nxhwc75wyvcoobl2i/h?dl=0&rlkey=va9jidmntohz7h0zzft0j7nky)
- Download pre-trained models for datasets provided above (if needed) from [model](https://www.dropbox.com/s/ps2pq15hr20fx9u/sim.pt?dl=0)
- Run command below to check the the arguments. It outputs a lsit arguments used to configure the training and testing experiments.
```{r, engine='python'}
     python EG.py --help
```

#### Step 4 Training and inference
- One example of training script. The learning rate and output path for the trained model can be modified in *TRAIN.LR* and *TRAIN.OUTPUT_PATH* under directory *new_code/src/config.py* . The configuration of network details such as the number of solver layers or Gumbel sinkhron samples can be modified under *experiments/ngm_qaplib.yaml*.
```{r, engine='python'}
    python EG.py --Train --exp-traind=~/TRAINING_DATA --exp-evald=~/TESTING_DATA --test-device=cuda
```

- One example of resume training script.
```{r, engine='python'}
    python EG.py --Resume-training --exp-traind=~/TRAINING_DATA --exp-evald=~/TESTING_DATA --test-device=cuda --weights-on-edges=1. --checkpoint-mod=~/MODEL_PATH --checkpoint-opt=~/OPTIMIZER_PATH
```

- One example of inference script. Inference script prints out a text file of all registrations, given a paired shape graph folder.
```{r, engine='python'}
    python EG.py --Inference --exp-model=./TRAINED_MODEL_PATH --exp-evald=./TESTING_DATA_PATH --exp-savep=./DIRECTORY_TO_SAVE_OUTPUTFILE --file-name=./NAME_OF _THE_TEXTFILE
```
- One example of computing the mean graph shape. Run first command to get all arguments of this script. Then run the second command and set all the arguments to your own preferences. Be aware for fast affinity matrix computation, one needs to embed a matlab engine. If speed is not the concern, please replace the matlab functions for computing $$K_p$$ and $$K_e$$ with functions defined in script *Shape/Statistics*
```{r, engine='python'}
    python compute_mean.py --help
```
```{r, engine='python'}
    python compute_mean.py --model-path='~/' --model-type='a string value' --I=int ...
```
#### Step 5 Inferences like graph geodescis and distance computation.
- **Please refer to [Guo](https://github.com/xiaoyangstat/statistical-shape-analysis-of-elastic-graphs.git) for the computations**.

## Acknowledgements
Our code is inspired by this repo [Wang](https://github.com/Thinklab-SJTU/ThinkMatch.git). Please cite Guo and Wang's work if use our code. 

## License

FSU

**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
