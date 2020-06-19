# 2048-api
# Code structure
* [`game2048/`](game2048/): the main package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): the `Agent` class with instances, is a strong agent based on rule.
    * [`displays.py`](game2048/displays.py): the `Display` class with instances, to show the `Game` state.
    * [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai)
    # My Design
    * ['model1/'](game2048/model1/): training code for model1 (online) 
        * ['dataloader.py'](game2048/model1/dataloader.py): dataloader for my model1
        * ['models.py](game2048/model1/models.py): my Neural Network Model based on Vgg
        * ['vgg.py](game2048/model1/vgg.py): vgg model
        * ['utils.py'](game2048/model1/utils.py): a tool for batchsize
        * ['train.py](game2048/model1/train.py): my code of online imitation training
    * ['model2/'](game2048/model2/): training code for model2
        * ['model.py'](game2048/model2/model.py): my Neural Network based on CNN
        * ['train.py'](game2048/model2/train.py): my code of CNN training
    * ['model3/'](game2048/model3/): training code for model3 & model4
        * ['CNN_4096.py'](game2048/model3/CNN.py): include Network Model and training code for model3 & model4
    * ['model5/'](game2048/model5/): training code for model5
        * ['CNN_deep.py'](game2048/model5/CNN_deep.py): include Network Model and training code for model5
    
    # agents
    * ['agent_voting.py']:(game2048/agents_voting.py): my design agent, load five models and voting to get direction
    # params
    * ['model1_pretrain_params.pkl']:(game2048/model1_pretrain_params.pkl): pre-train model for model1
    * ['model2_pretrain_params.pkl']:(game2048/model2_pretrain_params.pkl): pre-train model for model2
    * ['model3_pretrain_params.pkl']:(game2048/model3_pretrain_params.pkl): pre-train model for model3
    * ['model3_pretrain_params_2.pkl']:(game2048/model3_pretrain_params_2.pkl): pre-train model for model4
    * ['model5_pretrain_params.pkl']:(game2048/model5_pretrain_params.pkl): pre-train model for model5
    * For privacy and github limitation, model5 params haven't been uploaded.

* [`explore.ipynb`](explore.ipynb): introduce how to use the `Agent`, `Display` and `Game`.
* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`evaluate.py`](evaluate.py): evaluate your self-defined agent.
# evaluate_voting
* ['evaluate_voting.py']:(evaluate_voting.py): evaluate my voting agent

# Requirements
* code only tested on linux system (ubuntu 16.04)
* Python 3 (Anaconda 3.6.3 specifically) with numpy and flask
* Pytorch 0.4.0
* numpy 1.14.5

# To compile the pre-defined ExpectiMax agent

```bash
cd game2048/expectimax
bash configure
make
```

# To run the web app
```bash
python webapp.py
```
![demo](preview2048.gif)

# LICENSE
The code is under Apache-2.0 License.

# For EE369 students from SJTU only
Please read [here](EE369.md).
