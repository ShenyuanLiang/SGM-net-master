# For training:

1) Update config.py to point to the correct locations of dataset and results

2) Run:

$ python main.py


# For inference:

1) Update the model name in main_inf.py and main_horn.py in the following line:

model.load_state_dict(torch.load(os.path.join(cfg.TRAIN.OUTPUT_PATH,'models','params_0026.pt')))

2) Run:

$ python main_inf.py

or:

$ python main_horn.py

The first command runs the inference without alignment, and the second one runs the inference twice, the first one without alignment and the second using Horn's approach to align corresponding nodes.

