import random
import sys
import ast

from datetime import datetime

import numpy as np
import torch
from ML.models import vqc
from ML.Regression import Regression
from cardEnv import CardEnv
from ML import regressionInterpretation as interpretation

settings = {
    "features": "simple",
    "encoding": ["rx", "rz"],
    "reuploading": False,
    "reps": 6,
    "calc": "yz",
    "entangleType": "circular",
    "entangle": "cx",
    "reward": "rational",
    "numEpisodes": 1,
    "optimizer": "Adam",
    "lr": [0.01, 100, 0.9],
    "prefix": "Constant",
    "noise": 0,
    "seed": 42,
    "batchsize": 1,
    "loss": "constant",
    "data": "jobSimple/job",
    #"data" : "stats/statsCards6",
    #"data" : "mscnCosts",
    "valueType": "rowFactor",
    "numFeatures": 6
}

# update settings with command line option
if len(sys.argv) > 1:
    settings.update(ast.literal_eval(sys.argv[1]))

# init random
random.seed(settings["seed"])
np.random.seed(settings["seed"] + 1)
torch.manual_seed(settings["seed"] + 2)

start_time = datetime.now()

env = CardEnv(
    inputFile=settings["data"] + ".csv",
    settings=settings)

print("Size Input : {}, Output: {}".format(env.getInputSize(), env.getOutputSize()))

agent = interpretation.fromString(settings["loss"], None, env)

print("input size", env.getInputSize())

# use quantum model
model = vqc(settings=settings, nInputs=env.getInputSize(), nOutputs=agent.nInputs, norm=False)
agent.setModel(model)
optimizer = Regression(agent, env, settings)

# run the optimization
optimizer.run()

end_time = datetime.now()
print("Duration =", end_time - start_time)

optimizer.listSolutions()
exit()
