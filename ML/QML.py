from abc import ABC, abstractmethod
from datetime import datetime
import pprint
import torch

from ML.interpretation import Interpretation
from queryEnv import QueryEnv


class QML(ABC):
    """
    Parent class for quantum machine learning
    """

    # default settings
    settings = {"features": "simple", "encoding": "rx", "reuploading": False, "reps": 5, "calc": "yz", "entangleType": "circular", "entangle": "cx", "reward": "rational", "numEpisodes": 40, "optimizer": "sgd", "lr": 0.001, "prefix": "test", "seed": 42}

    agent: Interpretation
    "Agent to interpret probability vector into classification or regression result"
    env: QueryEnv
    "Environment containing data an evaluation"

    def __init__(self, agent, env, settings):
        self.settings.update(settings)
        self.agent = agent
        self.env = env

        self.logInterval = 100
        self.numEpisodes = self.settings["numEpisodes"]

        # logging files
        self.prefix = self.settings["prefix"] + self.prefix
        self.baseDir = "results"
        self.outputFilename = self.prefix + "_" + "_".join(str(e) for e in self.settings.values())
        # Replace problematic chars
        self.outputFilename = self.outputFilename.replace("/", "-").replace(" ", "-")

    def generateFilename(self, folder: str, postfix: str) -> str:
        """
        Generate a filename from settings. Used to store result in unique files
        """
        return self.baseDir + "/" + folder + "/" + self.outputFilename + "." + postfix

    def preRun(self):
        """
        Initialization before optimization loop
        """
        print("Start Time =", datetime.now().strftime("%H:%M:%S"))
        print("Settings: ")
        pprint.pprint(self.settings)

        # print settings to file
        with open(self.generateFilename("settings", "conf"), "w") as settingsLog:
            pprint.pprint(self.settings, settingsLog)

        # open result file
        self.resultFile = open(self.generateFilename(".", "csv"), "w")
        self.resultFile.write(self.env.resultHeader())

    @abstractmethod
    def run(self):
        pass

    def listSolutions(self):
        file = open(self.generateFilename("solutions", "sl.csv"), "w")
        solutions = self.env.listSolutions(self.agent)
        for entry in solutions:
            file.write(",".join(str(e) for e in entry) + "\n")

    def end(self):
        self.resultFile.close()

    def saveModel(self):
        torch.save(self.agent.model.state_dict(), self.generateFilename("models", "model"))

    def loadModel(self):
        self.agent.model.load_state_dict(torch.load(self.generateFilename("models", "model")))
