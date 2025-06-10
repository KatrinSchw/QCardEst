from abc import abstractmethod
from datetime import datetime
import pprint

from torch import Tensor
from .QML import QML
from torch.optim import SGD, Adam

from torch.optim.lr_scheduler import ConstantLR, StepLR
import torch


class GradientQML(QML):
    """
    Implements gradient base quantum machine learning
    """

    def __init__(self, agent, env, settings):
        self.prefix = ""
        super().__init__(agent, env, settings)
        # check for learning rate scheduler
        lrSettings = self.settings["lr"]
        if isinstance(lrSettings, list):
            lr = lrSettings[0]
        else:
            lr = lrSettings
        # chose optimizer
        if self.settings["optimizer"].lower() == "adam":
            self.optimizer = Adam(self.agent.model.parameters(), lr=lr, amsgrad=True)
        elif self.settings["optimizer"].lower() == "sgd":
            self.optimizer = SGD(self.agent.model.parameters(), lr=lr, momentum=0.9)
        else:
            self.optimizer = Adam(self.agent.model.parameters(), lr=lr)
        # create scheduler for decaying learning rate
        if isinstance(lrSettings, list):
            self.scheduler = StepLR(self.optimizer, step_size=lrSettings[1], gamma=lrSettings[2])
        else:
            self.scheduler = ConstantLR(self.optimizer, factor=1.0)

    def setModel(self, model):
        self.agent = model

    def run(self):
        self.preRun()

        # initialize variables for live evaluation
        logState = self.initLogState()

        # train the agent
        for episode in range(self.numEpisodes):

            loss = torch.zeros(1)
            for i in range(self.settings["batchsize"]):
                # learn a new state
                state, expected = self.env.step()
                prediction = self.agent(Tensor(state))
                closs, logState = self.interpret(state, expected, prediction, logState)
                # calculate loss
                loss += closs

            # backward calculation and optimizer step
            if (loss > 0):
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            self.printEpisode(episode, logState, loss)

            if (episode % self.logInterval) == 0 or episode == self.numEpisodes - 1:
                # log to results file
                stats = self.env.elvaluateModel(self.agent)
                self.resultFile.write(",".join(str(e) for e in ([episode] + stats)))
                self.resultFile.write("\n")
                self.resultFile.flush()

        # store generated model
        self.saveModel()
        self.end()
        print("End Time =", datetime.now().strftime("%H:%M:%S"))

    @abstractmethod
    def interpret(self, state, expected, prediction, logState):
        raise NotImplementedError("Abstract method should not be called")

    def initLogState(self):
        raise NotImplementedError("Abstract method should not be called")

    @abstractmethod
    def printEpisode(self, episode, logState, loss):
        pass