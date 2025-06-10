from collections import deque

from .GradientQML import GradientQML


class Regression(GradientQML):
    """
    Implements Regression using a VQC
    """

    def initLogState(self):
        return deque(maxlen=40)

    def interpret(self, state, expected, prediction, logState):
        #value = self.agent.predict(prediction,len(expected))

        loss = self.agent.loss(prediction, expected)
        logState.append(loss)

        return loss, logState

    def printEpisode(self, episode, logState, loss):
        averageLoss = sum(logState) / len(logState)
        print("Episode: {}, loss: {:.3f}, Average Loss : {:.3f}, LR : {:.4f}".format(episode, loss.item(), averageLoss.item(), self.scheduler.get_last_lr()[0]), end="\n")
