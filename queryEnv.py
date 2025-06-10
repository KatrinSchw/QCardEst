from abc import ABC, abstractmethod
from math import pi
import random

from ML.interpretation import Interpretation

class QueryEnv(ABC):
    """
    """

    settings = {"numFeatures": -1}

    def __init__(self, inputFile="", settings={}):
        self.settings.update(settings)
        self.data = []
        self.current_query = 0
        self.featureType = self.settings["features"]
        self.numFeatures = self.settings["numFeatures"]
        self.includeSelectivities = type(self.settings["encoding"]) is list and len(self.settings["encoding"]) > 1
        self.__evaluationSet = None

    @abstractmethod
    def expected(self, entry):
        pass

    @abstractmethod
    def elvaluateModel(self, model: Interpretation, evaluationSet=None) -> list:
        pass

    @abstractmethod
    def listSolutions(self, model) -> list:
        pass

    def step(self):
        """
        Select a random query from the dataset and return the feature vector and expected value 
        """
        self.current_query = random.randrange(0, len(self.data))
        datapoint = self.data[self.current_query]
        observation = datapoint["features"]
        return observation, self.expected(datapoint)

    def reset(self) -> list[float]:
        """Start/Restart by selecting a random query and returning it"""
        self.current_query = random.randrange(0, len(self.data))
        return self.data[self.current_query]["features"]

    def newEvaluationSet(self, size=20):
        self.__evaluationSet = random.sample(self.data, size)

# --------------------------------- Getter / Setter -------------------------------------------

    def getInputSize(self):
        #  TODO
        return 4
        #return len(self.data[0]["features"])

    def getEvaluationSet(self):
        if self.__evaluationSet is None:
            self.newEvaluationSet()
        return self.__evaluationSet

    def setRewardType(self, type):
        self.rewardType = type

    def setFeatureType(self, type):
        self.featureType = type

    def setMaxId(self, id):
        self.maxId = id
        self.shuffleArray = list(range(id + 1))
        random.shuffle(self.shuffleArray)

    def findEntryIdFeatures(self, ids):
        idSet = set(ids)
        for entry in self.data:
            if set(entry["features_raw"]) == idSet:
                return entry
        raise ValueError("No entry found for " + str(ids))

    def resultHeader(self):
        return ""


# ----------------------- Feature mapping -----------------------

    def feature_map(self, features: list[int]) -> list[float]:
        """
        Turns a list of ids into values suitable for angle encoding

        Maps the values in `features` from the interval
            [0,n]
        to the interval 
            [0,pi]

        Optionally extends/modifies the features      
        """
        if self.featureType == "double":
            return self.featureMapDouble(features)
        elif self.featureType == "shuffle":
            return self.featureMapDoubleShuffle(features)
        else:
            return self.featureMapSimple(features)

    def normalize(self, value):
        maxvalue = self.maxId
        newmax = pi
        return value * newmax / maxvalue

    def featureMapSimple(self, features):
        result = []
        for f in features:
            if type(f) is list:
                result.append(self.normalize(f[0]))
                result.append(f[1] * pi)
            else:
                result.append(self.normalize(f))
        return result

    def featureMapDouble(self, features):
        result = []
        for f in features:
            result.append(self.normalize(f))
        for f in features:
            result.append(self.normalize(f))
        return result

    def featureMapDoubleShuffle(self, features):
        result = []
        for f in features:
            result.append(self.normalize(f))
        for f in features:
            result.append(self.normalize(self.shuffleArray[f]))
        return result

    def featureMapBinary(self, features):
        result = [0.0] * (self.maxId + 1)
        for f in features:
            result[f] = pi
        return result
