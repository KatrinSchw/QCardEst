import math
import pprint
import statistics as stat

from torch import Tensor
import torch
from ML.regressionInterpretation import Interpretation
from queryEnv import QueryEnv

from util.dataHelper import extractTablesFromCardsCSVfile


class CardEnv(QueryEnv):
    """
    Class representing a benchmark of joins
    """

    settings = {
        "features": "simple",
        "valueType": "rows",
    }

    def __init__(self, inputFile="", settings={}):
        super().__init__(inputFile, settings)
        self.valueType = self.settings["valueType"]
        if inputFile != "":
            # load data
            with open("costs/" + inputFile, "r") as input:
                self.load_data(input)

    def expected(self, entry):
        return self.expectedWithType(entry, self.valueType)

    def expectedWithType(self, entry, type):
        if type in ["rowFactor", "rowsFactor"]:
            return self.expectedMapping(entry["rows"] / entry["rowsPredicted"])
        if type in ["costFactor", "costsFactor"]:
            return self.expectedMapping(entry["cost"] / entry["costPredicted"])
        return self.expectedMapping(entry[type])

    def expectedMapping(self, value):
        return math.log(value)

    def maxExpected(self):
        return self.expected(max(self.data, key=lambda e: self.expected(e)))

    def load_data(self, file):
        queryInfo = extractTablesFromCardsCSVfile(file)
        self.setMaxId(queryInfo["nTables"] - 1)
        self.numFeatures = queryInfo["maxQuerySize"]
        id = 0
        file.seek(0)
        for entry in file:
            tmp = entry.split(",")
            # extract features
            query = tmp[0].split(";")
            features = []
            if self.includeSelectivities:
                if self.numFeatures == -1:
                    self.numFeatures = len(query)
                selectivities = tmp[1].split(";")
                if len(query) != len(selectivities):
                    raise ValueError("Number of tables and number of selectivities have to be equal, but are " + str(len(query)) + " and " + str(len(selectivities)))
                for t in zip(query, selectivities):
                    features.append([queryInfo["tables"].index(t[0]), float(t[1])])
                while len(features) < self.numFeatures:
                    features.append([-self.maxId, 1.0])
                values = tmp[2:]
            else:
                for t in zip(query):
                    features.append(queryInfo["tables"].index(t))
                while len(features) < self.numFeatures:
                    features.append(-self.maxId)
                values = tmp[1:]

            datapoint = {"id": id, "features_raw": features, "features": self.feature_map(features), "rows": int(values[3]), "cost": float(values[2]), "rowsPredicted": int(values[1]), "costPredicted": float(values[0])}
            self.data.append(datapoint)
            id += 1

    def evaluate(self, state, action):
        for entry in self.data:
            diff = abs(sum(state - entry["features"]))
            if diff < 0.01:
                return entry["values"][action]
        print(state, " not found")

    def elvaluateModel(self, model: Interpretation, evaluationSet=None):
        if evaluationSet is None:
            evaluationSet = self.data
        with torch.no_grad():
            # Multi thread evaluation
            if self.includeSelectivities:
                tensor = torch.zeros(0, self.numFeatures * 2)
            else:
                tensor = torch.zeros(0, self.numFeatures)
            for entry in evaluationSet:
                tensor = torch.cat((tensor, torch.reshape(torch.tensor(entry["features"]), [1, len(entry["features"])])), 0)
            # calculate predictions
            with torch.no_grad():
                predictions = model.model(tensor)
            print("Prediction shape", predictions.size())

            diffs = []
            factors = []
            cards = []
            diffCards = []
            factorCards = []
            for i, prediction in enumerate(predictions):
                entry = evaluationSet[i]
                value = model.predict(prediction).item()
                expected = self.expected(entry)
                diffs.append(abs(value - expected))
                if expected == 0:
                    expected += 0.001
                if value == 0:
                    value += 0.001
                factor = abs(value / expected)
                if factor < 1 and factor != 0:
                    factor = 1 / factor
                factors.append(factor)

                # Cards
                if "Factor" in self.valueType:
                    correction = math.exp(value)
                    card = math.log(correction * entry["rowsPredicted"])
                else:
                    card = value
                expectedCard = math.log(entry["rows"])
                diffCards.append(abs(card - expectedCard))
                if card == 0:
                    card = 1
                factor = abs(expectedCard / card)
                if factor < 1 and factor != 0:
                    factor = 1 / factor
                factorCards.append(factor)

            # number of differences below 0.1
            close = sum(1 for i in factors if i < 1.1)
        return [stat.mean(diffs), stat.median(diffs), stat.variance(diffs), stat.mean(factors), stat.median(factors), stat.variance(factors), stat.mean(diffCards), stat.median(diffCards), stat.variance(diffCards), stat.mean(factorCards), stat.median(factorCards), stat.variance(factorCards), close]

    def listSolutions(self, model: Interpretation) -> list:
        result = []
        result.append(["id", "prediction", "expected", "factor", "loss"])
        with torch.no_grad():
            for index, entry in enumerate(self.data):
                state = Tensor(entry["features"])
                temp = model(state)
                prediction = model.predict(temp)
                expected = self.expected(entry)
                loss = model.loss(temp, expected)
                factor = abs(prediction / expected)
                if factor < 1 and factor != 0:
                    factor = 1 / factor
                result.append([index, prediction.item(), expected, factor.item(), loss.item()])
        return result

    def printData(self):
        for entry in self.data:
            print("Rows", entry["rows"], "/", entry["rowsPredicted"], "=", entry["rows"] / entry["rowsPredicted"])


# --------------------------------- Getter / Setter -------------------------------------------

    def getInputSize(self):
        return self.numFeatures

    def getOutputSize(self):
        return 2

    def getEvaluationSet(self):
        if self.__evaluationSet is None:
            self.newEvaluationSet()
        return self.__evaluationSet

    def findEntryIdFeatures(self, ids):
        idSet = set(ids)
        for entry in self.data:
            if set(entry["features_raw"]) == idSet:
                return entry
        raise ValueError("No entry found for " + str(ids))
