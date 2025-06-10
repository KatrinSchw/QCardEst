import ast
import os


def loadBenchmark(filename):
    """
    Load a benchmark file into a list of dictionaries

    Each line in the benchmarks except comments (#) has the format:
    query ; time ; tree ; timeout ; timePostgreSQL
    with
    query : filename of the query
    time : total execution time of the database call
    tree : join tree in python list notation
    timeout : 0 = no timeout , 1 = timeout 
    timePostgreSQL : execution time of the query meassured by PostgreSQL EXPLAIN
    orderId: Id of this order in enumeration according to joinHelper.generateJoinOrders    
    """

    result = {}
    with open("benchmarks/" + filename, "r") as file:
        for line in file:
            # check if line is comment
            if line[0] == "#": continue
            # split into entries
            temp = line.split(";")
            # check if data format is correct
            if len(temp) < 5:
                raise ValueError("Invalid data format. 4 entries required in row but only {} found".format(len(temp)))
            # extract query name. Remove folder name and spaces
            query = temp[0].split("/")[-1].strip()
            if not query in result:
                result[query] = []
            result[query].append({"time": float(temp[1]), "tree": ast.literal_eval(temp[2].lstrip()), "timeout": temp[3] == "1", "timePG": float(temp[4]), "raw": line.rstrip()})
            if len(temp) >= 6:
                result[query][-1]["id"] = int(temp[5])
            # if only one entry found for query remove the list step
        for k, v in result.items():
            if len(v) == 1:
                result[k] = result[k][0]
    return result


def loadSelectivities(filename):
    result = {}
    with open(filename, "r") as file:
        for line in file:
            # Only 3 splits as condition may contain ","
            temp = line.strip().split(",", 3)
            condition = temp[3].strip()
            selectivity = temp[2]
            result[condition] = float(selectivity)
    return result


def parseQuery(query):
    # getting index of substrings
    idFrom = query.index("FROM")
    idWhere = query.index("WHERE")
    relString = query[idFrom + len("FROM") + 1:idWhere]
    rels = [x.strip() for x in relString.split(",")]
    filterString = query[idWhere + len("WHERE") + 1:-1]
    filters = [x.strip() for x in filterString.split("AND")]
    return rels, filters


def parseQueryFromFolder(folder, file):
    filename = "data/{}/{}".format(folder, file)
    with open(filename, 'r') as f:
        query = f.read().replace('\r', ' ').replace('\n', ' ')
    return parseQuery(query)


# TODO: fix for constants in condition
def relsFromCondition(cond):
    temp = cond.split("=")
    return [temp[0].split(".")[0], temp[1].split(".")[0]]


def queryToTableList(folder, file):
    rels, _ = parseQueryFromFolder(folder, file)
    return rels


def readQueriesFromDir(directory):
    result = {}
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a query file
        if f.endswith(".sql"):
            with open(f, 'r') as file:
                result[filename] = file.read()
    return result


def readQueriesFromFile(filename):
    result = {}
    count = 0
    with open(filename, 'r') as file:
        for line in file:
            result["query" + str(count)] = line
            count += 1
    return result


def readQueries(src: str):
    if src.endswith(".sql"):
        return readQueriesFromFile(src)
    else:
        return readQueriesFromDir(src)


# extract alias from relation field of query
# if query doesn't use aliases the name itself is returned
def extractAliases(rels):
    result = []
    for r in rels:
        if " AS " in r:
            result.append(r.split(" AS ")[1])
            continue
        if " " in r:
            result.append(r.split(" ")[1])
            continue
        result.append(r)
    return result


def extractTablesFromCardsCSVfile(file):
    if type(file) == str:
        file = open(file, "r")
    tables = set()
    maxQuerySize = 0
    for line in file:
        temp = line.split(",")
        query = temp[0].split(";")
        for t in query:
            tables.add(t.strip())
        maxQuerySize = max(maxQuerySize, len(query))
    return {"nTables": len(tables), "maxQuerySize": maxQuerySize, "tables": list(tables)}
