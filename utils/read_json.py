# coding=utf-8
from __future__ import absolute_import, print_function

import json


def reader(fileName=""):

    if fileName != "":
        strList = fileName.split(".")
        if strList[len(strList) - 1].lower() == "json":
            with open(fileName, mode="r", encoding="utf-8") as file:
                return json.loads(file.read())
