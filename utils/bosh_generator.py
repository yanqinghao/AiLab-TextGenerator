# coding=utf-8
from __future__ import absolute_import, print_function

import sys, os, re
import random
from utils import read_json

data = read_json.reader(
    os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "utils", "data.json")
)
famous = data["famous"]  # a 代表前面垫话，b代表后面垫话
before = data["before"]  # 在名人名言前面弄点废话
after = data["after"]  # 在名人名言后面弄点废话
bosh = data["bosh"]  # 代表文章主要废话来源

repeat = 2


def shuffle(lists):
    global repeat
    pool = list(lists) * repeat
    while True:
        random.shuffle(pool)
        for ele in pool:
            yield ele


bosh_next = shuffle(bosh)
famous_next = shuffle(famous)


def make_famous():
    global famous_next
    line = next(famous_next)
    line = line.replace("a", random.choice(before))
    line = line.replace("b", random.choice(after))
    return line


def make_paragraph():
    line = "。"
    line += "\r\n"
    line += "    "
    return line


def make_bullshit(topic, length, samples):
    output = []
    for i in range(samples):
        tmp = str()
        while len(tmp) < length:
            branch = random.randint(0, 100)
            if branch < 5:
                tmp += make_paragraph()
            elif branch < 20:
                tmp += make_famous()
            else:
                tmp += next(bosh_next)
        tmp = tmp.replace("x", topic)
        output.append(tmp)
    return output


def merge_bullshit(samples, length):
    output = []
    for text in samples:
        topics = []
        for i in re.split("[\n。]", text):
            if i.endswith("。"):
                topics.append(i)
            else:
                topics.append(i + "。")
        tmp = str()
        while len(tmp) < length:
            branch = random.randint(0, 100)
            if branch < 5:
                tmp += make_paragraph()
            elif branch < 20:
                tmp += make_famous()
            else:
                tmp += next(bosh_next)
        tmp = re.split("x", tmp)
        tmp_res = []
        for i, j in enumerate(tmp):
            if i == len(tmp)-1:
                tmp_res.append(j)
            else:
                tmp_res.append(j)
                if i <= len(topics)-1:
                    tmp_res.append(topics[i].split("。")[0])
                else:
                    tmp_res.append(topics[0].split("。")[0])
        output.append("".join(tmp_res))
    return output
