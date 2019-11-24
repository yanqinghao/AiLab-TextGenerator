# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.app.arguments import String
from suanpan.app import app
from utils.bosh_generator import make_bullshit


@app.param(String(key="topic", default="中午吃什么"))
@app.output(String(key="outputData"))
def SPBullshitGenerator(context):
    args = context.args
    return make_bullshit(args.topic)


if __name__ == "__main__":
    SPBullshitGenerator()
