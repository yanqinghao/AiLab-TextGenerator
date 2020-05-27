# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.app import app
from suanpan.docker.arguments import Folder, String
from suanpan.storage import StorageProxy


@app.param(String(key="models", default="prose", help="prose,grover"))
@app.param(String(key="storageType", default="oss"))
@app.output(Folder(key="outputModel"))
def SPModelLoader(context):
    args = context.args
    if args.storageType == "oss":
        storage = StorageProxy(None, None)
        storage.setBackend(type=args.storageType)
    else:
        from suanpan.storage import storage

    storage.download("common/model/gpt-2/" + args.models, args.outputModel)

    return args.outputModel


if __name__ == "__main__":
    SPModelLoader()  # pylint: disable=no-value-for-parameter
