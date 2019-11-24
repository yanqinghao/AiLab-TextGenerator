# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.app.arguments import String, Int, Folder, Bool
from suanpan.app import app
from utils.gpt2_generate import main


@app.input(Folder(key="model"))
@app.param(
    String(
        key="prefix",
        default="天空还是一片浅蓝，颜色很浅。转眼间天边出现了一道红霞，慢慢地在扩大它的范围，加强它的亮光。我知道太阳要从天边升起来了，便不转眼地望着那里。",
    )
)
@app.param(Int(key="nsamples", default=1))
@app.param(Bool(key="fastPattern", default=True))
@app.param(Int(key="length", default=100))
@app.output(Folder(key="outputData"))
def SPGPT2Generator(context):
    args = context.args
    main(
        args.prefix,
        nsamples=args.nsamples,
        model_path=args.model,
        fast_pattern=args.fastPattern,
        length=args.length,
        save_samples=True,
        save_samples_path=args.outputData,
    )
    return args.outputData


if __name__ == "__main__":
    SPGPT2Generator()
