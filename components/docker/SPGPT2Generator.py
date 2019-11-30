# coding=utf-8
from __future__ import absolute_import, print_function

import os
from suanpan.app.arguments import String, Int, Folder, Bool
from suanpan.app import app
from suanpan.utils import text
from utils.gpt2_generate import main
from scripts.interactive_conditional_samples import gpt2_ml_main


@app.input(Folder(key="model"))
@app.param(Bool(key="modelType", default=False))
@app.param(
    String(
        key="prefix",
        default="本市发生中重度污染过程，14时全市PM2.5浓度均值达到重度污染级别。",
        help="天空还是一片浅蓝，颜色很浅。转眼间天边出现了一道红霞，慢慢地在扩大它的范围，加强它的亮光。我知道太阳要从天边升起来了，便不转眼地望着那里。",
    )
)
@app.param(Int(key="nsamples", default=1))
@app.param(Bool(key="fastPattern", default=True))
@app.param(Int(key="length", default=100))
@app.output(Folder(key="outputData"))
def SPGPT2Generator(context):
    args = context.args
    if args.modelType:
        main(
            args.prefix,
            nsamples=args.nsamples,
            model_path=args.model,
            fast_pattern=args.fastPattern,
            length=args.length,
            save_samples=True,
            save_samples_path=args.outputData,
        )
    else:
        params = {
            "input_text": args.prefix,
            "model_config_fn": "configs/mega.json",
            "max_batch_size": None,
            "batch_size": 1,
            "top_p": 0.95,
            "model_ckpt": os.path.join(args.model, "model.ckpt-100000"),
            "samples": args.nsamples,
            "eos_token": 511,
            "min_len": args.length,
        }
        result = gpt2_ml_main(**params)
        text.dump(result, os.path.join(args.outputData, "data.txt"))
    return args.outputData


if __name__ == "__main__":
    SPGPT2Generator()
