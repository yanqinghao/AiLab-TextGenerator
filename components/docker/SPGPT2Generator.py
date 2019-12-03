# coding=utf-8
from __future__ import absolute_import, print_function

import os
from suanpan.app.arguments import String, Int, Folder, Bool
from suanpan.app import app
from suanpan.utils import text
from utils.gpt2_generate import main
from scripts.interactive_conditional_samples import gpt2_ml_main
from utils.bosh_generator import make_bullshit, merge_bullshit


@app.input(Folder(key="model"))
@app.param(String(key="modelType", default="mixed", help="bullshit, gpt-2, mixed"))
@app.param(String(key="prefix", default="本市发生中重度污染过程，14时全市PM2.5浓度均值达到重度污染级别。",))
@app.param(Int(key="nsamples", default=2))
@app.param(Bool(key="fastPattern", default=True))
@app.param(Int(key="length", default=800))
@app.output(String(key="outputData"))
def SPGPT2Generator(context):
    args = context.args
    samples = []
    if args.modelType == "bullshit":
        samples += make_bullshit(args.prefix, args.length, args.nsamples)
    elif args.modelType == "gpt-2":
        filelist = os.listdir(args.model)
        if "config.json" in filelist:
            samples += main(
                args.prefix,
                nsamples=args.nsamples,
                model_path=args.model,
                fast_pattern=args.fastPattern,
                length=args.length,
                save_samples=False,
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
            samples += gpt2_ml_main(**params)
    else:
        samples_tmp = []
        if args.model is None:
            raise ("Need Input Model")
        else:
            filelist = os.listdir(args.model)
            if "config.json" in filelist:
                samples_tmp += main(
                    args.prefix,
                    nsamples=args.nsamples,
                    model_path=args.model,
                    fast_pattern=args.fastPattern,
                    length=args.length,
                    save_samples=False,
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
                samples_tmp += gpt2_ml_main(**params)
            samples += merge_bullshit(samples_tmp, args.length)
    output = ""
    for i in range(args.nsamples):
        output += "\n Sample,{} of {} \n".format(i + 1, args.nsamples)
        output += samples[i]
    return output


if __name__ == "__main__":
    SPGPT2Generator()
