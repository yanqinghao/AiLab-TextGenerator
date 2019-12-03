import sys
import os
import re
import tensorflow as tf
import numpy as np
from train.modeling import GroverModel, GroverConfig, sample
from tokenization import tokenization
from suanpan.log import logger


def extract_generated_target(output_tokens, tokenizer):
    """
    Given some tokens that were generated, extract the target
    :param output_tokens: [num_tokens] thing that was generated
    :param encoder: how they were encoded
    :param target: the piece of metadata we wanted to generate!
    :return:
    """
    # Filter out first instance of start token
    assert output_tokens.ndim == 1

    start_ind = 0
    end_ind = output_tokens.shape[0]

    return {
        "extraction": tokenization.printable_text(
            "".join(tokenizer.convert_ids_to_tokens(output_tokens))
        ),
        "start_ind": start_ind,
        "end_ind": end_ind,
    }


def gpt2_ml_main(**kwargs):
    input_text_args = kwargs.pop("input_text")
    model_config_fn_args = kwargs.pop("model_config_fn")
    max_batch_size_args = kwargs.pop("max_batch_size")
    batch_size_args = kwargs.pop("batch_size")
    top_p_args = kwargs.pop("top_p")
    model_ckpt_args = kwargs.pop("model_ckpt")
    samples_args = kwargs.pop("samples")
    eos_token_args = kwargs.pop("eos_token")
    min_len_args = kwargs.pop("min_len")

    proj_root_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    vocab_file_path = os.path.join(
        proj_root_path, "tokenization", "bert-base-chinese-vocab.txt"
    )
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file_path, do_lower_case=True
    )
    news_config = GroverConfig.from_json_file(model_config_fn_args)

    # We might have to split the batch into multiple chunks if the batch size is too large
    default_mbs = {12: 32, 24: 16, 48: 3}
    max_batch_size = (
        max_batch_size_args
        if max_batch_size_args is not None
        else default_mbs[news_config.num_hidden_layers]
    )

    # factorize args.batch_size = (num_chunks * batch_size_per_chunk) s.t. batch_size_per_chunk < max_batch_size
    num_chunks = int(np.ceil(batch_size_args / max_batch_size))
    batch_size_per_chunk = int(np.ceil(batch_size_args / num_chunks))

    # This controls the top p for each generation.
    top_p = np.ones((num_chunks, batch_size_per_chunk), dtype=np.float32) * top_p_args

    tf_config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
        initial_context = tf.placeholder(tf.int32, [batch_size_per_chunk, None])
        p_for_topp = tf.placeholder(tf.float32, [batch_size_per_chunk])
        eos_token = tf.placeholder(tf.int32, [])
        min_len = tf.placeholder(tf.int32, [])
        tokens, probs = sample(
            news_config=news_config,
            initial_context=initial_context,
            eos_token=eos_token,
            min_len=min_len,
            ignore_ids=None,
            p_for_topp=p_for_topp,
            do_topk=False,
        )

        saver = tf.train.Saver()
        saver.restore(sess, model_ckpt_args)
        output = []
        for i in range(samples_args):
            text = input_text_args
            logger.info("Sample,{} of {}".format(i + 1, samples_args))
            line = tokenization.convert_to_unicode(text)
            bert_tokens = tokenizer.tokenize(line)
            encoded = tokenizer.convert_tokens_to_ids(bert_tokens)
            context_formatted = []
            context_formatted.extend(encoded)
            # Format context end

            gens = []
            gens_raw = []
            gen_probs = []

            for chunk_i in range(num_chunks):
                tokens_out, probs_out = sess.run(
                    [tokens, probs],
                    feed_dict={
                        initial_context: [context_formatted] * batch_size_per_chunk,
                        eos_token: eos_token_args,
                        min_len: min_len_args,
                        p_for_topp: top_p[chunk_i],
                    },
                )

                for t_i, p_i in zip(tokens_out, probs_out):
                    extraction = extract_generated_target(
                        output_tokens=t_i, tokenizer=tokenizer
                    )
                    gens.append(extraction["extraction"])

            l = re.findall(
                ".{1,70}", gens[0].replace("[UNK]", "").replace("##", "")
            )
            sample_text = "\n".join(
                ["Sample,{} of {}".format(i + 1, samples_args)] + l
            )
            output.append("\n".join(l))
        return output
