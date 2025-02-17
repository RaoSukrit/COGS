#!/usr/bin/env python
# -*- coding: utf-8 -*-
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from collections import defaultdict
from onmt.translate.swap_translator import build_swap_translator


def tpr_swap_translate(opt, out_file=None):
    logger = init_logger(opt.log_file)
    print(f"opt.src_feats={opt.src_feats}")

    # translator, model = build_swap_translator(opt, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size)

    features_shards = []
    features_names = []

    opt.src_feats = {}
    print(f"opt.src_feats={opt.src_feats}")
    for feat_name, feat_path in opt.src_feats.items():
        features_shards.append(split_corpus(feat_path, opt.shard_size))
        features_names.append(feat_name)
    shard_pairs = zip(src_shards, tgt_shards, *features_shards)

    for i, (src_shard, tgt_shard, *features_shard) in enumerate(shard_pairs):
        print(src_shard)
        features_shard_ = defaultdict(list)
        for j, x in enumerate(features_shard):
            features_shard_[features_names[j]] = x

        logger.info("Translating shard %d." % i)
        translator.translate(
            src=src_shard,
            src_feats=features_shard_,
            tgt=tgt_shard,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
            attn_debug=opt.attn_debug,
            align_debug=opt.align_debug,
        )

    return


def _get_parser():
    parser = ArgumentParser(description="trp_swap_translate.py")

    opts.config_opts(parser)
    opts.translate_opts(parser)
    opts.tpr_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    tpr_swap_translate(opt)


if __name__ == "__main__":
    main()
