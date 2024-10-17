"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""

import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.modules
from onmt.encoders import str2enc

from onmt.decoders import str2dec

from onmt.modules import Embeddings, CopyGenerator
from onmt.modules.util_class import Cast
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
from onmt.utils.parse import ArgumentParser
from onmt.constants import ModelTask


def build_embeddings(opt, text_field, for_encoder=True):
    """
    Args:
        opt: the option in current environment.
        text_field(TextMultiField): word and feats field.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    emb_dim = opt.src_word_vec_size if for_encoder else opt.tgt_word_vec_size

    pad_indices = [f.vocab.stoi[f.pad_token] for _, f in text_field]
    word_padding_idx, feat_pad_indices = pad_indices[0], pad_indices[1:]

    num_embs = [len(f.vocab) for _, f in text_field]
    num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]

    freeze_word_vecs = (
        opt.freeze_word_vecs_enc if for_encoder else opt.freeze_word_vecs_dec
    )

    emb = Embeddings(
        word_vec_size=emb_dim,
        position_encoding=opt.position_encoding,
        feat_merge=opt.feat_merge,
        feat_vec_exponent=opt.feat_vec_exponent,
        feat_vec_size=opt.feat_vec_size,
        dropout=opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
        word_padding_idx=word_padding_idx,
        feat_padding_idx=feat_pad_indices,
        word_vocab_size=num_word_embeddings,
        feat_vocab_sizes=num_feat_embeddings,
        sparse=opt.optim == "sparseadam",
        freeze_word_vecs=freeze_word_vecs,
    )
    return emb


def build_encoder(opt, embeddings=None, encoder_type=None):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    # TPR CHANGE
    if encoder_type == "tpr":
        enc_type = encoder_type
    else:
        enc_type = opt.encoder_type if opt.model_type == "text" else opt.model_type

    # enc_type = opt.encoder_type if opt.model_type == "text" else opt.model_type
    return str2enc[enc_type].from_opt(opt, embeddings)


def build_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    dec_type = (
        "ifrnn" if opt.decoder_type == "rnn" and opt.input_feed else opt.decoder_type
    )
    return str2dec[dec_type].from_opt(opt, embeddings)


def load_test_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    fields = checkpoint["vocab"]

    # Avoid functionality on inference
    model_opt.update_vocab = False

    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint, opt.gpu)
    if opt.fp32:
        model.float()
    elif opt.int8:
        if opt.gpu >= 0:
            raise ValueError("Dynamic 8-bit quantization is not supported on GPU")
        torch.quantization.quantize_dynamic(model, inplace=True)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def build_src_emb(model_opt, fields):
    # Build embeddings.
    if model_opt.model_type == "text":
        src_field = fields["src"]
        src_emb = build_embeddings(model_opt, src_field)
    else:
        src_emb = None
    return src_emb


def build_encoder_with_embeddings(model_opt, fields):
    # Build encoder.
    src_emb = build_src_emb(model_opt, fields)
    encoder = build_encoder(model_opt, src_emb)
    return encoder, src_emb


def build_decoder_with_embeddings(
    model_opt, fields, share_embeddings=False, src_emb=None
):
    # Build embeddings.
    tgt_field = fields["tgt"]
    tgt_emb = build_embeddings(model_opt, tgt_field, for_encoder=False)

    if share_embeddings:
        tgt_emb.word_lut.weight = src_emb.word_lut.weight

    # Build decoder.
    decoder = build_decoder(model_opt, tgt_emb)
    return decoder, tgt_emb


def build_task_specific_model(model_opt, fields):
    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        assert (
            fields["src"].base_field.vocab == fields["tgt"].base_field.vocab
        ), "preprocess with -share_vocab if you use share_embeddings"

    if model_opt.model_task == ModelTask.SEQ2SEQ:
        encoder, src_emb = build_encoder_with_embeddings(model_opt, fields)
        decoder, _ = build_decoder_with_embeddings(
            model_opt,
            fields,
            share_embeddings=model_opt.share_embeddings,
            src_emb=src_emb,
        )
        return onmt.models.NMTModel(encoder=encoder, decoder=decoder)
    elif model_opt.model_task == ModelTask.LANGUAGE_MODEL:
        src_emb = build_src_emb(model_opt, fields)
        decoder, _ = build_decoder_with_embeddings(
            model_opt, fields, share_embeddings=True, src_emb=src_emb
        )
        return onmt.models.LanguageModel(decoder=decoder)
    else:
        raise ValueError(f"No model defined for {model_opt.model_task} task")


def use_embeddings_from_checkpoint(fields, model, generator, checkpoint):
    # Update vocabulary embeddings with checkpoint embeddings
    logger.info("Updating vocabulary embeddings with checkpoint embeddings")
    # Embedding layers
    enc_emb_name = "encoder.embeddings.make_embedding.emb_luts.0.weight"
    dec_emb_name = "decoder.embeddings.make_embedding.emb_luts.0.weight"

    for field_name, emb_name in [("src", enc_emb_name), ("tgt", dec_emb_name)]:
        if emb_name not in checkpoint["model"]:
            continue
        multifield = fields[field_name]
        checkpoint_multifield = checkpoint["vocab"][field_name]
        for (name, field), (checkpoint_name, checkpoint_field) in zip(
            multifield, checkpoint_multifield
        ):
            new_tokens = []
            for i, tok in enumerate(field.vocab.itos):
                if tok in checkpoint_field.vocab.stoi:
                    old_i = checkpoint_field.vocab.stoi[tok]
                    model.state_dict()[emb_name][i] = checkpoint["model"][emb_name][
                        old_i
                    ]
                    if field_name == "tgt":
                        generator.state_dict()["0.weight"][i] = checkpoint["generator"][
                            "0.weight"
                        ][old_i]
                        generator.state_dict()["0.bias"][i] = checkpoint["generator"][
                            "0.bias"
                        ][old_i]
                else:
                    # Just for debugging purposes
                    new_tokens.append(tok)
            logger.info("%s: %d new tokens" % (name, len(new_tokens)))
        # Remove old vocabulary associated embeddings
        del checkpoint["model"][emb_name]
    del checkpoint["generator"]["0.weight"], checkpoint["generator"]["0.bias"]


def build_base_model(model_opt, fields, gpu, checkpoint=None, gpu_id=None):
    """Build a model from opts.

    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        fields (dict[str, torchtext.data.Field]):
            `Field` objects for the model.
        gpu (bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
        gpu_id (int or NoneType): Which GPU to use.

    Returns:
        the NMTModel.
    """

    # for back compat when attention_dropout was not defined
    try:
        model_opt.attention_dropout
    except AttributeError:
        model_opt.attention_dropout = model_opt.dropout

    # Build Model
    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")

    model = build_task_specific_model(model_opt, fields)

    # Build Generator.
    if not model_opt.copy_attn:
        if model_opt.generator_function == "sparsemax":
            gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(
            nn.Linear(model_opt.dec_rnn_size, len(fields["tgt"].base_field.vocab)),
            Cast(torch.float32),
            gen_func,
        )
        if model_opt.share_decoder_embeddings:
            generator[0].weight = model.decoder.embeddings.word_lut.weight
    else:
        tgt_base_field = fields["tgt"].base_field
        vocab_size = len(tgt_base_field.vocab)
        pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
        generator = CopyGenerator(model_opt.dec_rnn_size, vocab_size, pad_idx)
        if model_opt.share_decoder_embeddings:
            generator.linear.weight = model.decoder.embeddings.word_lut.weight

    # Load the model states from checkpoint or initialize them.
    if checkpoint is None or model_opt.update_vocab:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(model, "encoder") and hasattr(model.encoder, "embeddings"):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc
            )
        if hasattr(model.decoder, "embeddings"):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec
            )

    if checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r"(.*)\.layer_norm((_\d+)?)\.b_2", r"\1.layer_norm\2.bias", s)
            s = re.sub(r"(.*)\.layer_norm((_\d+)?)\.a_2", r"\1.layer_norm\2.weight", s)
            return s

        checkpoint["model"] = {fix_key(k): v for k, v in checkpoint["model"].items()}
        # end of patch for backward compatibility

        if model_opt.update_vocab:
            # Update model embeddings with those from the checkpoint
            # after initialization
            use_embeddings_from_checkpoint(fields, model, generator, checkpoint)

        model.load_state_dict(checkpoint["model"], strict=False)
        generator.load_state_dict(checkpoint["generator"], strict=False)

    model.generator = generator

    if model_opt.freeze_encoder:
        model.encoder.requires_grad_(False)
        model.encoder.embeddings.requires_grad_()

    if model_opt.freeze_decoder:
        model.decoder.requires_grad_(False)
        model.decoder.embeddings.requires_grad_()

    model.to(device)
    if model_opt.model_dtype == "fp16" and model_opt.optim == "fusedadam":
        model.half()
    return model


def load_tpr_test_model(opt, tpr_model_path=None, nmt_model_path=None):
    if tpr_model_path is None:
        tpr_model_path = opt.tpr_ckpt

    tpr_checkpoint = torch.load(
        tpr_model_path, map_location=lambda storage, loc: storage
    )

    if nmt_model_path is None:
        nmt_model_path = opt.nmt_model[0]

    nmt_model_ckpt = torch.load(
        nmt_model_path, map_location=lambda storage, loc: storage
    )

    nmt_model_opt = ArgumentParser.ckpt_model_opts(nmt_model_ckpt["opt"])
    ArgumentParser.update_model_opts(nmt_model_opt)
    ArgumentParser.validate_model_opts(nmt_model_opt)
    fields = nmt_model_ckpt["vocab"]

    print("fields", fields, fields["src"], fields["src"].fields, type(fields["src"]))
    print("fieldls indices", fields["indices"].__dict__)

    import pickle

    vocab = {
        "src": fields["src"].fields[0][-1],
        "tgt": fields["tgt"].fields[0][-1],
        "indices": fields["indices"],
    }
    print("\nvocab_src", vocab["src"].__dict__, type(vocab["src"]))

    # torch.save(vocab['src'], 'src_vocab.pt')
    # torch.save(vocab['tgt'], 'tgt_vocab.pt')
    # torch.save(vocab, 'vocab.pt')

    # print('\nvocab pickle', vocab)
    # with open('vocab.pkl', 'wb') as fh:
    #    pickle.dump(vocab, fh)

    with open("src_vocab.pkl", "wb") as fh:
        pickle.dump(vocab["src"], fh)

    with open("tgt_vocab.pkl", "wb") as fh:
        pickle.dump(vocab["tgt"], fh)

    # with open('indices_vocab.pkl', 'wb') as fh:
    #    pickle.dump(fields['indices'], fh)

    # Avoid functionality on inference
    nmt_model_opt.update_vocab = False
    print(nmt_model_opt.generator_function)
    model = build_tpr_swap_model(
        opt,
        nmt_model_opt,
        fields,
        use_gpu(opt),
        gpu_id=opt.gpu,
        tpr_model_checkpoint=tpr_checkpoint,
        nmt_model_checkpoint=nmt_model_ckpt,
    )
    if opt.fp32:
        model.float()
    elif opt.int8:
        if opt.gpu >= 0:
            raise ValueError("Dynamic 8-bit quantization is not supported on GPU")
        torch.quantization.quantize_dynamic(model, inplace=True)

    model.encoder.eval()
    model.decoder.eval()
    # model.eval()
    model.generator.eval()
    model.training = False
    print(f"model.training={model.training}")

    return fields, model, nmt_model_opt


def build_tpr_swap_model(
    opt,
    nmt_model_opt,
    fields,
    gpu,
    gpu_id=None,
    tpr_model_checkpoint=None,
    nmt_model_checkpoint=None,
):
    """Build a model from opts.
    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        fields (dict[str, torchtext.data.Field]):
            `Field` objects for the model.
        gpu (bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
        gpu_id (int or NoneType): Which GPU to use.
    Returns:
        the NMTModel.
    """

    # for back compat when attention_dropout was not defined
    try:
        nmt_model_opt.attention_dropout
    except AttributeError:
        nmt_model_opt.attention_dropout = nmt_model_opt.dropout

    # Build Model
    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")

    # Build tpr_encoder.
    tpr_encoder = onmt.model_builder.build_encoder(opt, encoder_type="tpr")
    tpr_encoder.load_state_dict(tpr_model_checkpoint)
    print("LOADED TPR ENCODER")

    # ----
    # Build NMT model.
    nmt_model = build_task_specific_model(nmt_model_opt, fields)
    print("LOADED NMT MODEL")

    # Build Generator.
    if not nmt_model_opt.copy_attn:
        if nmt_model_opt.generator_function == "sparsemax":
            gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(
            nn.Linear(nmt_model_opt.dec_rnn_size, len(fields["tgt"].base_field.vocab)),
            Cast(torch.float32),
            gen_func,
        )
        if nmt_model_opt.share_decoder_embeddings:
            generator[0].weight = nmt_decoder.embeddings.word_lut.weight
    else:
        tgt_base_field = fields["tgt"].base_field
        vocab_size = len(tgt_base_field.vocab)
        pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
        generator = CopyGenerator(nmt_model_opt.dec_rnn_size, vocab_size, pad_idx)
        if nmt_model_opt.share_decoder_embeddings:
            generator.linear.weight = nmt_decoder.embeddings.word_lut.weight

    print("LOADED MODEL GENERATOR")

    # Load the model states from checkpoint or initialize them.
    if nmt_model_checkpoint is None or nmt_model_opt.update_vocab:
        if nmt_model_opt.param_init != 0.0:
            for p in nmt_model.parameters():
                p.data.uniform_(-nmt_model_opt.param_init, nmt_model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-nmt_model_opt.param_init, nmt_model_opt.param_init)
        if nmt_model_opt.param_init_glorot:
            for p in nmt_model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(nmt_model, "encoder") and hasattr(nmt_model.encoder, "embeddings"):
            nmt_model.encoder.embeddings.load_pretrained_vectors(
                nmt_model_opt.pre_word_vecs_enc
            )
        if hasattr(nmt_model.decoder, "embeddings"):
            nmt_model.decoder.embeddings.load_pretrained_vectors(
                nmt_model_opt.pre_word_vecs_dec
            )

    if nmt_model_checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r"(.*)\.layer_norm((_\d+)?)\.b_2", r"\1.layer_norm\2.bias", s)
            s = re.sub(r"(.*)\.layer_norm((_\d+)?)\.a_2", r"\1.layer_norm\2.weight", s)
            return s

        nmt_model_checkpoint["model"] = {
            fix_key(k): v for k, v in nmt_model_checkpoint["model"].items()
        }
        # end of patch for backward compatibility

        nmt_model.load_state_dict(nmt_model_checkpoint["model"], strict=False)
        generator.load_state_dict(nmt_model_checkpoint["generator"], strict=False)

    nmt_decoder = nmt_model.decoder

    tpr_model = onmt.models.TPRSwapModel(tpr_encoder, nmt_decoder)
    tpr_model.generator = generator

    tpr_model.to(device)
    if nmt_model_opt.model_dtype == "fp16" and nmt_model_opt.optim == "fusedadam":
        tpr_model.half()

    return tpr_model


def build_model(model_opt, opt, fields, checkpoint):
    logger.info("Building model...")
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    logger.info(model)
    return model
