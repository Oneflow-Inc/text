from .encoder import ElmobiLm, LstmbiLm
from .utils import (
    dict2namedtuple,
    read_list,
    recover,
    create_batches,
    load_state_dict_from_url,
)
from .embed_layer import ConvTokenEmbedder, LstmTokenEmbedder, EmbeddingLayer
import json
import codecs
import os
import logging

import numpy as np
import oneflow as flow
from oneflow import nn

logger = logging.getLogger("elmo")


model_urls = {
    "elmo-simplified-chinese": "http://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowtext/elmo/elmo-simplified-chinese.tar.gz",
    "elmo-traditional-chinese": "http://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowtext/elmo/elmo-traditional-chinese.tar.gz",
    "elmo-english": "http://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowtext/elmo/elmo-english.tar.gz",
}


class ElmoModel(nn.Module):
    def __init__(self, config, word_emb_layer, char_emb_layer, use_cuda=False):
        super(ElmoModel, self).__init__()
        self.use_cuda = use_cuda
        self.config = config

        if config["token_embedder"]["name"].lower() == "cnn":
            self.token_embedder = ConvTokenEmbedder(
                config, word_emb_layer, char_emb_layer, use_cuda
            )

        elif config["token_embedder"]["name"].lower() == "lstm":
            self.token_embedder = LstmTokenEmbedder(
                config, word_emb_layer, char_emb_layer, use_cuda
            )

        if config["encoder"]["name"].lower() == "elmo":
            self.encoder = ElmobiLm(config, use_cuda)
        elif config["encoder"]["name"].lower() == "lstm":
            self.encoder = LstmbiLm(config, use_cuda)

        self.output_dim = config["encoder"]["projection_dim"]

    def forward(self, word_inp, chars_package, mask_package):

        token_embedding = self.token_embedder(
            word_inp, chars_package, (mask_package[0].size(0), mask_package[0].size(1))
        )

        if self.config["encoder"]["name"] == "elmo":
            mask = (
                flow.Tensor(mask_package[0]).long().cuda()
                if self.use_cuda
                else flow.Tensor(mask_package[0]).long()
            )
            encoder_output = self.encoder(token_embedding, mask)
            sz = encoder_output.size()
            token_embedding = flow.cat([token_embedding, token_embedding], dim=2).view(
                1, sz[1], sz[2], sz[3]
            )
            encoder_output = flow.cat([token_embedding, encoder_output], dim=0)

        elif self.config["encoder"]["name"] == "lstm":
            encoder_output = self.encoder(token_embedding)

        else:
            raise ValueError(
                "Unknown encoder: {0}".format(self.config["encoder"]["name"])
            )

        return encoder_output

    def load_model(self, path):
        self.token_embedder.load_state_dict(
            flow.load(os.path.join(path, "token_embedder"))
        )
        self.encoder.load_state_dict(flow.load(os.path.join(path, "encoder")))


class Embedder(object):
    def __init__(self, model_dir, batch_size=64):
        self.model_dir = model_dir
        self.model, self.config = self.get_model()
        self.batch_size = batch_size

    def get_model(self):
        self.use_cuda = flow.cuda.is_available()
        config_path = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                "Could not find the model config in either the model directory "
                "or the default configs.  Path in config file: {}".format(config_path)
            )
        with open(config_path, "r") as fin:
            config = json.load(fin)

        if config["token_embedder"]["char_dim"] > 0:
            self.char_lexicon = {}
            with codecs.open(
                os.path.join(self.model_dir, "char.dic"), "r", encoding="utf-8"
            ) as fpi:
                for line in fpi:
                    tokens = line.strip().split("\t")
                    if len(tokens) == 1:
                        tokens.insert(0, "\u3000")
                    token, i = tokens
                    self.char_lexicon[token] = int(i)
            char_emb_layer = EmbeddingLayer(
                config["token_embedder"]["char_dim"],
                self.char_lexicon,
                fix_emb=False,
                embs=None,
            )
            logger.info("char embedding size: " + str(len(char_emb_layer.word2id)))
        else:
            self.char_lexicon = None
            char_emb_layer = None

        if config["token_embedder"]["word_dim"] > 0:
            self.word_lexicon = {}
            with codecs.open(
                os.path.join(self.model_dir, "word.dic"), "r", encoding="utf-8"
            ) as fpi:
                for line in fpi:
                    tokens = line.strip().split("\t")
                    if len(tokens) == 1:
                        tokens.insert(0, "\u3000")
                    token, i = tokens
                    self.word_lexicon[token] = int(i)
            word_emb_layer = EmbeddingLayer(
                config["token_embedder"]["word_dim"],
                self.word_lexicon,
                fix_emb=False,
                embs=None,
            )
            logger.info("word embedding size: " + str(len(word_emb_layer.word2id)))
        else:
            self.word_lexicon = None
            word_emb_layer = None

        model = ElmoModel(config, word_emb_layer, char_emb_layer, self.use_cuda)

        if self.use_cuda:
            model.cuda()

        logger.info(str(model))
        model.load_model(self.model_dir)

        model.eval()
        return model, config

    def sents2elmo(self, sents, output_layer=-1):
        read_function = read_list

        if self.config["token_embedder"]["name"].lower() == "cnn":
            test, text = read_function(
                sents, self.config["token_embedder"]["max_characters_per_token"]
            )
        else:
            test, text = read_function(sents)
        test_w, test_c, test_lens, test_masks, test_text, recover_ind = create_batches(
            test,
            self.batch_size,
            self.word_lexicon,
            self.char_lexicon,
            self.config,
            text=text,
        )
        cnt = 0

        after_elmo = []
        for w, c, lens, masks, texts in zip(
            test_w, test_c, test_lens, test_masks, test_text
        ):
            output = self.model.forward(w, c, masks)
            for i, text in enumerate(texts):
                if self.config["encoder"]["name"].lower() == "lstm":
                    data = output[i, 1 : lens[i] - 1, :].data
                    if self.use_cuda:
                        data = data.cpu()
                    data = data.numpy()

                elif self.config["encoder"]["name"].lower() == "elmo":
                    data = output[:, i, 1 : lens[i] - 1, :].data
                    if self.use_cuda:
                        data = data.cpu()
                    data = data.numpy()
                if output_layer == -1:
                    payload = np.average(data, axis=0)
                elif output_layer == -2:
                    payload = data
                else:
                    payload = data[output_layer]
                after_elmo.append(payload)

                cnt += 1
                if cnt % 1000 == 0:
                    logger.info("Finished {0} sentences.".format(cnt))

        after_elmo = recover(after_elmo, recover_ind)
        return after_elmo


def elmo(model_type: str = "elmo-simplified-chinese", checkpoint_path: str = None):
    if checkpoint_path != None:
        try:
            Embedder(checkpoint_path)
        except FileNotFoundError:
            print("Load error, Start downloading")
        embedder = Embedder(checkpoint_path)
        return embedder.model, embedder.config
    assert (
        model_type in model_urls
    ), "The model_type {} not identifiable, please confirm."
    file = load_state_dict_from_url(model_urls[model_type], checkpoint_path)
    embedder = Embedder(file)
    return embedder.model, embedder.config
