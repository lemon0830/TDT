# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE processors and helpers """

import logging
import os
import json

from .utils import DataProcessor, InputExample, InputFeatures
from ...file_utils import is_tf_available

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def glue_convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label))

    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield  ({'input_ids': ex.input_ids,
                         'attention_mask': ex.attention_mask,
                         'token_type_ids': ex.token_type_ids},
                        ex.label)

        return tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             tf.int64),
            ({'input_ids': tf.TensorShape([None]),
              'attention_mask': tf.TensorShape([None]),
              'token_type_ids': tf.TensorShape([None])},
             tf.TensorShape([])))

    return features


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ChineseProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.shuf.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.shuf.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class EngProcessor(DataProcessor):
    """Processor for the my data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["学校生活", "旅行", "商品", "我们所发生的变化", "科普知识与现代技术", "音乐与舞蹈", "环境保护", 
                "工作与职业", "语言与文化", "学校活动", "学习科目", "服装", "合作与交流", "著名人物", "节假日活动", 
                "未来生活", "社会行为", "情绪", "社区", "朋友", "电影与戏剧", "家庭生活", "情感", "计划", 
                "学校设施", "个人信息", "点餐", "中国历史与世界历史", "体育健身", "交通运输方式", "故事", "食物", 
                "自我保护", "兴趣与爱好", "礼仪习俗", "新闻", "健康饮食", "学会学习", "居室", "娱乐活动", 
                "有趣的地方", "其他人", "学校信息", "问路和指路", "规章制度", "饮食习俗", "安全守则", 
                "志愿服务及慈善活动", "动物与植物", "游戏与休闲", "疾病与医疗", "家人和亲友", "季节", "语言学习经历", 
                "房屋与住所", "急救", "家具和家庭用品", "天气情况与预报", "语言学习策略", "体态语", "国家与民族", 
                "神秘事件", "家务琐事与许可", "意外", "使用互联网", "体育与赛事", "安排"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i + 1)
            text_a = line[1]
            label = None if set_type == "test" else line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['premise'].numpy().decode('utf-8'),
                            tensor_dict['hypothesis'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['question1'].numpy().decode('utf-8'),
                            tensor_dict['question2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['question'].numpy().decode('utf-8'),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class tnewsProcessor(DataProcessor):
    """Processor for the tnews data set (GLUE version)."""
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        train_file = os.path.join(data_dir, 'train.json')
        # train_ext_file = os.path.join(data_dir, 'toutiao_filter_with_key.json')
        train = self._read_json(train_file)
        # train_ext = self._read_json(train_ext_file)
        return self._create_examples(train, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_file = os.path.join(data_dir, 'dev.json')
        dev = self._read_json(dev_file)
        return self._create_examples(dev, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_file = os.path.join(data_dir, 'test_new.json')
        test = self._read_json(test_file)
        return self._create_examples(test, "test")

    def get_labels(self):
        """See base class."""
        return ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112",
                "113", "114", "115", "116"]

    def _read_json(self, input_dir):
        json_lines = []
        with open(input_dir, 'r', encoding='utf-8') as fin:
            #data_raw = json.load(fin)
            lines = fin.readlines()
            for line in lines:
                jsonstr = json.loads(line.strip())
                json_lines.append(jsonstr)
        return json_lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line["sentence"]
            text_b = ""
            # text_b = line["keywords"]
            if "label" not in line:
                label = "100"
            else:
                label = line["label"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class iflytekProcessor(DataProcessor):
    """Processor for the iflytek data set (GLUE version)."""
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        train_file = os.path.join(data_dir, 'train.json')
        train = self._read_json(train_file)
        return self._create_examples(train, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_file = os.path.join(data_dir, 'dev.json')
        dev = self._read_json(dev_file)
        return self._create_examples(dev, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_file = os.path.join(data_dir, 'test.json')
        test = self._read_json(test_file)
        return self._create_examples(test, "test")

    def get_labels(self):
        """See base class."""
        return [str(i) for i in range(119)]
        # return ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112",
        #         "113", "114", "115", "116"]

    def _read_json(self, input_dir):
        json_lines = []
        with open(input_dir, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                jsonstr = json.loads(line.strip())
                json_lines.append(jsonstr)
        return json_lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line["sentence"]
            if "label" not in line:
                label = "100"
            else:
                label = line["label"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class AfqmcProcessor(DataProcessor):
    """Processor for the Afqmc data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        train_file = os.path.join(data_dir, 'train_clean.json')
        #train_augtrue_file = os.path.join(data_dir, 'train_augtrue.json')
        #train_augfalse_file = os.path.join(data_dir, 'train_augfalse.json')
        train = self._read_json(train_file)
        #train_augtrue = self._read_json(train_augtrue_file)
        #train_augfalse = self._read_json(train_augfalse_file)

        return self._create_examples(train, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_file = os.path.join(data_dir, 'afqmc_test_label.json')
        dev = self._read_json(dev_file)
        return self._create_examples(dev, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_file = os.path.join(data_dir, 'test_clean.json')
        test = self._read_json(test_file)
        return self._create_examples(test, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _read_json(self, input_dir):
        json_lines = []
        with open(input_dir, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                jsonstr = json.loads(line.strip())
                json_lines.append(jsonstr)
        return json_lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line["sentence1"]
            text_b = line["sentence2"]
            if "label" not in line:
                label = "0"
            else:
                label = line["label"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class OcnliProcessor(DataProcessor):
    """Processor for the ocnli data set (CLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['premise'].numpy().decode('utf-8'),
                            tensor_dict['hypothesis'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        train_file = os.path.join(data_dir, 'train.50k.json')
        train = self._read_json(train_file)
        return self._create_examples(train, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_file = os.path.join(data_dir, 'dev.json')
        dev = self._read_json(dev_file)
        return self._create_examples(dev, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_file = os.path.join(data_dir, 'test.json')
        test = self._read_json(test_file)
        return self._create_examples(test, "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _read_json(self, input_dir):
        json_lines = []
        with open(input_dir, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                jsonstr = json.loads(line.strip())
                json_lines.append(jsonstr)
        return json_lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line["sentence1"]
            text_b = line["sentence2"]
            
            if "label" not in line:
                label = "contradiction"
            else:
                label = line["label"]
            if label == '-':
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class CslProcessor(DataProcessor):
    """Processor for the Csl data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_file = os.path.join(data_dir, 'train.json')
        train = self._read_json(train_file)
        return self._create_examples(train, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_file = os.path.join(data_dir, 'dev.json')
        dev = self._read_json(dev_file)
        return self._create_examples(dev, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_file = os.path.join(data_dir, 'test.json')
        test = self._read_json(test_file)
        return self._create_examples(test, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _read_json(self, input_dir):
        json_lines = []
        with open(input_dir, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                jsonstr = json.loads(line.strip())
                json_lines.append(jsonstr)
        return json_lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            
            # text_a = " ".join(line['keyword'])
            text_a = line['abst']
            text_b = "["+ ",".join(line['keyword'])+"]"
            # label = line["label"]
            label = str(line['label']) if set_type != 'test' else '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WscProcessor(DataProcessor):
    """Processor for the WSC data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_file = os.path.join(data_dir, 'train.json')
        train = self._read_json(train_file)
        return self._create_examples(train, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_file = os.path.join(data_dir, 'dev.json')
        dev = self._read_json(dev_file)
        return self._create_examples(dev, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_file = os.path.join(data_dir, 'test.json')
        test = self._read_json(test_file)
        return self._create_examples(test, "test")

    def get_labels(self):
        """See base class."""
        return ["true", "false"]

    def _read_json(self, input_dir):
        json_lines = []
        with open(input_dir, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                jsonstr = json.loads(line.strip())
                json_lines.append(jsonstr)
        return json_lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['text']
            text_a_list = list(text_a)
            target = line['target']
            query = target['span1_text']
            query_idx = target['span1_index']
            pronoun = target['span2_text']
            pronoun_idx = target['span2_index']
            assert text_a[pronoun_idx: (pronoun_idx + len(pronoun))] == pronoun, "pronoun: {}".format(pronoun)
            assert text_a[query_idx: (query_idx + len(query))] == query, "query: {}".format(query)
            if pronoun_idx > query_idx:
                text_a_list.insert(query_idx, "_")
                text_a_list.insert(query_idx + len(query) + 1, "_")
                text_a_list.insert(pronoun_idx + 2, "[")
                text_a_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
            else:
                text_a_list.insert(pronoun_idx, "[")
                text_a_list.insert(pronoun_idx + len(pronoun) + 1, "]")
                text_a_list.insert(query_idx + 2, "_")
                text_a_list.insert(query_idx + len(query) + 2 + 1, "_")
            text_a = "".join(text_a_list)
            text_b = None
            label = str(line['label']) if set_type != 'test' else 'true'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ImdbProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.shuf.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "eng": 67,
    "chinese": 2,
    "tnews": 17,
    "iflytek": 119,
    "afqmc": 2,
    "ocnli": 3,
    "csl": 2,
    "wsc": 2,
    "imdb": 2
}

glue_processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "eng": EngProcessor,
    "chinese": ChineseProcessor,
    "tnews": tnewsProcessor,
    "iflytek": iflytekProcessor,
    "afqmc": AfqmcProcessor,
    "ocnli": OcnliProcessor,
    "csl": CslProcessor,
    "wsc": WscProcessor,
    "imdb": ImdbProcessor
}

glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "eng": "classification",
    "chinese": "classification",
    "tnews": "classification",
    "iflytek": "classification",
    "afqmc": "classification",
    "ocnli": "classification",
    "csl": "classification",
    "wsc": "classification",
    "imdb": "classification"
}
