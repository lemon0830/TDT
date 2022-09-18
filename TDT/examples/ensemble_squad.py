"""
集成不同的检查点
"""

import argparse
import logging
import os
import timeit

import torch
import torch.nn as nn
from torch.utils.data import SequentialSampler, DataLoader

from examples.run_squad import MODEL_CLASSES, evaluate, load_and_cache_examples, to_list
from examples.utils_squad import RawResultExtended, RawResult, write_predictions_extended, write_predictions
from examples.utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad

logger = logging.getLogger(__name__)

MODEL_SPECS = {
    'albert': 'albert-large-v2',
    'bert': 'bert-base-uncased'
}


def evaluate_multi_ckpt(args, models, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    for i, model in enumerate(models):
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            models[i] = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    start_time = timeit.default_timer()
    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1]
                      }
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = None if args.model_type == 'xlm' else batch[2]  # XLM don't use segment_ids
            example_indices = batch[3]
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4],
                               'p_mask': batch[5]})
            all_outputs = None
            for model in models:
                model.eval()
                outputs = model(**inputs)
                if all_outputs is None:
                    all_outputs = list(outputs)
                else:
                    for i, o in enumerate(outputs):
                        assert all_outputs[i].shape == o.shape
                        all_outputs[i] += o
            for i, o in enumerate(all_outputs):
                all_outputs[i] = o / len(models)
            outputs = all_outputs

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            if args.model_type in ['xlnet', 'xlm']:
                # XLNet uses a more complex post-processing procedure
                result = RawResultExtended(unique_id=unique_id,
                                           start_top_log_probs=to_list(outputs[0][i]),
                                           start_top_index=to_list(outputs[1][i]),
                                           end_top_log_probs=to_list(outputs[2][i]),
                                           end_top_index=to_list(outputs[3][i]),
                                           cls_logits=to_list(outputs[4][i]))
            else:
                result = RawResult(unique_id=unique_id,
                                   start_logits=to_list(outputs[0][i]),
                                   end_logits=to_list(outputs[1][i]))
            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    if args.model_type in ['xlnet', 'xlm']:
        # XLNet uses a more complex post-processing procedure
        write_predictions_extended(examples, features, all_results, args.n_best_size,
                                   args.max_answer_length, output_prediction_file,
                                   output_nbest_file, output_null_log_odds_file, args.predict_file,
                                   model.config.start_n_top, model.config.end_n_top,
                                   args.version_2_with_negative, tokenizer, args.verbose_logging)
    else:
        write_predictions(examples, features, all_results, args.n_best_size,
                          args.max_answer_length, args.do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                          args.version_2_with_negative, args.null_score_diff_threshold)

    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=args.predict_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)
    return results


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_file", default=None, type=str, required=True,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the predictions will be written.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--checkpoints", default=None, type=str, required=True, nargs='+',
                        help='Path to checkpoint dir, should contains pytorch_model.bin and config.json')
    parser.add_argument("--tokenizer_dir", default=None, type=str, required=True,
                        help='Path to tokenizer dir')
    parser.add_argument("--avg_target", default='weight', type=str, required=True, choices=['weight', 'logit'])

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--version_2_with_negative', action='store_true', default=True,
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")

    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.local_rank = -1
    args.n_gpu = torch.cuda.device_count()
    args.model_name_or_path = MODEL_SPECS[args.model_type]

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    models = []
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_dir, do_lower_case=args.do_lower_case)
    for checkpoint in args.checkpoints:
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        model.eval()
        models.append(model)

    if args.avg_target == 'weight':
        param_dict = {}
        for model in models:
            for param_name, param in model.named_parameters():
                if param_name not in param_dict:
                    param_dict[param_name] = param
                else:
                    assert param_dict[param_name].shape == param.shape, \
                        'checkpoints for weight averaging must have same architecture'
                    param_dict[param_name] += param
        for k in param_dict:
            param_dict[k] = param_dict[k] / len(models)
        model = models[0]
        model.load_state_dict(param_dict)
        while len(models) > 1:
            del models[1]

        result = evaluate(args, model, tokenizer)
        print(result)
    else:
        result = evaluate_multi_ckpt(args, models, tokenizer)
        print(result)

    return result


if __name__ == '__main__':
    main()
    exit(0)
    with torch.no_grad():
        xlarge_ckpts = ['models/debug_squadxl/', 'models/debug_squadxl/checkpoint-5500',
                        'models/debug_squadxl/checkpoint-5250']
        xxlarge_ckpts = ['models/120606/checkpoint-25000', 'models/120606/checkpoint-24000',
                         'models/120606/checkpoint-26000']
        parser = argparse.ArgumentParser()
        parser.add_argument("--predict_file", default=None, type=str, required=True,
                            help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
        parser.add_argument("--output_dir", default=None, type=str, required=True,
                            help="The output directory where the predictions will be written.")
        parser.add_argument("--model_type", default=None, type=str, required=True,
                            help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
        parser.add_argument("--checkpoints", default=None, type=str, required=True, nargs='+',
                            help='Path to checkpoint dir, should contains pytorch_model.bin and config.json')
        parser.add_argument("--tokenizer_dir", default=None, type=str, required=True,
                            help='Path to tokenizer dir')
        parser.add_argument("--avg_target", default='weight', type=str, required=True, choices=['weight', 'logit'])

        parser.add_argument("--max_seq_length", default=384, type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                                 "longer than this will be truncated, and sequences shorter than this will be padded.")
        parser.add_argument("--doc_stride", default=128, type=int,
                            help="When splitting up a long document into chunks, how much stride to take between chunks.")
        parser.add_argument("--max_query_length", default=64, type=int,
                            help="The maximum number of tokens for the question. Questions longer than this will "
                                 "be truncated to this length.")
        parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                            help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument("--do_lower_case", action='store_true',
                            help="Set this flag if you are using an uncased model.")

        parser.add_argument("--verbose_logging", action='store_true',
                            help="If true, all of the warnings related to data processing will be printed. "
                                 "A number of warnings are expected for a normal SQuAD evaluation.")
        parser.add_argument('--version_2_with_negative', action='store_true', default=True,
                            help='If true, the SQuAD examples contain some that do not have an answer.')
        parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                            help="If null_score - best_non_null is greater than the threshold predict null.")

        parser.add_argument("--n_best_size", default=20, type=int,
                            help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
        parser.add_argument("--max_answer_length", default=30, type=int,
                            help="The maximum length of an answer that can be generated. This is needed because the start "
                                 "and end predictions are not conditioned on one another.")

        parser.add_argument('--overwrite_output_dir', action='store_true',
                            help="Overwrite the content of the output directory")
        parser.add_argument('--overwrite_cache', action='store_true',
                            help="Overwrite the cached training and evaluation sets")
        parser.add_argument('--seed', type=int, default=42,
                            help="random seed for initialization")

        args = parser.parse_args()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.device = device
        args.local_rank = -1
        args.n_gpu = torch.cuda.device_count()
        args.model_name_or_path = MODEL_SPECS[args.model_type]

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_dir, do_lower_case=args.do_lower_case)

        xlarge_models = []
        for checkpoint in xlarge_ckpts:
            model = model_class.from_pretrained(checkpoint)
            model.eval()
            xlarge_models.append(model)
        xxlarge_models = []
        for checkpoint in xxlarge_ckpts:
            model = model_class.from_pretrained(checkpoint)
            model.eval()
            xxlarge_models.append(model)

        models = []
        param_dict = {}
        for model in xlarge_models:
            for param_name, param in model.named_parameters():
                if param_name not in param_dict:
                    param_dict[param_name] = param
                else:
                    assert param_dict[param_name].shape == param.shape, \
                        'checkpoints for weight averaging must have same architecture'
                    param_dict[param_name] += param
        for k in param_dict:
            param_dict[k] = param_dict[k] / len(xlarge_models)
        model = xlarge_models[0]
        model.load_state_dict(param_dict)
        model.to(args.device)
        models.append(model)

        param_dict = {}
        for model in xxlarge_models:
            for param_name, param in model.named_parameters():
                if param_name not in param_dict:
                    param_dict[param_name] = param
                else:
                    assert param_dict[param_name].shape == param.shape, \
                        'checkpoints for weight averaging must have same architecture'
                    param_dict[param_name] += param
        for k in param_dict:
            param_dict[k] = param_dict[k] / len(xxlarge_models)
        model = xxlarge_models[0]
        model.load_state_dict(param_dict)
        model.to(args.device)
        models.append(model)

        result = evaluate_multi_ckpt(args, models, tokenizer)
        print(result)
