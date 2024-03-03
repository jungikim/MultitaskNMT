import logging

import tensorflow as tf
import opennmt
from opennmt.config import load_config
from opennmt.models import EmbeddingsSharingLevel
from opennmt.utils import checkpoint as checkpoint_util
from opennmt.utils import misc
from opennmt import evaluation
from opennmt import inference

from ShareTransformer import ShareTransformer
from TransformerBigLB import TransformerBigLB


def translate(runner, checkpoint_path, source_file, predictions_file):

  config = runner._finalize_config(training=False)
  model = runner._init_model(config)
  checkpoint = checkpoint_util.Checkpoint.from_config(config, model)
  checkpoint.restore(checkpoint_path=checkpoint_path, weights_only=True)

  batch_size = config['infer']['batch_size']
  # beam_size = config['params']['beam_width']

  dataset = model.examples_inputter.make_inference_dataset(source_file, batch_size)
  inference.predict_dataset(model,
                            dataset,
                            predictions_file=predictions_file,
                            log_time=True)


# runner.py evaluate()
def evaluate(runner, features_file=None, labels_file=None, predictions_file=None, checkpoint_path=None):
    config = runner._finalize_config(training=False)
    model = runner._init_model(config)
    checkpoint = checkpoint_util.Checkpoint.from_config(config, model)
    checkpoint_path = checkpoint.restore(
        checkpoint_path=checkpoint_path, weights_only=True
    )
    step = checkpoint_util.get_step_from_checkpoint_prefix(checkpoint_path)
    evaluator = evaluation.Evaluator.from_config(
        model, config, features_file=features_file, labels_file=labels_file
    )

    dataset = model.examples_inputter.make_evaluation_dataset(
            features_file or config["data"].get("eval_features_file"),
            labels_file or config["data"].get("eval_labels_file"),
            config["eval"]["batch_size"],
            batch_type=config["eval"].get("batch_type", "tokens"),
            length_bucket_width=config["eval"].get("length_bucket_width"),
            num_threads=1,
            prefetch_buffer_size=1,
        )

    return evaluator__call__(evaluator, dataset, predictions_file, step)

# evaluation.py __call__()
def evaluator__call__(evaluator, dataset, predictions_file, step):
    """Runs the evaluator.

    Args:
        step: The current training step.

    Returns:
        A dictionary of evaluation metrics.
    """
    tf.get_logger().info("Running evaluation for step %d", step)
    output_file = None
    output_path = None
    if predictions_file:
        if predictions_file == 'predictions.txt' or predictions_file == 'predictions_rev.txt':
          output_path = os.path.join(evaluator._eval_dir, predictions_file+".%d" % step)
        else:
          output_path = predictions_file
        output_file = tf.io.gfile.GFile(output_path, "w")
        params = {"n_best": 1}
        write_fn = lambda prediction: (
            evaluator._model.print_prediction(
                prediction, params=params, stream=output_file
            )
        )
        index_fn = lambda prediction: prediction.get("index")
        ordered_writer = misc.OrderRestorer(index_fn, write_fn)

    loss_num = 0
    loss_den = 0
    metrics = evaluator._model.get_metrics()
    for batch in dataset:
        features, labels = evaluator._model.split_features_labels(batch)
        loss, predictions = evaluator._eval_fn(features, labels)
        if isinstance(loss, tuple):
            loss_num += loss[0]
            loss_den += loss[1]
        else:
            loss_num += loss
            loss_den += 1
        if metrics:
            evaluator._model.update_metrics(metrics, predictions, labels)
        if output_file is not None:
            predictions = {k: v.numpy() for k, v in predictions.items()}
            for prediction in misc.extract_batches(predictions):
                ordered_writer.push(prediction)
    if loss_den == 0:
        raise RuntimeError("No examples were evaluated")
    loss = loss_num / loss_den

    results = dict(loss=loss, perplexity=tf.math.exp(loss))
    if metrics:
        for name, metric in metrics.items():
            results[name] = metric.result()
    if predictions_file:
        tf.get_logger().info("Evaluation predictions saved to %s", output_path)
        output_file.close()
        for scorer in evaluator._scorers:
            score = scorer(evaluator._labels_file, output_path)
            if isinstance(score, dict):
                results.update(score)
            else:
                results[scorer.name] = score

    for name, value in results.items():
        if isinstance(value, tf.Tensor):
            results[name] = value.numpy()

    evaluator._record_results(step, results)
    evaluator._maybe_export(step, results)
    evaluator._maybe_garbage_collect_exports()
    return results


def main():
  tf.get_logger().setLevel(logging.INFO)

  gpu_devices = tf.config.experimental.list_physical_devices('GPU')
  for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("run",      choices=["train", "finetune", "translate", "evaluate", "score"], help="Run type")

  parser.add_argument("--config",               default="run/opennmt_koen.yml", nargs="+", help="List of configuration files.")
  parser.add_argument("--model_dir",            default="run/opennmt_koen",   help="")

  parser.add_argument("--model_type",           default="deltalm", choices=["deltalm", "shareTransformer", "shareTransformerDouble", "TransformerBigLB"])

  parser.add_argument("--learning_rate",        default=-1,   type=float, help="")
  parser.add_argument("--batch_type",           default="",   help="")
  parser.add_argument("--batch_size",           default=-1,   type=int, help="")
  parser.add_argument("--effective_batch_size", default=-1,   type=int, help="")

  parser.add_argument("--mixed_precision",      default=False, action="store_true", help="")
  parser.add_argument("--num_gpus",             default=1,    type=int, help="")

  parser.add_argument("--features_file",         default="", type=str, help="")
  parser.add_argument("--labels_file",           default="", type=str, help="")
  parser.add_argument("--predictions_file",      default="", type=str, help="")
  parser.add_argument("--output_file",           default="", type=str, help="")
  parser.add_argument("--checkpoint_path",       default="", type=str, help="")

  # parser.add_argument("--num_layers",           default=2,    type=int, help="")
  # parser.add_argument("--num_units",            default=1024, type=int, help="")
  # parser.add_argument("--num_heads",            default=16,   type=int, help="")
  # parser.add_argument("--ffn_inner_dim",        default=2048, type=int, help="")
  # parser.add_argument("--dropout",              default=0.3,  type=float, help="")
  # parser.add_argument("--attention_dropout",    default=0.3,  type=float, help="")
  # parser.add_argument("--ffn_dropout",          default=0.3,  type=float, help="")
  # parser.add_argument("--input_dropout",        default=0.3,  type=float, help="")

  args = parser.parse_args()

  if not (args.config == None or args.config == []):
    config = load_config(args.config)

  if len(args.model_dir)>0:
    config['model_dir'] = args.model_dir

  if args.learning_rate > 0:
    config['params']['learning_rate'] = args.learning_rate
  if len(args.batch_type) > 0:
    config['train']['batch_type'] = args.batch_type
  if args.batch_size > 0:
    config['train']['batch_size'] = args.batch_size
  if args.effective_batch_size > 0:
    config['train']['effective_batch_size'] = args.effective_batch_size

  model = None
  if args.model_type == 'deltalm':
    model = opennmt.models.Transformer(
          # DeltaLM
          num_layers = 12,
          num_units = 768,
          num_heads = 12,
          ffn_inner_dim = 3072,
          share_embeddings = EmbeddingsSharingLevel.SOURCE_TARGET_INPUT)
  elif args.model_type == 'TransformerBig':
    model = opennmt.models.TransformerBig(
      num_units=1024,
      num_heads=16,
      ffn_inner_dim=4096,
      share_embeddings = EmbeddingsSharingLevel.SOURCE_TARGET_INPUT)
  elif args.model_type == 'shareTransformer':
    # DeltaLM
    model = ShareTransformer(
          num_layers = 12,
          num_units = 768,
          num_heads = 12,
          ffn_inner_dim = 3072)
  elif args.model_type == 'shareTransformerDouble':
    # DeltaLM
    model = ShareTransformer(
          num_layers = 21,
          num_units = 768,
          num_heads = 12,
          ffn_inner_dim = 3072)
  elif args.model_type == 'TransformerBigLB':
    model = TransformerBigLB()
  else:
     raise Exception(f'Unknown model type {args.model_type}')

  tf.get_logger().info(f'Model type {args.model_type}')


  if args.run == "train":
    runner = opennmt.Runner(model, config, auto_config=True, mixed_precision=args.mixed_precision)
    runner.train(
          num_devices=args.num_gpus, 
          with_eval=True,
          fallback_to_cpu=False)

  elif args.run == "finetune":
    raise Exception('finetune is not implemented yet')
  elif args.run == "translate":
    runner = opennmt.Runner(model, config, auto_config=False, mixed_precision=args.mixed_precision)
    translate(runner,
              args.checkpoint_path,
              args.features_file,
              args.predictions_file)
  elif args.run == 'evaluate':
    runner = opennmt.Runner(model, config, auto_config=False, mixed_precision=args.mixed_precision)
    # runner.evaluate(
    #   features_file   = args.features_file if len(args.features_file) > 0 else None,
    #   labels_file     = args.labels_file if len(args.labels_file) > 0 else None,
    #   checkpoint_path = args.checkpoint_path if len(args.checkpoint_path) > 0 else None
    # )
    evaluate(runner,
             features_file    = args.features_file if len(args.features_file) > 0 else None,
             labels_file      = args.labels_file if len(args.labels_file) > 0 else None,
             predictions_file = args.predictions_file if len(args.predictions_file) > 0 else None,
             checkpoint_path  = args.checkpoint_path if len(args.checkpoint_path) > 0 else None
    )
  elif args.run == 'score':
    runner = opennmt.Runner(model, config, auto_config=False, mixed_precision=args.mixed_precision)
    runner.score(
      features_file    = args.features_file if len(args.features_file) > 0 else None,
      predictions_file = args.predictions_file if len(args.predictions_file) > 0 else None,
      checkpoint_path  = args.checkpoint_path if len(args.checkpoint_path) > 0 else None,
      output_file      = args.output_file if len(args.output_file) > 0 else None,
    )
  else:
    raise Exception('Unknown command: %s' % (args.run))

if __name__ == "__main__":
  main()
