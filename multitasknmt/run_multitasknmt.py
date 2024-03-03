import logging
import os

import math
import tensorflow as tf
import tensorflow_addons as tfa
import opennmt
from opennmt import evaluation, inference
from opennmt.config import load_config
from opennmt.models import EmbeddingsSharingLevel
from opennmt.utils import checkpoint as checkpoint_util
from opennmt import training as training_util
from opennmt.utils import misc

from multitasknmt.spanCorruption_inputter import SpanCorruptionInputter
from multitasknmt.spanCorruptionReconstruction_inputter import SpanCorruptionReconstructionInputter
from multitasknmt.translationSpanCorruption_inputter import TranslationSpanCorruptionInputter
from multitasknmt.translationPairSpanCorruption_inputter import TranslationPairSpanCorruptionInputter
from multitasknmt.examplesMTPrompt_inputter import ExamplesMTPromptInputter
from multitasknmt.sc_mt_inputter import ScMtInputter
from multitasknmt.scr_mt_inputter import ScrMtInputter
from multitasknmt.sc_scr_mt_inputter import ScScrMtInputter
from multitasknmt.sc_tpsc_tsc_mt_inputter import ScTpscTscMtInputter
from multitasknmt.sc_scr_tpsc_tsc_mt_inputter import ScScrTpscTscMtInputter
from multitasknmt.tpsc_tsc_mt_inputter import TpscTscMtInputter
from ShareTransformer import ShareTransformer
from TransformerBigLB import TransformerBigLB

# runner.py helper functions
def _count_batch_accum(batch_size, target_batch_size, num_replicas=1):
    """Given the current batch size, the number of replicas, and the requested
    effective batch size, returns the number of gradients to accumulate.
    """
    return int(math.ceil(float(target_batch_size) / (batch_size * num_replicas)))


# runner.py train()
def train(
        runner,
        inputter,
        num_devices=1,
        num_threads=4,
        with_eval=False,
        checkpoint_path=None,
        hvd=None,
        return_summary=False,
        fallback_to_cpu=True,
        continue_from_checkpoint=False,
    ):
        if hvd is None:
            num_replicas = num_devices
            is_master = True
        else:
            if num_devices > 1:
                raise ValueError(
                    "num_devices (or num_gpus) should be set to 1 when using Horovod"
                )
            num_replicas = hvd.size()
            is_master = hvd.rank() == 0

        devices = misc.get_devices(count=num_devices, fallback_to_cpu=fallback_to_cpu)

        config = runner._finalize_config(
            training=True, num_replicas=num_replicas, num_devices=num_devices
        )

        mixed_precision = runner._mixed_precision and misc.enable_mixed_precision()
        model = runner._init_model(config)
        optimizer = model.get_optimizer()

        data_config = config["data"]
        train_config = config["train"]
        eval_config = config["eval"]

        batch_type = train_config["batch_type"]
        batch_size = train_config["batch_size"]
        batch_size_multiple = (
            8
            if batch_type == "tokens" and (mixed_precision or runner._jit_compile)
            else 1
        )
        batch_autotune_mode = train_config.get("batch_autotune_mode")
        length_bucket_width = train_config["length_bucket_width"]
        pad_to_bucket_boundary = train_config.get("pad_to_bucket_boundary")

        if runner._jit_compile:
            length_bucket_width = max(length_bucket_width, batch_size_multiple)
            pad_to_bucket_boundary = True

        dataset_fn = (
            lambda input_context: inputter.make_training_dataset(    
                                    data_config['train_features_file'],
                                    data_config['train_labels_file'],
                                    batch_size=train_config['batch_size'],
                                    batch_type=train_config['batch_type'],
                                    shuffle_buffer_size=train_config['sample_buffer_size'],
                                    length_bucket_width=train_config['length_bucket_width'],
                                    maximum_features_length=train_config['maximum_features_length'],
                                    maximum_labels_length=train_config['maximum_labels_length'],
                                    single_pass=train_config['single_pass'],
                                    num_threads=num_threads,
                                  )
        )

        checkpoint = None
        evaluator = None
        if is_master:
            checkpoint = checkpoint_util.Checkpoint.from_config(
                config, model, optimizer=optimizer
            )
            checkpoint.restore(
                checkpoint_path=checkpoint_path,
                weights_only=(
                    checkpoint_path is not None and not continue_from_checkpoint
                ),
            )
            if with_eval:
                evaluator = evaluation.Evaluator.from_config(model, config)

        # Set gradients accumulation based on the requested effective batch size.
        effective_batch_size = train_config.get("effective_batch_size")
        if effective_batch_size is not None:
            accum_steps = _count_batch_accum(
                batch_size,
                effective_batch_size,
                num_replicas=num_replicas,
            )
            if batch_autotune_mode and accum_steps > 2:
                # When autotuning the batch size, the memory usage should be the same
                # whether we are accumulating 2 steps or N steps.
                accum_steps = 2
                effective_batch_size = batch_size * num_replicas * accum_steps
            tf.get_logger().info(
                "Accumulate gradients of %d iterations to reach effective batch size of %d",
                accum_steps,
                effective_batch_size,
            )
        else:
            accum_steps = 1

        if hvd is not None:
            trainer = training_util.HorovodTrainer(
                model, optimizer, hvd, checkpoint=checkpoint
            )
        elif num_devices > 1:
            trainer = training_util.MirroredStrategyTrainer(
                model, optimizer, checkpoint=checkpoint, devices=devices
            )
        else:
            trainer = training_util.Trainer(model, optimizer, checkpoint=checkpoint)

        summary = trainer(
            dataset_fn,
            max_step=train_config.get("max_step"),
            accum_steps=accum_steps,
            report_steps=train_config.get("save_summary_steps", 100),
            save_steps=train_config.get("save_checkpoints_steps", 5000),
            evaluator=evaluator,
            eval_steps=eval_config.get("steps", 5000),
            moving_average_decay=train_config.get("moving_average_decay"),
        )

        average_last_checkpoints = train_config.get("average_last_checkpoints", 0)
        if checkpoint is None:
            output_dir = None
        elif average_last_checkpoints > 0:
            output_dir = runner.average_checkpoints(
                os.path.join(checkpoint.model_dir, "avg"),
                max_count=average_last_checkpoints,
            )
        else:
            output_dir = checkpoint.model_dir

        if mixed_precision:
            misc.disable_mixed_precision()

        if return_summary:
            return output_dir, summary
        return output_dir


# runner.py evaluate()
def evaluate(runner, inputter, features_file=None, labels_file=None, predictions_file=None, checkpoint_path=None):
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

    dataset = inputter.make_evaluation_dataset(
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


# runner.py score()
def score(
        runner,
        inputter,
        features_file,
        predictions_file=None,
        checkpoint_path=None,
        output_file=None,
    ):
        """Scores existing predictions.

        Args:
          features_file: The input file.
          predictions_file: The predictions file to score.
          checkpoint_path: Path to specific checkpoint to load. If ``None``,
            the latest is used.
          output_file: The file where the scores are saved. Otherwise, they will be
            printed on the standard output.
        """
        config = runner._finalize_config(training=False)
        model = runner._init_model(config)
        checkpoint = checkpoint_util.Checkpoint.from_config(config, model)
        checkpoint.restore(checkpoint_path=checkpoint_path, weights_only=True)
        score_config = config["score"]
        dataset = inputter.make_evaluation_dataset(
            features_file or config["data"].get("eval_features_file"),
            predictions_file,
            score_config["batch_size"],
            batch_type=score_config["batch_type"],
            length_bucket_width=score_config["length_bucket_width"],
            prefetch_buffer_size=score_config.get("prefetch_buffer_size"),
        )
        inference.score_dataset(
            model, dataset, print_params=score_config, output_file=output_file
        )


def translate(runner, checkpoint_path, inputter, source_file, predictions_file, srcLang, tgtLang):
  config = runner._finalize_config(training=False)
  model = runner._init_model(config)
  checkpoint = checkpoint_util.Checkpoint.from_config(config, model)
  checkpoint.restore(checkpoint_path=checkpoint_path, weights_only=True)

  batch_size = config['infer']['batch_size']

  dataset = inputter.make_inference_dataset(source_file, batch_size, srcLang=srcLang, tgtLang=tgtLang)
  inference.predict_dataset(model,
                            dataset,
                            predictions_file=predictions_file,
                            log_time=True)



def main():
  tf.get_logger().setLevel(logging.INFO)

  gpu_devices = tf.config.experimental.list_physical_devices('GPU')
  for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("run",      choices=["train", "finetune", "translate", "evaluate", "score"], help="Run type")

  parser.add_argument("--train_inputter",      default="ScTpscTscMtInputter", choices=["ScTpscTscMtInputter",
                                                                                       "ScScrTpscTscMtInputter",
                                                                                       "ScMtInputter",
                                                                                       "ScrMtInputter",
                                                                                       "ScScrMtInputter",
                                                                                       "TpscTscMtInputter",
                                                                                       "SpanCorruptionInputter",
                                                                                       "SpanCorruptionReconstructionInputter",
                                                                                       "TranslationSpanCorruptionInputter",
                                                                                       "TranslationPairSpanCorruptionInputter",
                                                                                       "ExamplesMTPromptInputter",
                                                                                      ], help="")

  parser.add_argument("--config",               default="", nargs="+", help="List of configuration files.")

  parser.add_argument("--model_type",           default="deltalm", choices=["deltalm", "shareTransformer", "shareTransformerDouble", "TransformerBigLB"])
  parser.add_argument("--model_dir",            default="",   help="")

  parser.add_argument("--learning_rate",        default=-1,   type=float, help="")
  parser.add_argument("--batch_type",           default="",   help="")
  parser.add_argument("--batch_size",           default=-1,   type=int, help="")
  parser.add_argument("--effective_batch_size", default=-1,   type=int, help="")
  parser.add_argument("--beam_width",           default=-1,   type=int, help="")

  parser.add_argument("--mixed_precision",      default=False, action="store_true", help="")
  parser.add_argument("--num_gpus",             default=1,    type=int, help="")
  parser.add_argument("--num_threads",          default=4,    type=int, help="")

  parser.add_argument("--features_file",         default="", type=str, help="")
  parser.add_argument("--labels_file",           default="", type=str, help="")
  parser.add_argument("--predictions_file",      default="", type=str, help="")
  parser.add_argument("--output_file",           default="", type=str, help="")
  parser.add_argument("--checkpoint_path",       default="", type=str, help="")
  parser.add_argument("--continue_from_checkpoint", default=False, action="store_true", help="")

  parser.add_argument("--srcLang",               default="", type=str, help="")
  parser.add_argument("--tgtLang",               default="", type=str, help="")

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
    config['infer']['batch_size'] = args.batch_size
    config['eval']['batch_size'] = args.batch_size
    config['score']['batch_size'] = args.batch_size
  if args.effective_batch_size > 0:
    config['train']['effective_batch_size'] = args.effective_batch_size
    config['infer']['batch_size'] = args.effective_batch_size
    config['eval']['batch_size'] = args.effective_batch_size
    config['score']['batch_size'] = args.effective_batch_size
  if args.beam_width > 0:
    config['params']['beam_width'] = args.beam_width


  model = None
  if args.model_type == 'deltalm':
    model = opennmt.models.Transformer(
          # num_layers = 16,
          # num_units = 1024,
          # num_heads = 16,
          # ffn_inner_dim = 4096,
          # DeltaLM
          num_layers = 12,
          num_units = 768,
          num_heads = 12,
          ffn_inner_dim = 3072,
          # dropout = ,
          # attention_dropout = ,
          # ffn_dropout = ,
          # ffn_activation =
          share_embeddings = EmbeddingsSharingLevel.SOURCE_TARGET_INPUT)
  elif args.model_type == 'shareTransformer':
    # DeltaLM
    model = ShareTransformer(
          num_layers = 12,
          num_units = 768,
          num_heads = 12,
          ffn_inner_dim = 3072,
          # dropout = ,
          # attention_dropout = ,
          # ffn_dropout = ,
          # ffn_activation =
          )
  elif args.model_type == 'shareTransformerDouble':
    # DeltaLM
    model = ShareTransformer(
          num_layers = 21,
          num_units = 768,
          num_heads = 12,
          ffn_inner_dim = 3072,
          # dropout = ,
          # attention_dropout = ,
          # ffn_dropout = ,
          # ffn_activation =
          )
  elif args.model_type == 'TransformerBigLB':
    model = TransformerBigLB()
  else:
     raise Exception(f'Unknown model type {args.model_type}')

  tf.get_logger().info(f'Model type {args.model_type}')

  if args.run == "train":
    if args.train_inputter == "ScTpscTscMtInputter":
      datasetInputter = ScTpscTscMtInputter(
        model.features_inputter,
        model.labels_inputter,
        span_corruption_noise_density=0.15,
        span_corruption_mean_noise_span_length=3.0,
        translation_span_corruption_noise_density=0.50,
        translation_span_corruption_mean_noise_span_length=3.0,
      )
    elif args.train_inputter == "ScScrTpscTscMtInputter":
      datasetInputter = ScScrTpscTscMtInputter(
        model.features_inputter,
        model.labels_inputter,
        span_corruption_noise_density=0.15,
        span_corruption_mean_noise_span_length=3.0,
        translation_span_corruption_noise_density=0.50,
        translation_span_corruption_mean_noise_span_length=3.0,
      )
    elif args.train_inputter == "ScMtInputter":
      datasetInputter = ScMtInputter(
        model.features_inputter,
        model.labels_inputter,
        span_corruption_noise_density=0.15,
        span_corruption_mean_noise_span_length=3.0,
      )
    elif args.train_inputter == "ScrMtInputter":
      datasetInputter = ScrMtInputter(
        model.features_inputter,
        model.labels_inputter,
        span_corruption_noise_density=0.15,
        span_corruption_mean_noise_span_length=3.0,
      )
    elif args.train_inputter == "ScScrMtInputter":
      datasetInputter = ScScrMtInputter(
        model.features_inputter,
        model.labels_inputter,
        span_corruption_noise_density=0.15,
        span_corruption_mean_noise_span_length=3.0,
      )
    elif args.train_inputter == "TpscTscMtInputter":
      datasetInputter = TpscTscMtInputter(
        model.features_inputter,
        model.labels_inputter,
        translation_span_corruption_noise_density=0.50,
        translation_span_corruption_mean_noise_span_length=3.0,
      )
    elif args.train_inputter == "SpanCorruptionInputter":
      datasetInputter = SpanCorruptionInputter(
        model.features_inputter,
        model.labels_inputter,
        span_corruption_noise_density=0.15,
        span_corruption_mean_noise_span_length=3.0,
      )
    elif args.train_inputter == "SpanCorruptionReconstructionInputter":
      datasetInputter = SpanCorruptionReconstructionInputter(
        model.features_inputter,
        model.labels_inputter,
        span_corruption_noise_density=0.15,
        span_corruption_mean_noise_span_length=3.0,
      )
    elif args.train_inputter == "TranslationSpanCorruptionInputter":
      datasetInputter = TranslationSpanCorruptionInputter(
        model.features_inputter,
        model.labels_inputter,
        span_corruption_noise_density=0.15,
        span_corruption_mean_noise_span_length=3.0,
      )
    elif args.train_inputter == "TranslationPairSpanCorruptionInputter":
      datasetInputter = TranslationPairSpanCorruptionInputter(
        model.features_inputter,
        model.labels_inputter,
        span_corruption_noise_density=0.15,
        span_corruption_mean_noise_span_length=3.0,
      )
    elif args.train_inputter == "ExamplesMTPromptInputter":
      datasetInputter = ExamplesMTPromptInputter(
          model.features_inputter,
          model.labels_inputter,
      )
    else:
      raise Exception(f"Dataset inputer {args.train_inputter} not recognized")

    datasetInputter.initialize(config['data'])

    runner = opennmt.Runner(model, config, auto_config=False, mixed_precision=args.mixed_precision)
    train(runner,
          datasetInputter,
          num_devices=args.num_gpus,
          num_threads=args.num_threads,
          with_eval=True,
          fallback_to_cpu=True,
          continue_from_checkpoint=args.continue_from_checkpoint)

  elif args.run == "finetune":
    raise Exception('finetune is not implemented yet')
  elif args.run == "translate":
    if len(args.srcLang)==0 or len(args.tgtLang) == 0:
       raise Exception('srclang and tgtlang need to be specified for translate mode')

    datasetInputter = ExamplesMTPromptInputter(
        model.features_inputter,
        model.labels_inputter,
    )
    datasetInputter.initialize(config['data'])
    runner = opennmt.Runner(model, config, auto_config=False, mixed_precision=args.mixed_precision)
    translate(
      runner,
      args.checkpoint_path,
      datasetInputter,
      args.features_file,
      args.predictions_file,
      args.srcLang,
      args.tgtLang,
    )
  elif args.run == "evaluate":
    datasetInputter = ExamplesMTPromptInputter(
      model.features_inputter,
      model.labels_inputter,
    )
    datasetInputter.initialize(config['data'])
    runner = opennmt.Runner(model, config, auto_config=False, mixed_precision=args.mixed_precision)
    evaluate(
      runner,
      datasetInputter,
      features_file    = args.features_file if len(args.features_file) > 0 else None,
      labels_file      = args.labels_file if len(args.labels_file) > 0 else None,
      predictions_file = args.predictions_file if len(args.predictions_file) > 0 else "predictions.txt",
      checkpoint_path  = args.checkpoint_path if len(args.checkpoint_path) > 0 else None
    )
  elif args.run == "score":
    datasetInputter = ExamplesMTPromptInputter(
      model.features_inputter,
      model.labels_inputter,
    )
    datasetInputter.initialize(config['data'])
    runner = opennmt.Runner(model, config, auto_config=False, mixed_precision=args.mixed_precision)
    score(
       runner,
       datasetInputter,
       features_file    = args.features_file if len(args.features_file) > 0 else None,
       predictions_file = args.predictions_file if len(args.predictions_file) > 0 else None,
       checkpoint_path  = args.checkpoint_path if len(args.checkpoint_path) > 0 else None,
       output_file      = args.output_file if len(args.output_file) > 0 else None,
    )
  else:
    raise Exception('Unknown command: %s' % (args.run))

if __name__ == "__main__":
  main()

