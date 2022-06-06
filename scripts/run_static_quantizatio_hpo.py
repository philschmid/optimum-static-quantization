import argparse
import copy
import json
import logging
import os
import shutil
import sys
import tempfile
from functools import partial
from pathlib import Path

import evaluate
import optuna
from datasets import load_dataset
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoCalibrationConfig, AutoQuantizationConfig
from optimum.onnxruntime.preprocessors import QuantizationPreprocessor
from optimum.onnxruntime.preprocessors.passes import (
    ExcludeGeLUNodes,
    ExcludeLayerNormNodes,
    ExcludeNodeAfter,
    ExcludeNodeFollowedBy,
)
from transformers import AutoTokenizer, pipeline

# Set up logging
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--model_id", type=str, default="mrm8488/distilroberta-finetuned-banking77")

    parser.add_argument("--dataset_id", type=str, default="banking77")
    parser.add_argument("--dataset_config", type=str, default="default")
    parser.add_argument("--n_trials", type=int, default=50)

    parser.add_argument("--onnx_path", type=str, default="onnx")

    # # Data, model, and output directories
    # parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    # parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    # parser.add_argument("--validation_dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    # parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()
    return args


def evaluate_model(pipe, dataset, metric, text_column="text", label_column="label"):
    def eval(sample):
        onnx = pipe(sample[text_column])
        return {
            "pred": pipe.model.config.label2id[onnx[0]["label"]],
            "ref": sample[label_column],
        }

    res = dataset.map(eval)
    return metric.compute(references=res["ref"], predictions=res["pred"])


def main(args):
    onnx_path = Path(args.onnx_path)
    # Evaluate during training and a bit more often
    # than the default to be able to prune bad trials early.
    # Disabling tqdm is a matter of preference.

    # load vanilla transformers and convert to onnx
    model = ORTModelForSequenceClassification.from_pretrained(args.model_id, from_transformers=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # save onnx checkpoint and tokenizer
    model.save_pretrained(onnx_path)
    tokenizer.save_pretrained(onnx_path)

    # create ORTQuantizer and define quantization configuration
    quantizer = ORTQuantizer.from_pretrained(args.model_id, feature=model.pipeline_task)
    qconfig = AutoQuantizationConfig.avx512_vnni(
        is_static=True,
        format=QuantFormat.QOperator,
        mode=QuantizationMode.QLinearOps,
        per_channel=False,
    )

    def preprocess_fn(ex, tokenizer):
        return tokenizer(ex["text"], padding="longest")

    # Create the calibration dataset
    calibration_samples = 256
    calibration_dataset = quantizer.get_calibration_dataset(
        args.dataset_id,
        dataset_config_name=args.dataset_config if args.dataset_config != "default" else None,
        preprocess_function=partial(preprocess_fn, tokenizer=quantizer.tokenizer),
        num_samples=calibration_samples,
        dataset_split="train",
    )

    def static_quantization(trial):
        hpo_percentile = trial.suggest_float("percentile", 99, 99.9999)
        # hpo_percentile = trial.suggest_categorical("percentile", [99.99, 99.999, 99.9999])

        # Create the calibration configuration containing the parameters related to calibration.
        calibration_config = AutoCalibrationConfig.percentiles(calibration_dataset, percentile=hpo_percentile)
        # Perform the calibration step: computes the activations quantization ranges
        shards = 8
        for i in range(shards):
            shard = calibration_dataset.shard(shards, i)
            quantizer.partial_fit(
                dataset=shard,
                calibration_config=calibration_config,
                onnx_model_path=onnx_path / "model.onnx",
                operators_to_quantize=qconfig.operators_to_quantize,
                batch_size=calibration_samples // shards,
                use_external_data_format=False,
            )
        ranges = quantizer.compute_ranges()

        # Create a quantization preprocessor to determine the nodes to exclude
        quantization_preprocessor = QuantizationPreprocessor()

        # Exclude the nodes constituting LayerNorm
        quantization_preprocessor.register_pass(ExcludeLayerNormNodes())
        # Exclude the nodes constituting GELU
        quantization_preprocessor.register_pass(ExcludeGeLUNodes())
        # Exclude the residual connection Add nodes
        quantization_preprocessor.register_pass(ExcludeNodeAfter("Add", "Add"))
        # Exclude the Add nodes following the Gather operator
        quantization_preprocessor.register_pass(ExcludeNodeAfter("Gather", "Add"))
        # Exclude the Add nodes followed by the Softmax operator
        quantization_preprocessor.register_pass(ExcludeNodeFollowedBy("Add", "Softmax"))

        # remove temp augmented model again
        os.remove("augmented_model.onnx")

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Quantize the same way we did for dynamic quantization!
            quantizer.export(
                onnx_model_path=onnx_path / "model.onnx",
                onnx_quantized_model_output_path=Path(tmpdirname) / "model-quantized.onnx",
                calibration_tensors_range=ranges,
                quantization_config=qconfig,
                # preprocessor=quantization_preprocessor,
            )

            shutil.copyfile(onnx_path / "config.json", Path(tmpdirname) / "config.json")

            model = ORTModelForSequenceClassification.from_pretrained(tmpdirname, file_name="model-quantized.onnx")

            clx = pipeline("text-classification", model=model, tokenizer=quantizer.tokenizer)

            # evaluate
            test_dataset = load_dataset(
                args.dataset_id, name=args.dataset_config if args.dataset_config != "default" else None, split="test"
            )
            metric = evaluate.load("accuracy")
            results = evaluate_model(clx, test_dataset, metric)

        return results["accuracy"]

    study = optuna.create_study(direction="maximize", study_name=f"quantize-{args.model_id}")
    study.optimize(static_quantization, n_trials=args.n_trials)

    print(study.best_params)  # E.g. {'x': 2.002108042}


if __name__ == "__main__":
    args = parse_args()

    main(args)
