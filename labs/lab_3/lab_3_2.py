import os
from pathlib import Path
import dill
from chop.tools import get_trainer
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
checkpoint = "DeepWokLab/bert-tiny"
tokenizer_checkpoint = "DeepWokLab/bert-tiny"
dataset_name = "imdb"
import torch
from chop.nn.quantized.modules.linear import (
    LinearInteger,
    LinearMinifloatDenorm,
    LinearMinifloatIEEE,
)
from chop.tools.utils import deepsetattr
from copy import deepcopy
from chop.tools import get_tokenized_dataset

with open(f"{Path.home()}/tutorial_5_best_model.pkl", "rb") as f:
    base_model = dill.load(f)

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

search_space = {
    "linear_layer_choices": [
        torch.nn.Linear,
        LinearInteger,
        LinearMinifloatDenorm,
        LinearMinifloatIEEE,
    ],
    "width_choices": [8, 16, 32],
    "frac_width_choices": [2, 4, 8],
}


def construct_model(trial):

    # Fetch the model
    trial_model = deepcopy(base_model)

    # Quantize layers according to optuna suggestions
    for name, layer in trial_model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            new_layer_cls = trial.suggest_categorical(
                f"{name}_type",
                search_space["linear_layer_choices"],
            )

            if new_layer_cls == torch.nn.Linear:
                continue

            kwargs = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
            }

            # If the chosen layer is integer, define the low precision config
            if new_layer_cls == LinearInteger:
                width = trial.suggest_categorical(
                    "width",
                    search_space["width_choices"],
                )
                frac_width = trial.suggest_categorical(
                    "frac_width",
                    search_space["frac_width_choices"],
                )
                kwargs["config"] = {
                    "data_in_width": width,
                    "data_in_frac_width": frac_width,
                    "weight_width": width,
                    "weight_frac_width": frac_width,
                    "bias_width": width,
                    "bias_frac_width": frac_width,
                }
            elif new_layer_cls == LinearMinifloatDenorm:
                width = trial.suggest_categorical(
                    "width",
                    search_space["width_choices"],
                )
                exponent_width = trial.suggest_categorical(
                    "exponent_width",
                    search_space["frac_width_choices"],
                )
                exponent_bias = trial.suggest_categorical(
                    "exponent_bias",
                    search_space["frac_width_choices"],
                )
                kwargs["config"] = {
                    "data_in_width": width,
                    "data_in_exponent_width": exponent_width,
                    "data_in_exponent_bias": exponent_bias,
                    "weight_width": width,
                    "weight_exponent_width": exponent_width,
                    "weight_exponent_bias": exponent_bias,
                    "bias_width": width,
                    "bias_exponent_width": exponent_width,
                    "bias_exponent_bias": exponent_bias,
                }

            elif new_layer_cls == LinearMinifloatIEEE:
                width = trial.suggest_categorical(
                    "width",
                    search_space["width_choices"],
                )
                exponent_width = trial.suggest_categorical(
                    "exponent_width",
                    search_space["frac_width_choices"],
                )
                exponent_bias = trial.suggest_categorical(
                    "exponent_bias",
                    search_space["frac_width_choices"],
                )
                kwargs["config"] = {
                    "data_in_width": width,
                    "data_in_exponent_width": exponent_width,
                    "data_in_exponent_bias": exponent_bias,
                    "weight_width": width,
                    "weight_exponent_width": exponent_width,
                    "weight_exponent_bias": exponent_bias,
                    "bias_width": width,
                    "bias_exponent_width": exponent_width,
                    "bias_exponent_bias": exponent_bias,
                }

            # Create the new layer (copy the weights)
            new_layer = new_layer_cls(**kwargs)
            new_layer.weight.data = layer.weight.data

            # Replace the layer in the model
            deepsetattr(trial_model, name, new_layer)

    return trial_model


def objective(trial):

    # Define the model
    model = construct_model(trial)

    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=0.02,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    trial.set_user_attr("model", model)
    # Flush the model to save memory
    del model
    torch.cuda.empty_cache()

    return eval_results["eval_accuracy"]


if __name__ == "__main__":
    from optuna.samplers import GridSampler, RandomSampler, TPESampler
    import optuna

    sampler = RandomSampler()
    study = optuna.create_study(
        direction="maximize",
        study_name="bert-tiny-nas-study",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=20,
        timeout=60 * 60 * 24,
    )
    study.trials_dataframe().to_csv("results_3_2.csv")
