import os
import torch
from chop.tools import get_tokenized_dataset
from optuna.samplers import GridSampler, TPESampler
import optuna
import torch.nn as nn
from chop.nn.modules import Identity
from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools.utils import deepsetattr
from chop.tools import get_trainer
from chop.pipelines import CompressionPipeline
from chop import MaseGraph

device = torch.device("cuda")

checkpoint = "DeepWokLab/bert-tiny"
tokenizer_checkpoint = "DeepWokLab/bert-tiny"
dataset_name = "imdb"

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)


# Define the search space
search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    "linear_layer_choices": [
        nn.Linear,
        Identity,
    ],
}



def construct_model(trial):
    config = AutoConfig.from_pretrained(checkpoint)

    # Update the paramaters in the config
    for param in [
        "num_layers",
        "num_heads",
        "hidden_size",
        "intermediate_size",
    ]:
        chosen_idx = trial.suggest_categorical(name=param, choices=search_space[param])
        #chosen_idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
        #print(f"Chosen idx {chosen_idx} for param {param} ({len(search_space[param])})")
        setattr(config, param, chosen_idx)

    trial_model = AutoModelForSequenceClassification.from_config(config)

    for name, layer in trial_model.named_modules():
        if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
            new_layer_cls = trial.suggest_categorical(
                f"{name}_type",
                search_space["linear_layer_choices"],
            )
            if new_layer_cls == nn.Linear:
                continue
            elif new_layer_cls == Identity:
                new_layer = Identity()
                deepsetattr(trial_model, name, new_layer)
            else:
                raise ValueError(f"Unknown layer type: {new_layer_cls}")

    return trial_model


def objective(trial):
    quantization_config = {
            "by": "type",
            "default": {
                "config": {
                    "name": None,
                }
            },
            "linear": {
                "config": {
                    "name": "integer",
                    # data
                    "data_in_width": 16,
                    "data_in_frac_width": 8,
                    # weight
                    "weight_width": 16,
                    "weight_frac_width": 8,
                    # bias
                    "bias_width": 16,
                    "bias_frac_width": 8,
                }
            },
        }
        
    pruning_config = {
        "weight": {
            "sparsity": 0.3,
            "method": "l1-norm",
            "scope": "local",
        },
        "activation": {
            "sparsity": 0.3,
            "method": "l1-norm",
            "scope": "local",
        },
    }
    # Define the model
    model = construct_model(trial)
    model.to(device)
    print(device)
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=2,
    )

    trainer.train()

    mg = MaseGraph(model, hf_input_names=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
    mg.model.to("cpu")
    pipe = CompressionPipeline()

    mg, _ = pipe(
        mg,
        pass_args={
            "quantize_transform_pass": quantization_config,
            "prune_transform_pass": pruning_config,
        },
    )
    model = mg.model.to(device)
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    
    # Set the model as an attribute so we can fetch it later
    
    trainer.train()
    eval_results = trainer.evaluate()
    # Flush the cache
    torch.cuda.empty_cache()
    # Delete the model from the GPU
    del model
    return eval_results["eval_accuracy"]


if __name__ == "__main__":
    sampler = TPESampler()
    study = optuna.create_study(
        direction="maximize",
        study_name="bert-tiny-compression-pipeline",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=20,
        timeout=60 * 60 * 24,
        n_jobs=1,
    )

    # Save the results to a file
    study.trials_dataframe().to_csv("results_compression.csv")