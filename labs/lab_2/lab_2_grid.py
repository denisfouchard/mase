import os
from chop.tools import get_tokenized_dataset
from optuna.samplers import GridSampler, TPESampler
import optuna
import torch.nn as nn
from chop.nn.modules import Identity
from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools.utils import deepsetattr
from chop.tools import get_trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
                f"linear_layer_choices",
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

    # Define the model
    model = construct_model(trial)
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    # Set the model as an attribute so we can fetch it later
    trial.set_user_attr("model", model)

    return eval_results["eval_accuracy"]


if __name__ == "__main__":
    sampler = GridSampler(search_space=search_space)
    study = optuna.create_study(
        sampler=sampler,
    )
    study.optimize(
        objective,
        n_trials=10,
        timeout=60 * 60 * 24,
        n_jobs=20,
    )

    # Save the results to a file
    study.trials_dataframe().to_csv("results_grid_sampler.csv")

    sampler = TPESampler()
    study = optuna.create_study(
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=10,
        timeout=60 * 60 * 24,
        n_jobs=20,
    )

    # Save the results to a file
    study.trials_dataframe().to_csv("results_tpe_sampler.csv")
