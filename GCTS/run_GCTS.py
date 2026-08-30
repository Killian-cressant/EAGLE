from hyperparams import get_args_from_input
from gcts.types import cast_dataset
from dataset.loader import load_training_data
from dataset.dataset import get_data_loader
import random
from dataset.graph import get_edge_index
import torch
from gcts.experiment import Experiment
from gcts.experiment_version_latent_space import Experiment_Latent
from gcts.experiment_version_leed import Experiment_Partial
import numpy as np
from leed.leed_selection import get_all_graph

import time


def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''

    random.seed(seed)

DETERMINISTIC = True
SEED = 42
if DETERMINISTIC:
    init_seed(SEED)


args = get_args_from_input()

name = args.dataset


GPU_NUM= args.device
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

print(args)

dataset=cast_dataset(name)
splitter=load_training_data(dataset=dataset, normalize=True, clean=True)

train_set, val_set, test_set, train_label, val_label, test_label=splitter[0],splitter[1], splitter[2], splitter[3], splitter[4], splitter[5]
print(f"Initial data shapes: Train: {train_set.shape}, Val: {val_set.shape}, Test: {test_set.shape}")

#init load graphs
edge_index_list_train = get_edge_index(train_set, device, name)
edge_index_list_val = get_edge_index(val_set, device, name, "val")
edge_index_list_test = get_edge_index(test_set, device, name, "test")


train_loader = get_data_loader(
        dataset_name=dataset,
        X=train_set,
        edge_index=edge_index_list_train,
        y=train_label,
        window_size=args.window_size,
        batch_size=args.batch_size,
        horizon=args.horizon,
        shuffle= False, 
        graph_type=args.graph_type,
        top_k=args.top_k,
        cache_path=f"/my_cache_path" 
    )

val_loader = get_data_loader(
        dataset_name=dataset,
        X=val_set,
        edge_index=edge_index_list_val,
        y=val_label,
        window_size=args.window_size,
        batch_size=args.batch_size,
        horizon=args.horizon,
        shuffle=False,
        graph_type=args.graph_type,
        top_k=args.top_k,
        cache_path=f"/my_cache_path"
)

test_loader=get_data_loader(
    dataset_name=dataset,
     X=test_set,
     edge_index=edge_index_list_test,
     y=test_label,
     window_size=args.window_size,
     batch_size=args.batch_size,
     horizon=args.horizon,
     shuffle=False,
     graph_type=args.graph_type,
     top_k=args.top_k,
     cache_path=f"/my_cache_path"
)

#virtual nodes compute




validation_accuracies = []
test_accuracies = []
time_last=[]

trained_experiment = None
centralities_test=None
for trial in range(args.num_trials): #or rm Latent !
    if args.experiment_name == 'LTS':
        exp = Experiment_Latent(args=args, train_dataset=train_loader,
                         validation_dataset=val_loader, test_dataset=test_loader)
    elif args.experiment_name == 'FTS':
        exp = Experiment(args=args, train_dataset=train_loader,
                         validation_dataset=val_loader, test_dataset=test_loader)
    elif args.experiment_name == 'LEED':

        #compute LEED
        centralities_train=get_all_graph(loader=train_loader, args=args, data_type="train", dataset_name=dataset)
        print("Centralities Train done")
        centralities_val=get_all_graph(loader=val_loader, args=args, data_type="val", dataset_name=dataset)
        print("Centralities Val done")
        centralities_test=get_all_graph(loader=test_loader, args=args, data_type="test", dataset_name=dataset)
        print("Centralities Test done")

        #then experiment latent
        exp=Experiment_Partial(args=args, train_dataset=train_loader,
                         validation_dataset=val_loader, test_dataset=test_loader,
                        centralities_train=centralities_train, centralities_val=centralities_val,
                        centralities_test=centralities_test)

    else:
        raise KeyError(f"Experiment name '{args.experiment_name}' is not recognized.")


    start= time.time()
    validation_acc, test_acc = exp.run()
    elapsed = time.time() - start
    validation_accuracies.append(validation_acc)
    test_accuracies.append(test_acc)
    time_last.append(elapsed)



    if trained_experiment is None:
        trained_experiment = exp 


    print("trial:", trial)
    print("test acc:", test_acc)
    print(f"train took {elapsed:.2f}s")


if args.metric != 'all':
    val_mean = 100 * np.mean(validation_accuracies)
    test_mean = 100 * np.mean(test_accuracies)
    time_mean= np.mean(time_last)
    #energy_mean = 100 * np.mean(energies)
    val_ci = 200 * np.std(validation_accuracies)/(args.num_trials ** 0.5)
    test_ci = 200 * np.std(test_accuracies)/(args.num_trials ** 0.5)
    #energy_ci = 200 * np.std(energies)/(args.num_trials ** 0.5)
    print(f"Validation Accuracy: {val_mean:.2f} ± {val_ci:.2f}")
    print(f"Test Accuracy: {test_mean:.2f} ± {test_ci:.2f}")
    print(f"Average Training Time: {time_mean:.2f}s")
    with open("output.txt", "a") as f:
        f.write(f"validation Accuracy: {val_mean}, Test Accuracy:{test_mean}\n, Training Time: {time_mean:.2f}s\n")

else:
    metric_names = validation_accuracies[0].keys()

    val_means = {}
    test_means = {}
    val_cis = {}
    test_cis = {}

    for metric in metric_names:
        # Extract values for this metric across trials
        val_values = np.array([d[metric] for d in validation_accuracies])
        test_values = np.array([d[metric] for d in test_accuracies])

        # Mean
        val_mean = 100 * np.mean(val_values)
        test_mean = 100 * np.mean(test_values)
        time_mean= np.mean(time_last)

        # Confidence interval (same formula you used)
        val_ci = 200 * np.std(val_values) / (args.num_trials ** 0.5)
        test_ci = 200 * np.std(test_values) / (args.num_trials ** 0.5)

        val_means[metric] = val_mean
        test_means[metric] = test_mean
        val_cis[metric] = val_ci
        test_cis[metric] = test_ci

        print(f"{metric.upper()} → "
              f"Val: {val_mean:.2f} ± {val_ci:.2f} | "
              f"Test : {test_mean:.2f} ± {test_ci:.2f}")

    # Optional: save to file
    with open("output.txt", "a") as f:
        for metric in metric_names:
            f.write(
                f"{metric} | "
                f"Val: {val_means[metric]:.4f} ± {val_cis[metric]:.4f} | "
                f"Test : {test_means[metric]:.4f} ± {test_cis[metric]:.4f}\n"
                f"Training Time: {time_mean:.2f} ± {np.std(time_last):.2f}\n"
            )


if args.experiment_name == 'LEED':
    trained_experiment.visualize_reconstruction_leed(
        loader=test_loader,
        ts_indices=[0,1,2,3], #[10, 11, 12, 0, 3, 4]
        save_dir="recon_ts_plots",
        tag="trained",
        centralities=centralities_test
    )
else:
    trained_experiment.visualize_reconstruction(
    loader=test_loader,
    ts_indices=[10, 11, 12, 0, 3, 4],
    save_dir="recon_ts_plots",
    tag="trained"
)