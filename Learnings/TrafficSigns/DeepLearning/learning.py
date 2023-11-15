from Convolutional.networks import build_lenet5, train_model_with_generator, build_convolutional_model
from preprocessing.preprocessing import lazy_load_and_augment_batches, lazy_load_test_batches


def hyperparameter_search(search_ranges:dict, params:dict, layer_list:list, path_to_train:str, path_to_test=None) -> list:
    result_list, final_losses, final_accuracies = [], [], []
    best_loss, best_accuracy = 9999999, 0
    best_run_loss, best_run_acc = None, None
    for batch_size in search_ranges["batch_size"]:
        params["batch_size"] = batch_size
        for epochs in search_ranges["epochs"]:
            params["epochs"] = epochs
            for learning_rate in search_ranges["learning_rate"]:
                for frac in [1.0, 0.9, 0.8, 0.7, 0.5, 0.3]:
                    params["learning_rate"] = learning_rate
                    model_own = build_convolutional_model(layer_list=layer_list, params=params)
                    print(f"Training model for config {params}")
                    train_generator = lazy_load_and_augment_batches(path_to_train,
                                                                    batch_size=params["batch_size"],
                                                                    subset='training',
                                                                    dataset_fraction=frac,
                                                                    validation_split=params["validation_split"])
                    validation_generator = lazy_load_test_batches(path_to_test, batch_size=params["batch_size"])

                    history = train_model_with_generator(train_generator, validation_generator, model_own, params, "saved_models/model" + str(frac), "saved_models/model_acc" + str(frac) + ".txt")
                    print("finished")


    print(final_losses)
    print(f"Best loss: {best_loss}")
    print(f"Best run (loss): {best_run_loss}")
    print(final_accuracies)
    print(f"Best accuracy: {best_run_acc}")
    print(f"Best run (acc): {best_accuracy}")
    return result_list
