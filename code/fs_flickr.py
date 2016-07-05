from sample import *
from evolutionary import *
from spvlinkprediction import *
from wrapper import *

sample = CDConstructor("flickr_al.txt", 10000, (0.5, 0.5))
sample.set_classification_dataset('flickr_final_version_attributes', 'flickr_final_version')
table = sample.get_classification_dataset()
number_of_attributes = len(sample.ordered_attributes_list)
message = "Dataset Flickr\n"
metrics = ["precision", "f1", "roc_auc"]
classifiers = (("MLP", {"activation": "logistic", "hidden_layer_sizes": (4,), "learning_rate_init": 0.7}),
               ("MLP", {"activation": "logistic", "hidden_layer_sizes": (6,), "learning_rate_init": 0.7}),
               ("MLP", {"activation": "logistic", "hidden_layer_sizes": (8,), "learning_rate_init": 0.7}),
               ("MLP", {"activation": "logistic", "hidden_layer_sizes": (10,), "learning_rate_init": 0.7}),
               ("MLP", {"activation": "logistic", "hidden_layer_sizes": (12,), "learning_rate_init": 0.7}),
               ("MLP", {"activation": "logistic", "hidden_layer_sizes": (24,), "learning_rate_init": 0.7}),
               ("MLP", {"activation": "logistic", "hidden_layer_sizes": (8, 10), "learning_rate_init": 0.7}),
               ("MLP", {"activation": "logistic", "hidden_layer_sizes": (16, 12), "learning_rate_init": 0.7}),
               ("MLP", {"activation": "logistic", "hidden_layer_sizes": (24, 20, 16), "learning_rate_init": 0.7}),
               ("MLP", {"activation": "logistic", "hidden_layer_sizes": (20, 24, 16), "learning_rate_init": 0.7})
               )

for metric in metrics:
    for classifier, classifier_params in classifiers:
        BaFE = BackwardFeatureElimination(number_of_attributes, classifier, classifier_params, metric, table[:, range(number_of_attributes)], table[:, number_of_attributes], sample.get_fold_list())
        FFS = ForwardFeatureSelection(number_of_attributes, classifier, classifier_params, metric, table[:, range(number_of_attributes)], table[:, number_of_attributes], sample.get_fold_list())
        CFS = SupervisedLinkPrediction(table, 10, classifier, classifier_params, metric)
        FFS.perform_feature_selection()
        BaFE.perform_feature_selection()

	message = str(classifier_params["hidden_layer_sizes"])
	message.replace(",", "_")

        FS1 = EvolutionaryFeatureSelection(number_of_attributes, classifier, classifier_params, metric, table[:, range(number_of_attributes)], table[:, number_of_attributes], sample.get_fold_list(), 20, 100, 0.65, 0.05)
        FS2 = EvolutionaryFeatureSelection(number_of_attributes, classifier, classifier_params, metric, table[:, range(number_of_attributes)], table[:, number_of_attributes], sample.get_fold_list(), 20, 100, 0.65, 0.05, selection_function="tournament", selection_parameters={"tournsize": 3})
        FS3 = EvolutionaryFeatureSelection(number_of_attributes, classifier, classifier_params, metric, table[:, range(number_of_attributes)], table[:, number_of_attributes], sample.get_fold_list(), 50, 100, 0.65, 0.05, "steady_state_mode")
        FS4 = EvolutionaryFeatureSelection(number_of_attributes, classifier, classifier_params, metric, table[:, range(number_of_attributes)], table[:, number_of_attributes], sample.get_fold_list(), 30, 100, 0.65, 0.05, linear_normalization_generation=10)
        FS1.perform_feature_selection()
        FS2.perform_feature_selection()
        FS3.perform_feature_selection()
        FS4.perform_feature_selection()

	
	message += ",{},[{}],{},[{}],{},[{}],".format(CFS.apply_classifier(), 23, FFS.best_score, len(FFS.best_var_subset), BaFE.best_score, len(BaFE.best_var_subset))

	message += "{},[{}],{},[{}],{},[{}],{},[{}];\n".format(
							       FS1.best_solution.fitness.values[0], sum(i for i in FS1.best_solution),
							       FS2.best_solution.fitness.values[0], sum(i for i in FS2.best_solution),
							       FS3.best_solution.fitness.values[0], sum(i for i in FS3.best_solution),
							       FS4.best_solution.fitness.values[0], sum(i for i in FS4.best_solution),
						               )
	with open("flickr.csv", "a") as res:
		res.write(message)
