from models.CTM import CTM
from models.ProdLDA import ProdLDA
from models.pytorchavitm import AVITM
from dataset.dataset import Dataset
from utils.optimization.optimizer import Optimizer
from skopt.space.space import Real, Categorical, Integer
from utils.metrics.coherence_metrics import Coherence

dataset = Dataset()
dataset.fetch_dataset("M10")

model = ProdLDA()


# Train the model using default partitioning choice
output = model.train_model(dataset)

print(*list(output.keys()), sep="\n") # Print the output identifiers


# model = CTM(num_topics=10, num_epochs=30, inference_type='zeroshot', bert_model="bert-base-nli-mean-tokens")
npmi = Coherence(texts=dataset.get_corpus())
search_space = {"num_layers": Categorical({1, 2, 3}),
                "num_neurons": Categorical({100, 200, 300}),
                "activation": Categorical({'sigmoid', 'relu', 'softplus'}),
                "dropout": Real(0.0, 0.95)
                }

optimization_runs = 30
model_runs = 1

optimizer = Optimizer()
optimization_result = optimizer.optimize(
    model, dataset, npmi, search_space, number_of_call=optimization_runs,
    model_runs=model_runs, save_models=True,
    extra_metrics=None,  # to keep track of other metrics
    save_path='results/test_ctm//')

print(optimization_result)
