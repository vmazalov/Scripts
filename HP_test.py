from __future__ import print_function
import numpy as np
import cntk as C
from cntk.learners import sgd, learning_rate_schedule, UnitType
from cntk.logging import ProgressPrinter
from cntk.layers import Dense, Sequential

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def generate_random_data(sample_size, feature_dim, num_classes):
     # Create synthetic data using NumPy.
     Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

     # Make sure that the data is separable
     X = (np.random.randn(sample_size, feature_dim) + 3) * (Y + 1)
     X = X.astype(np.float32)
     # converting class 0 into the vector "1 0 0",
     # class 1 into vector "0 1 0", ...
     class_ind = [Y == class_number for class_number in range(num_classes)]
     Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
     return X, Y

def ffnet(args):
    lr, mb = args
    inputs = 2
    outputs = 2
    layers = 2
    hidden_dimension = 50

    # input variables denoting the features and label data
    features = C.input((inputs), np.float32)
    label = C.input((outputs), np.float32)

    # Instantiate the feedforward classification model
    my_model = Sequential ([
                    Dense(hidden_dimension, activation=C.sigmoid),
                    Dense(outputs)])
    z = my_model(features)

    ce = C.cross_entropy_with_softmax(z, label)
    pe = C.classification_error(z, label)

    # Instantiate the trainer object to drive the model training
    lr_per_minibatch = learning_rate_schedule(lr, UnitType.minibatch)
    progress_printer = ProgressPrinter(0)
    trainer = C.Trainer(z, (ce, pe), [sgd(z.parameters, lr=lr_per_minibatch)], [progress_printer])

    # Get minibatches of training data and perform model training
    minibatch_size = mb
    num_minibatches_to_train = 10
    print("learning rate ", lr)
    aggregate_loss = 0.0
    for i in range(num_minibatches_to_train):
        train_features, labels = generate_random_data(minibatch_size, inputs, outputs)
        # Specify the mapping of input variables in the model to actual minibatch data to be trained with
        trainer.train_minibatch({features : train_features, label : labels})
        sample_count = trainer.previous_minibatch_sample_count
        aggregate_loss += trainer.previous_minibatch_loss_average * sample_count

    last_avg_error = aggregate_loss / trainer.total_number_of_samples_seen

    test_features, test_labels = generate_random_data(minibatch_size, inputs, outputs)
    avg_error = trainer.test_minibatch({features : test_features, label : test_labels})
    print(' error rate on an unseen minibatch: {}'.format(avg_error))
    return avg_error

np.random.seed(98052)
trials = Trials()
best = fmin(ffnet,
    space=[hp.uniform('lr', 0.1, 0.2), hp.quniform('mb', 20, 40, 2)],
    algo=tpe.suggest,
    max_evals=100,
    trials=trials)

print (best)