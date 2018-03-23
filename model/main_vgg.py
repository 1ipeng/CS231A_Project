import sys
from utils import Params, plotLabImage
from main_utils import argument_parser, load_training_set, load_dev_test_set, show1Result, showResult, showFinalResult
import os
from train_evaluate import train_evaluate
from vgg_model import vgg_model
import numpy as np

args = argument_parser(sys.argv)

# Load data
# 50,000/5,000/5,000
params = Params("../experiments/base_model/params.json")
train_resized_images, train_labels = load_training_set(args, seed = 318)
dev_resized_images, dev_labels, test_resized_images, test_labels = load_dev_test_set(args, dev_size = 100, seed = 318)

# Weight directory
model_dir = "./weights_transfer_learning"
if not os.path.exists(model_dir):
	os.mkdir(model_dir)
best_path = os.path.join(model_dir, 'best_weights')
last_path = os.path.join(model_dir, 'last_weights')

# Build model
train_evaluate = train_evaluate(params, vgg_model, "../../CS230_Project/model/vgg16_weights.npz")

# Train and predict
if args.train:
	if args.restore:
	    train_evaluate.train(train_resized_images, train_labels, dev_resized_images, dev_labels, model_dir, last_path)
	else:
	    train_evaluate.train(train_resized_images, train_labels, dev_resized_images, dev_labels, model_dir)

if args.predict:
    index = np.array([74, 70, 65, 69, 63])
    save_dir = "./result/"
    X = dev_resized_images[index]
    Y = dev_labels[index]

    # show1Result(train_evaluate, X, Y, 0, last_path)

    # X = train_resized_images
    # Y = train_labels
    # show1Result(train_evaluate, X, Y, 0, last_path)
    showFinalResult(train_evaluate, X, Y, last_path, save_dir)
    




