import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import Params, plotLabImage
from skimage import color

def argument_parser(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", help="restore training from last epoch",
                        action="store_true")
    parser.add_argument("--train", help="train model",
                        action="store_true")
    parser.add_argument("--predict", help="show predict results",
                        action="store_true")
    parser.add_argument("--small", help="train on small dataset",
                        action="store_true")
    parser.add_argument("--toy", help="train on toy dataset",
                        action="store_true")
    parser.add_argument("--superlarge", help="train on superlarge dataset",
                        action="store_true")
    if len(argv) < 2:
        parser.print_usage()
        exit()
    else:
        return parser.parse_args()

def load_training_set(args, size = None, seed = None):
    DIR_TRAIN = "../../CS230_Project/data/lab_result/train_lab/"
    if args.toy:
        DIR_TRAIN = "../../CS230_Project/data/lab_result/100_train_lab/"  
    if args.superlarge:
        DIR_TRAIN = "../../CS230_Project/data/lab_result/super_train_lab/"


    if seed is not None:
        np.random.seed(seed)

    if size is None:
        if args.toy:
            size = 100
        elif args.small:
            size = 5000
        elif args.superlarge:
            size = 200000
        else:
            size = 50000

    train_resized_images = np.load(DIR_TRAIN+"resized_images.npy")
    train_labels = np.load(DIR_TRAIN+"labels.npy")

    m = train_resized_images.shape[0]
    permutation = list(np.random.permutation(m))

    train_resized_images = train_resized_images[permutation[0:size]]
    train_labels = train_labels[permutation[0:size]]

    return train_resized_images, train_labels

def load_dev_test_set(args, dev_size = None, seed = None):
    DIR_TEST = "../../CS230_Project/data/lab_result/test_lab/"
    if args.toy:
        DIR_TEST = "../../CS230_Project/data/lab_result/100_test_lab/"
    if args.superlarge:
        DIR_TEST = "../../CS230_Project/data/lab_result/super_test_lab/"

    if seed is not None:
        np.random.seed(seed)

    if dev_size is None:
        if args.toy:
            dev_size = 30
        elif args.small:
            dev_size = 500
        else:
            dev_size = 5000

    test_dev_resized_images = np.load(DIR_TEST + "resized_images.npy")
    test_dev_labels = np.load(DIR_TEST + "labels.npy")

    m = test_dev_resized_images.shape[0]
    permutation = list(np.random.permutation(m))
    dev_index = permutation[0:dev_size]
    test_index = permutation[dev_size:]

    # Build dev/test sets
    dev_resized_images = test_dev_resized_images[dev_index]
    dev_labels = test_dev_labels[dev_index]

    test_resized_images = test_dev_resized_images[test_index]
    test_labels = test_dev_labels[test_index]

    return dev_resized_images, dev_labels, test_resized_images, test_labels


def load_training_dev_test_set(args, dev_size = 2500, seed = None):
    DIR_TRAIN = "../../CS230_Project/data/lab_result/train_lab/"
    DIR_TEST = "../../CS230_Project/data/lab_result/test_lab/"

    train_L = np.load(DIR_TRAIN + "L.npy")
    train_ab = np.load(DIR_TRAIN + "ab.npy")
    train_bins = np.load(DIR_TRAIN + "bins.npy")
    train_grayRGB = np.load(DIR_TRAIN + "grayRGB.npy")

    test_dev_L = np.load(DIR_TEST + "L.npy")
    test_dev_ab = np.load(DIR_TEST + "ab.npy")
    test_dev_bins = np.load(DIR_TEST + "bins.npy")
    test_dev_grayRGB = np.load(DIR_TEST + "grayRGB.npy")

    m = test_dev_L.shape[0]
    permutation = list(np.random.permutation(m))
    dev_index = permutation[0:dev_size]
    test_index = permutation[dev_size:2 * dev_size]
    train_index = permutation[2 * dev_size:]

    dev_L = test_dev_L[dev_index]
    dev_ab = test_dev_ab[dev_index]
    dev_bins = test_dev_bins[dev_index]
    dev_grayRGB = test_dev_grayRGB[dev_index]

    test_L = test_dev_L[test_index]
    test_ab = test_dev_ab[test_index]
    test_bins = test_dev_bins[test_index]
    test_grayRGB = test_dev_grayRGB[test_index]

    train_L = np.concatenate((train_L, test_dev_L[train_index]), axis=0)
    train_ab = np.concatenate((train_ab, test_dev_ab[train_index]), axis=0)
    train_bins = np.concatenate((train_bins, test_dev_bins[train_index]), axis=0)
    train_grayRGB = np.concatenate((train_grayRGB, test_dev_grayRGB[train_index]), axis=0)

    return train_L, train_ab, train_bins, train_grayRGB, dev_L, dev_ab, dev_bins, dev_grayRGB, test_L, test_ab, test_bins, test_grayRGB  

def show1Result(train_evaluate, X, Y, start_index, save_path):
    labels_dict = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    predict_cost, predict_logits, predict_accuracy, predict_probs, predict_predictions = train_evaluate.predict(X[start_index:start_index + 1], Y[start_index:start_index + 1], save_path)
    plt.imshow(X[start_index])
    print("prediction:", labels_dict[int(predict_predictions[0])])
    print("predict_probs",predict_probs)
    print("cost:", predict_cost)
    print("accuracy:", predict_accuracy)
    plt.show()

def showResult(train_evaluate, X, Y, save_path):
    labels_dict = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    predict_cost, predict_logits, predict_accuracy, predict_probs, predict_predictions = train_evaluate.predict(X, Y, save_path)
    return predict_accuracy


def showFinalResult(train_evaluate, X, Y, save_path,  save_dir,):
    labels_dict = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    predict_cost, predict_logits, predict_accuracy, predict_probs, predict_predictions = train_evaluate.predict(X, Y, save_path)


    for i in range(5):
        one_set_X = np.zeros((4, 224, 224, 3))
        one_set_Y = np.ones((4, 1)) * Y[i, 0]

        gray = np.load(save_dir + "gray_" + str(i) + ".npy")
        gray= color.gray2rgb(gray)
        colored = np.load(save_dir+"color_"+str(i)+".npy")
        annealed=np.load(save_dir+"annealed_color_"+str(i)+".npy")
        truth = np.load(save_dir+"ground_truth_"+str(i)+".npy")

        one_set_X[0, :, :, :] = gray
        one_set_X[1, :, :, :] = colored
        one_set_X[2, :, :, :] = annealed
        one_set_X[3, :, :, :] = truth

        predict_cost, predict_logits, predict_accuracy, predict_probs, predict_predictions = train_evaluate.predict(one_set_X, one_set_Y, save_path)
        print("Truth:", labels_dict[Y[i, 0].astype(int)])
        print("Predictions:", labels_dict[predict_predictions.astype(int)])
        print("Accuracy:", predict_accuracy)
        #print("Probs:", predict_probs)
        print("Probs:", np.max(predict_probs,axis=1))

    return predict_accuracy








