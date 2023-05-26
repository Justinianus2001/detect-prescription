import cv2
import os
import pandas as pd

images_dir = "./datasets/train/images"
labels_json_dir = "./datasets/train/labels_json"
labels_dir = "./datasets/train/labels"
for file in os.listdir(labels_json_dir):
    filename, ext = os.path.splitext(file)
    img = cv2.imread(images_dir + "/" + filename + ".png")
    height, width, size = img.shape
    df = pd.read_json(labels_json_dir + "/" + file)
    lst_label = ["date", "diagnose", "drugname", "quantity", "usage", "other"]
    with open(labels_dir + "/" + filename + ".txt", "w") as f:
        for i in range(len(df["box"])):
            f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(lst_label.index(df["label"][i]),
                                                              (df["box"][i][0] + df["box"][i][2]) / (width * 2),
                                                              (df["box"][i][1] + df["box"][i][3]) / (height * 2),
                                                              (df["box"][i][2] - df["box"][i][0]) / width,
                                                              (df["box"][i][3] - df["box"][i][1]) / height))
        f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(6, 100 / (width * 2), 100 / (height * 2), 100 / width, 100 / height))
        f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(6, 1 - (100 / (width * 2)), 100 / (height * 2), 100 / width, 100 / height))
        f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(6, 100 / (width * 2), 1 - (100 / (height * 2)), 100 / width, 100 / height))
        f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(6, 1 - (100 / (width * 2)), 1 - (100 / (height * 2)), 100 / width, 100 / height))