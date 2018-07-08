"""
K-Means clustering example of image color reduction. Reduces
number of colors in the given image by clustering the pixels
into k groups. Each pixel is labeled with the nearest cluster
centroid. Resulting array of labels for each pixel is then
converted to the image which can be saved to disk.

Copyright 2018, Ada Piekarska, All rights reserved
"""


from PIL import Image
import numpy as np
from os import path
from kmeans import kmeans
from Exceptions import KValueError,\
    PathDoesntExistError, FileAlreadyExistsError

def main():

    print("Image color palette reduction using K-Means example.\n")
    print("Note: Color palette values are initialized randomly.\n"
          "Therefore the result will vary with each iteration.\n"
          "To get the best results, please run the algorithm\n"
          "a few times and pick the best resulting image.")
    print("====================================================")

    while True:
        try:
            filename = input("Enter path to the image file: ")
            if not path.exists(filename):
                raise PathDoesntExistError
            break
        except PathDoesntExistError:
            print("Path does not exist.")

    while True:
        try:
            k = int(input("Enter k value (number of colors after reduction): "))
            if k <= 0:
                raise KValueError
            break
        except KValueError:
            print("K should be positive integer.")

    # open image
    im = Image.open(filename)
    (width, height) = im.size

    # prepare data set
    pix = list(im.getdata())
    data = np.array(pix)

    # perform k-means
    # important: when reducing image palette, k-means
    # should not be run until convergence. In fact,
    # only running 1 iteration produces the best result.
    print("Performing k-means (it may take a few seconds)...")

    min_vals = [0] * 3
    max_vals = [255] * 3
    labels = kmeans(k, data, min_vals, max_vals, 1)

    # convert k-means result first to ndarray, then to an image
    i = 0
    pix_clustered = np.zeros(shape=(height, width, 3))
    for x in range(height):
        for y in range(width):
            pix_clustered[x, y] = labels[i]
            i += 1
    im = Image.fromarray(np.uint8(pix_clustered))
    im.show()

    save = input("Finished. Do you want to save the result? y/n\n")
    if save == "y":
        while True:
            try:
                out_filename = input("Enter a filename (press Enter for default): ")
                if out_filename == "":
                    out_filename = "out.png"
                    break
                else:
                    if not (out_filename.endswith(".png") or out_filename.endswith(".jpg")):
                        out_filename += ".png"
                        break
                if path.exists(filename):
                    raise FileAlreadyExistsError
                break
            except FileAlreadyExistsError:
                print("File already exists.")
                continue
        try:
            im.save(out_filename)
            print("Saved to", out_filename)
        except OSError:
            print("Invalid filename.")


if __name__ == "__main__":
    main()
