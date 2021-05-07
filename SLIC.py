import numpy as np
import cv2 as cv
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import sys
"""
@author Arjun Albert
@email arjunalbert@brandeis.edu
@date 5/7/2021
"""

distance_weight = 500


def random_pixel(img):
    random_x = random.randint(0, img.shape[0] - 1)
    random_y = random.randint(0, img.shape[1] - 1)
    return random_x, random_y


def assign_pixel(img, pixel, pixel_coordinates, centroids, S):
    min_dist_index = -1
    min_dist = sys.maxsize
    count = 0
    for centroid in centroids:
        centroid_pixel = img[centroid[0], centroid[1]]
        current_dist = distance(pixel, pixel_coordinates,
                                centroid_pixel, (centroid[0], centroid[1]), S)
        if current_dist < min_dist:
            min_dist = current_dist
            min_dist_index = count
        count += 1
    return min_dist_index


def distance(pixel_1, pixel_1_coordinates, pixel_2, pixel_2_coordinates, S):

    l_1 = int(pixel_1[0])
    a_1 = int(pixel_1[1])
    b_1 = int(pixel_1[2])
    l_2 = int(pixel_2[0])
    a_2 = int(pixel_2[1])
    b_2 = int(pixel_2[2])

    x_1 = int(pixel_1_coordinates[0])
    y_1 = int(pixel_1_coordinates[1])
    x_2 = int(pixel_2_coordinates[0])
    y_2 = int(pixel_2_coordinates[1])

    distance_lab = ((l_1 - l_2) ** 2 + (a_1 - a_2)
                    ** 2 + (b_1 - b_2) ** 2) ** 0.5
    distance_xy = ((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2) ** 0.5
    distance_total = distance_lab + (distance_weight/S) * distance_xy
    return distance_total


def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        avg_x = int(sum([pixel_data[1][0]
                         for pixel_data in cluster]) / len(cluster))
        avg_y = int(sum([pixel_data[1][1]
                         for pixel_data in cluster]) / len(cluster))
        new_centroids.append([avg_x, avg_y])
    return new_centroids


def show_centroids(img, centroids):
    plt.imshow(img)
    centroids_x = [centroid[1] for centroid in centroids]
    centroids_y = [centroid[0] for centroid in centroids]
    plt.scatter(centroids_x, centroids_y, color='r')
    plt.show()


def make_superpixel_img(img, clusters):
    new_img = copy.deepcopy(img)
    for cluster in clusters:
        for pixel_data in cluster:
            pixel_data[0] = pixel_data[0].astype('int64')
        avg_cluster_pixel = sum([pixel_data[0]
                                 for pixel_data in cluster])/len(cluster)
        for pixel_data in cluster:
            new_img[pixel_data[1][0], pixel_data[1][1]] = avg_cluster_pixel
    return new_img


def SLIC(img, k):
    height, width, _ = img.shape
    num_pixels = height * width
    superpixel_size = num_pixels / k
    superpixel_radius = superpixel_size ** 0.5

    centroids = [random_pixel(img) for i in range(0, k)]
    clusters = [[] for i in range(0, k)]
    iteration = 0
    while True:
        for row in tqdm(range(0, height)):
            for col in range(0, width):
                pixel = img[row, col]
                cluster_index = assign_pixel(
                    img, pixel, (row, col), centroids, superpixel_radius)
                pixel_data = [pixel, [row, col]]
                clusters[cluster_index].append(pixel_data)
        new_centroids = update_centroids(clusters)
        if new_centroids == centroids:
            break
        else:
            centroids = new_centroids
        cv.imwrite("clusters" + str(k) + "_distweight" + str(distance_weight) + "_slic_iteration" +
                   str(iteration) + ".png", cv.cvtColor(make_superpixel_img(img, clusters), cv.COLOR_Lab2BGR))
        iteration += 1


raw_img = cv.imread("brandeis_campus.jpg")
lab_img = cv.cvtColor(raw_img, cv.COLOR_RGB2Lab)

SLIC(lab_img, 128)
