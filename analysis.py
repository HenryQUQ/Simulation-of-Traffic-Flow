import os
import shutil
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import re

def findall_file():
    """
    Find all file names in current path
    :return: list of all files
    """
    filepath = "./"  # current path
    filelist = []
    for _, _, file in os.walk(filepath):
        filelist.append(file)
    return filelist



def pattern_fit_files(filelist, pattern):
    """
    Find out the files which is fitted to the pattern
    :param filelist: list of all files
    :param pattern: string which is the pattern need to be fitted
    :return: list of all files we need
    """
    file_needed = []
    for type in filelist:  # because os.walk classifies the files with their file type
        for item in type:  # match for each files
            m = pattern.findall(item)
            if not m == []:
                file_needed.append(m[0])
    return file_needed


def get_density_and_tf_length(file_needed, pattern):
    """
    Return the density and traffic flow while length of road is changing.
    :param file_needed: name of the file which is waiting to be processed
    :param pattern: string which is the pattern need to be fitted
    :return: the density and traffic flow
    """
    density = []
    tf = []
    for item in file_needed:
        one_tf_value = get_tf_value(item)  # get the average traffic flow in this file
        tf.append(one_tf_value)
        length = item.replace(pattern, '')  # get the corresponding length of road
        density.append(20 / int(length))

    return density, tf
def get_tf_value(file):
    """
    Open the file and return the average traffic flow in it.
    :param file: the name of the file we needed
    :return: average traffic flow
    """
    f = open(file)
    contain = []
    for line in f.readlines():
        contain.append(float(line.replace('\n', '')))
    return contain[0]  # average traffic flow is on first line

def round_to_3_sf(data):
    """
    Standardize the data to 3 significant figures
    :param data: float of raw data
    :return: string with 3 significant figures
    """
    return '%s' % float('%.3g' % data)


def plot_fitting_line_length(density, tf,a,b,label):
    """
    Plot the best fit line within the range
    :param density: cars per site on the road
    :param tf: traffic flow, cars passing the end point
    :param range: list with 2 elements, first one is the starting point of best fit line, second one is the end point.
    :param label: label for best fit line
    :return slope_text: string for the slope of best fit line
    :return intercept_text: string for the intercept of best fit line
    """
    density_in_range = []
    tf_in_range = []

    for index in range(len(density)):
        if a <= density[index] <= b:  # we need the density only within the range to plot a best fit line
            density_in_range.append(density[index])
            tf_in_range.append(tf[index])
    # get the data and error for the best fit line
    fitting_data, fitting_error = np.polyfit(density_in_range, tf_in_range, 1, cov=True)
    fitting_y = []
    for one_desity in density_in_range:
        fitting_y.append(one_desity * fitting_data[0] + fitting_data[1])
    plt.plot(density_in_range, fitting_y, label=label)

    # create the text for showing the slope and intercept with their error
    slope_text = ('Slope = ' + round_to_3_sf(fitting_data[0]) + ' +/- ' + round_to_3_sf(
        abs(min(fitting_error[0][0], fitting_error[0][1]))))
    intercept_text = ('Intercept = ' + round_to_3_sf(fitting_data[1]) + ' +/- ' + round_to_3_sf(
        abs(min(fitting_error[1][0], fitting_error[1][1]))))

    return slope_text, intercept_text

def get_density_and_tf_car_number(file_needed, pattern):
    """
    Return the density and traffic flow while number of cars in the road is changing.
    :param file_needed: name of the file which is waiting to be processed
    :param pattern: string which is the pattern need to be fitted
    :return: the density and traffic flow
    """
    density = []
    tf = []
    for item in file_needed:
        one_tf_value = get_tf_value(item)  # get the average traffic flow in this file
        tf.append(one_tf_value)
        car_number_raw = item.replace('500_', '')  # get the corresponding length of road
        car_number = car_number_raw.replace(pattern, '')
        density.append(int(car_number) / 500)
    return density, tf

def car_number_change_with_different_probability(filelist):
    """
    Main function to plot the traffic flow against density graph with different probability
    while the number of cars is changing.
    :param filelist: list of all files
    :return: True if successive
    """
    # Find all file names with matched
    pattern_025 = re.compile(r'500_\d+_5_0.25_tf.txt')
    file_needed_025 = pattern_fit_files(filelist, pattern_025)
    pattern_050 = re.compile(r'500_\d+_5_0.5_tf.txt')
    file_needed_050 = pattern_fit_files(filelist, pattern_050)
    pattern_075 = re.compile(r'500_\d+_5_0.75_tf.txt')
    file_needed_075 = pattern_fit_files(filelist, pattern_075)

    # get the density and traffic flow for each of them
    density_025, tf_025 = get_density_and_tf_car_number(file_needed_025, '_5_0.25_tf.txt')
    density_050, tf_050 = get_density_and_tf_car_number(file_needed_050, '_5_0.5_tf.txt')
    density_075, tf_075 = get_density_and_tf_car_number(file_needed_075, '_5_0.75_tf.txt')

    # Plot the graph
    title = 'The Best Fit lines With Different Probabilities While Car Number Changes'
    plt.title(title)
    # plot_density_tf_graph_probability(density_025, density_050, density_075, tf_025, tf_050, tf_075)
    a, b = plot_fitting_line_length(density_025, tf_025,0.1,0.6,'best fit when p = 0.25')
    c, d = plot_fitting_line_length(density_050, tf_050, 0.08, 0.6, 'best fit when p = 0.50')
    e, f = plot_fitting_line_length(density_075, tf_075, 0.06, 0.6, 'best fit when p = 0.75')
    print(a,b)
    print(c,d)
    print(e,f)
    plt.legend()
    plt.show()

    return True


def plot_density_tf_graph_probability(density_025, density_050, density_075, tf_025, tf_050, tf_075):
    """
    Plot the traffic flow against density graph with different probability.
    :param density_025: density at probability of 0.25
    :param density_050: density at probability of 0.50
    :param density_075: density at probability of 0.75
    :param tf_025: traffic flow at probability of 0.25
    :param tf_050: traffic flow at probability of 0.50
    :param tf_075: traffic flow at probability of 0.75
    :return: True if successive
    """
    plt.plot(density_025, tf_025, '.', label='traffic flow at p = 0.25')
    plt.plot(density_050, tf_050, '.', label='traffic flow at p = 0.50')
    plt.plot(density_075, tf_075, '.', label='traffic flow at p = 0.75')
    plt.xlabel('density')
    plt.ylabel('traffic flow')

    return True
filelist = findall_file()
# pattern_0 = re.compile(r'500_\d+_5_0_tf.txt')
# file_needed_0 = pattern_fit_files(filelist, pattern_0)
# density_0, tf_0 = get_density_and_tf_car_number(file_needed_0, '_5_0_tf.txt')
# title = 'Traffic Flow Against Density When p = 0 While Num Changes'
# plt.title(title)
# plt.plot(density_0, tf_0, '.', label='traffic flow at p = 0')
# plt.xlabel('density')
# plt.ylabel('traffic flow')
# # if best_fitting_para is True:
# print(len(density_0))
# slope_txt, intercept_txt = plot_fitting_line_length(density_0, tf_0, 'best fit line')
# plt.text(0.22, 0.4, slope_txt)
# plt.text(0.22, 0.35, intercept_txt)
# plt.legend()
# plt.show()
car_number_change_with_different_probability(filelist)

#
# def change_to_color(x):
#     """
#     Change the array to the colour to paint the graph
#     :param x: the array of road and cars
#     :return: the list of colour with same order of cars
#     """
#     color_show = []
#     for item in x:
#         if item == -1:
#             color_show.append('gray')  # road color is gray
#         elif item == 0:
#             color_show.append('lime')  # speed 0 color is lime
#         elif item == 1:
#             color_show.append('cyan')  # speed 1 color is cyan
#         elif item == 2:
#             color_show.append('yellow')  # speed 2 color is yellow
#         elif item == 3:
#             color_show.append('orange')  # speed 3 color is orange
#         elif item == 4:
#             color_show.append('coral')  # speed 4 color is coral
#         elif item == 5:
#             color_show.append('red')  # speed 5 color is red
#     return color_show
# x = (-1,1,-1,-1,2,-1,-1,2,0,0)
# color_show = change_to_color(x)
# pie_sep = []
# for i in range(10):
#     pie_sep.append(1/10)
# plt.pie(pie_sep, radius=1, colors=color_show,)  # main body of the road
# plt.pie([1, 0, 0], radius=0.95, colors='w')  # decoration part, white circle to make the road be like a road
# # label part
# plt.title('Step 5 in a busy traffic jam')
# plt.text(0.3, 0.4, 'Road', color='gray', fontsize=6)
# plt.text(0.3, 0.3, 'Speed:0', color='lime', fontsize=6)
# plt.text(0.3, 0.2, 'Speed:1', color='cyan', fontsize=6)
# plt.text(0.3, 0.1, 'Speed:2', color='yellow', fontsize=6)
# plt.text(0.3, 0, 'Speed:3', color='orange', fontsize=6)
# plt.text(0.3, -0.1, 'Speed:4', color='coral', fontsize=6)
# plt.text(0.3, -0.2, 'Speed:5', color='red', fontsize=6)
#
# ax = plt.gca()  # decoration part,
# ax.add_patch(Rectangle((0.26, -0.26), 0.35, 0.76, color='dimgray', fill=False))  # make the label obvious.
#
# plt.show()