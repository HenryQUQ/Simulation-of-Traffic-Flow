# Import Section
import numpy as np
import random
import os
import re
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

plt.ion()  # Turn the interactive mode on


# Function Section
################# Running Part ###################
def create_empty_road():
    """
    Create an empty road
    :return: an empty road
    """
    x = np.zeros(L, int)
    for i in range(L):
        x[i] = -1
    return x


def initialize_road():
    """
    Initialize the road by using parameters
    :return: initial road
    """
    x = create_empty_road()

    car_position = random.sample(range(L), N)  # set the cars' position randomly

    for car in car_position:
        x[car] = 0
    return x


def accelerate_to_speed_limit(x):
    """
    First rule: every car will accelerate up to the speed limit
    :param x: the road and cars with initial speed
    :return: the road and cars with accelerated speed
    """
    x_new = []
    for cell in x:
        if cell != -1 and cell < V_max:  # if the speed of the car does not reach the maximum speed,
            cell += 1                    # the car will accelerates.
        x_new.append(cell)
    return x_new


def distance_detect(x, index_cell):
    """
    Detect the distance between current car and front car
    :param x: the road and cars
    :param index_cell: the index of current car
    :return: the distance between current car and front car
    """
    distance = 1
    while distance:
        front_index = index_cell + distance
        if front_index >= L:  # if the index is beyond the maximum length,
            front_index -= L  # it will return to the beginning because the road is a closed circle
        if x[front_index] != -1:  # if the front car is detected,
            break                 # jump out the loop
        distance += 1
    return distance


def avoid_hit(x):
    """
    Second rule: the drivers do not want to hit the car in front
    :param x: the road and cars with initial speed
    :return: the road and cars with speed which may be adjusted
    """
    x_new = []
    for index_cell, cell in enumerate(x):
        if cell != -1:
            distance = distance_detect(x, index_cell)  # get the distance between this car and front car
            if cell >= distance:     # if front car is close,
                cell = distance - 1  # the car will decelerate to avoid hitting.
        x_new.append(cell)
    return x_new


def unexpected_events(x):
    """
    Third rule: After the former two rules, unexpected events cause slowing down.
    :param x: the road and cars with initial speed
    :return: the road and cars with speed under unexpected events
    """
    x_new = []
    for cell in x:
        if cell > 0 and random.random() <= p:  # if the speed of car is not 0,
            cell -= 1                          # it is possible to slow down due to unexpected events.
        x_new.append(cell)
    return x_new


def move_forward(x):
    """
    In the end of one cycle, the car will move forward based on their speed.
    :param x: the road and cars at the initial positions
    :return x_new: the road and cars at the final positions;
    :return tf_count: the count for traffic flow when the cars pass the end point.
    """
    x_new = create_empty_road()
    tf_count = 0

    for index_cell, cell in enumerate(x):
        if cell != -1:
            new_index = index_cell + cell
            if new_index >= L:  # because the road is a closed circle again
                new_index -= L
                tf_count += 1
            x_new[new_index] = cell
    return x_new, tf_count


def one_cycle(x):
    """
    The whole process during one unit time
    :param x: the initial road and cars with speed
    :return x_new: the new road an cars with speed
    :return tf_count: the count for traffic flow when the cars pass the end point.
    """
    x_accelerate = accelerate_to_speed_limit(x)
    x_avoid = avoid_hit(x_accelerate)
    x_unexpected = unexpected_events(x_avoid)
    x_new, tf_count = move_forward(x_unexpected)
    return x_new, tf_count


def change_to_color(x):
    """
    Change the array to the colour to paint the graph
    :param x: the array of road and cars
    :return: the list of colour with same order of cars
    """
    color_show = []
    for item in x:
        if item == -1:
            color_show.append('gray')  # road color is gray
        elif item == 0:
            color_show.append('lime')    # speed 0 color is lime
        elif item == 1:
            color_show.append('cyan')    # speed 1 color is cyan
        elif item == 2:
            color_show.append('yellow')  # speed 2 color is yellow
        elif item == 3:
            color_show.append('orange')  # speed 3 color is orange
        elif item == 4:
            color_show.append('coral')   # speed 4 color is coral
        elif item == 5:
            color_show.append('red')     # speed 5 color is red
    return color_show


def plot_section(x, t):
    """
    The function to plot the road and cars with colours
    :param x: the array of the road and cars
    :return: True if all clear
    """
    color_show = change_to_color(x)  # change the array to colour parameter

    title = 'The Preview Graph for the Road when t = ' + str(t)
    plt.title(title)

    # label part
    plt.text(0.3, 0.4, 'Road', color='gray', fontsize=6)
    plt.text(0.3, 0.3, 'Speed:0', color='lime', fontsize=6)
    plt.text(0.3, 0.2, 'Speed:1', color='cyan', fontsize=6)
    plt.text(0.3, 0.1, 'Speed:2', color='yellow', fontsize=6)
    plt.text(0.3, 0, 'Speed:3', color='orange', fontsize=6)
    plt.text(0.3, -0.1, 'Speed:4', color='coral', fontsize=6)
    plt.text(0.3, -0.2, 'Speed:5', color='red', fontsize=6)

    # plot the road
    pie_sep = []
    for i in range(L):
        pie_sep.append(1 / L)
    plt.pie(pie_sep, radius=1, colors=color_show)  # main body of the road
    plt.pie([1, 0, 0], radius=0.95, colors='w')    # decoration part, white circle to make the road be like a road

    ax = plt.gca()                                                                   # decoration part,
    ax.add_patch(Rectangle((0.26, -0.26), 0.35, 0.76, color='dimgray', fill=False))  # make the label obvious.

    plt.pause(0.1)  # After showing a graph, the graph pauses for 0.1 second.

    return True


def record_traffic_flow(x):
    """
    Record the traffic flow for each cycle.
    :param x: road and cars with speed
    :return: True if successive
    """
    t = 0
    traffic_flow = []  # create an empty list to save the data of traffic flow

    while not t == 1000:  # run the simulation 1000 times to reduce the error
        t += 1
        x, tf_count = one_cycle(x)
        traffic_flow.append(tf_count)

    text_file_name = str(L) + '_' + str(N) + '_' + str(V_max) + '_' + str(p) + '_' + 'tf.txt'
    traffic_flow_text_file = open(text_file_name, 'w+')  # create the txt file with name
    traffic_flow_text_file.write(str(np.mean(traffic_flow)))  # write the average value on the first line

    for one_traffic_flow in traffic_flow:  # record the data for each run in case of further research
        traffic_flow_text_file.write(str(one_traffic_flow))
        traffic_flow_text_file.write('\n')

    traffic_flow_text_file.close()

    return True


################# Analysis Part ###################
def analysis():
    """
    Analysis main function
    :return: True if successive
    """
    plt.ioff()
    filelist = findall_file()  # find all file in current path

    # plot the graph of the density against traffic flow
    # with different probability when length of road is changing.
    length_change_with_different_probability(filelist)
    # with different speed limit when length of road is changing.
    length_change_with_different_speed(filelist)
    # with different probability when number of cars is changing.
    car_number_change_with_different_probability(filelist)
    # with different speed limit when number of cars is changing.
    car_number_change_with_different_speed(filelist)
    return True


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


def plot_fitting_line_length(density, tf, range, label):
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
        if range[0] < density[index] < range[1]:  # we need the density only within the range to plot a best fit line
            density_in_range.append(density[index])
            tf_in_range.append(tf[index])  # we need to record the corresponding traffic flow data

    # get the data and error for the best fit line
    fitting_data, fitting_error = np.polyfit(density_in_range, tf_in_range, 1, cov=True)
    fitting_y = []
    for one_desity in density_in_range:
        fitting_y.append(one_desity * fitting_data[0] + fitting_data[1])
    plt.plot(density, fitting_y, label=label)

    # create the text for showing the slope and intercept with their error
    slope_text = ('Slope = ' + round_to_3_sf(fitting_data[0]) + ' +/- ' + round_to_3_sf(
        abs(min(fitting_error[0][0], fitting_error[0][1]))))
    intercept_text = ('Intercept = ' + round_to_3_sf(fitting_data[1]) + ' +/- ' + round_to_3_sf(
        abs(min(fitting_error[1][0], fitting_error[1][1]))))

    return slope_text, intercept_text


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


def plot_density_tf_graph_speed(density_5, density_10, density_15, density_20, tf_5, tf_10, tf_15, tf_20):
    """
    Plot the traffic flow against density graph with different speed limit.
    :param density_5: density at speed limit of 5
    :param density_10: density at speed limit of 10
    :param density_15: density at speed limit of 15
    :param density_20: density at speed limit of 20
    :param tf_5: traffic flow at speed limit of 5
    :param tf_10: traffic flow at speed limit of 10
    :param tf_15: traffic flow at speed limit of 15
    :param tf_20: traffic flow at speed limit of 20
    :return: True if successive
    """
    plt.plot(density_5, tf_5, '.', label='traffic flow at v_max = 5')
    plt.plot(density_10, tf_10, '.', label='traffic flow at v_max = 10')
    plt.plot(density_15, tf_15, '.', label='traffic flow at v_max = 15')
    plt.plot(density_20, tf_20, '.', label='traffic flow at v_max = 20')
    plt.xlabel('density')
    plt.ylabel('traffic flow')

    return True


def length_change_with_different_probability(filelist):
    """
    Main function to plot the traffic flow against density graph with different probability
    while the length of road is changing.
    :param filelist: list of all files
    :return: True if successive
    """
    # Find all file names with matched
    pattern_025 = re.compile(r'\d+_20_5_0.25_tf.txt')
    file_needed_025 = pattern_fit_files(filelist, pattern_025)
    pattern_050 = re.compile(r'\d+_20_5_0.5_tf.txt')
    file_needed_050 = pattern_fit_files(filelist, pattern_050)
    pattern_075 = re.compile(r'\d+_20_5_0.75_tf.txt')
    file_needed_075 = pattern_fit_files(filelist, pattern_075)

    # get the density and traffic flow for each of them
    density_025, tf_025 = get_density_and_tf_length(file_needed_025, '_20_5_0.25_tf.txt')
    density_050, tf_050 = get_density_and_tf_length(file_needed_050, '_20_5_0.5_tf.txt')
    density_075, tf_075 = get_density_and_tf_length(file_needed_075, '_20_5_0.75_tf.txt')

    # Plot the graph
    title = 'Traffic Flow Against Density with Different Probabilities While Length Changes'
    plt.title(title)
    plot_density_tf_graph_probability(density_025, density_050, density_075, tf_025, tf_050, tf_075)
    plt.legend()
    plt.show()

    return True


def length_change_with_different_speed(filelist):
    """
    Main function to plot the traffic flow against density graph with different speed limit
    while the length of road is changing.
    :param filelist: list of all files
    :return: True if successive
    """
    # Find all file names with matched
    pattern_5 = re.compile(r'\d+_20_5_0.25_tf.txt')
    file_needed_5 = pattern_fit_files(filelist, pattern_5)
    pattern_10 = re.compile(r'\d+_20_10_0.25_tf.txt')
    file_needed_10 = pattern_fit_files(filelist, pattern_10)
    pattern_15 = re.compile(r'\d+_20_15_0.25_tf.txt')
    file_needed_15 = pattern_fit_files(filelist, pattern_15)
    pattern_20 = re.compile(r'\d+_20_20_0.25_tf.txt')
    file_needed_20 = pattern_fit_files(filelist, pattern_20)

    # get the density and traffic flow for each of them
    density_5, tf_5 = get_density_and_tf_length(file_needed_5, '_20_5_0.25_tf.txt')
    density_10, tf_10 = get_density_and_tf_length(file_needed_10, '_20_10_0.25_tf.txt')
    density_15, tf_15 = get_density_and_tf_length(file_needed_15, '_20_15_0.25_tf.txt')
    density_20, tf_20 = get_density_and_tf_length(file_needed_20, '_20_20_0.25_tf.txt')

    # Plot the graph
    title = 'Traffic Flow Against Density with Different Speed Limit While Length Changes'
    plt.title(title)
    plot_density_tf_graph_speed(density_5, density_10, density_15, density_20, tf_5, tf_10, tf_15, tf_20)
    plt.legend()
    plt.show()

    return True


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
    title = 'Traffic Flow Against Density With Different Probabilities While Car Number Changes'
    plt.title(title)
    plot_density_tf_graph_probability(density_025, density_050, density_075, tf_025, tf_050, tf_075)
    plt.legend()
    plt.show()

    return True


def car_number_change_with_different_speed(filelist):
    """
    Main function to plot the traffic flow against density graph with different speed limit
    while number of cars is changing.
    :param filelist: list of all files
    :return: True if successive
    """
    # Find all file names with matched
    pattern_5 = re.compile(r'500_\d+_5_0.25_tf.txt')
    file_needed_5 = pattern_fit_files(filelist, pattern_5)
    pattern_10 = re.compile(r'500_\d+_10_0.25_tf.txt')
    file_needed_10 = pattern_fit_files(filelist, pattern_10)
    pattern_15 = re.compile(r'500_\d+_15_0.25_tf.txt')
    file_needed_15 = pattern_fit_files(filelist, pattern_15)
    pattern_20 = re.compile(r'500_\d+_20_0.25_tf.txt')
    file_needed_20 = pattern_fit_files(filelist, pattern_20)

    # get the density and traffic flow for each of them
    density_5, tf_5 = get_density_and_tf_car_number(file_needed_5, '_5_0.25_tf.txt')
    density_10, tf_10 = get_density_and_tf_car_number(file_needed_10, '_10_0.25_tf.txt')
    density_15, tf_15 = get_density_and_tf_car_number(file_needed_15, '_15_0.25_tf.txt')
    density_20, tf_20 = get_density_and_tf_car_number(file_needed_20, '_20_0.25_tf.txt')

    # Plot the graph
    title = 'Traffic Flow Against Density with Different Speed Limit While Car Number Changes'
    plt.title(title)
    plot_density_tf_graph_speed(density_5, density_10, density_15, density_20, tf_5, tf_10, tf_15, tf_20)
    plt.legend()
    plt.show()

    return True


# Setting Section
"""""""""""""""""""""
PLEASE READ

L, N, V_max, p and t_pre_run_max are adjustable parameters for simulation, which are also preset.

If you want to see the process of simulating the model from beginning,
text_para_pre_run or/and graphical_para_pre_run need be set to True.
graphical_para_pre_run is recommended for accessing the process of model visually.

If you want to see the process when the model is in the stable state,
graphical_para_final need be set to True.

If you want to save traffic flow data for further research,
traffic_flow_save_para need to be set to True.
Maybe you need to run the simulation several time with adjusting the simulation parameters to access different data.

If you want to analysis data,
analysis_para need to be set to True.
only_analysis_para is provided to skip the simulation process.
"""""""""""""""""""""

# simulation parameters
global L, N, V_max, p
L = 200                # length of road
N = 30                 # number of cars in the road
V_max = 5              # maximum speed of cars
p = 0.25               # probability of unexpected events
t_pre_run_max = 10000  # the maximum time of pre-run

# text and graphical output
global text_para_pre_run, graphical_para_pre_run, graphical_para_final
text_para_pre_run = False      # whether show the array of road when it is pre-run, show is True.
graphical_para_pre_run = True  # whether show the graph of road when it is pre-run , show is True.
graphical_para_final = False   # whether show the graph of road when the system is in the final state, show is True.

# Data save
traffic_flow_save_para = False  # whether record the data of traffic flow

# Analysis parameter
analysis_para = False       # if need analysis, show is True
only_analysis_para = False  # if only need analysis and not run the simulation, show is True
if only_analysis_para is True:
    analysis_para = True


# Main Body
if __name__ == '__main__':
    if only_analysis_para is False:  # See whether need to run the simulation
        t = 0   # time is 0 in the beginning
        x = initialize_road()  # create an initial road
        if graphical_para_pre_run is True:  # whether show the graph
            plot_section(x, t)
        if text_para_pre_run is True:  # whether show the array
            print(x)

        while not t == t_pre_run_max:  # stop when the time reaches the maximum time for pre-run
            t += 1  # time increases one each cycle
            x, _ = one_cycle(x)  # run the logic

            if graphical_para_pre_run is True:  # whether show the graph
                plot_section(x, t)
            if text_para_pre_run is True:  # whether show the array
                print(x)

        if graphical_para_final is True:
            for time in range(100):
                t+=1
                x, _ = one_cycle(x)
                plot_section(x, t)
    if traffic_flow_save_para is True:  # see whether need to save the traffic flow
        # save the data is better, because it does not require us run the simulation again and again.
        record_traffic_flow(x)

    if analysis_para is True:  # see whether need to run analysis part
        analysis()  # use the data we saved to do analysis

    exit()
