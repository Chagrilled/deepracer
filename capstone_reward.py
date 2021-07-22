import math


class Reward:
    def __init__(self, verbose=False):
        self.first_racingpoint_index = None
        self.verbose = verbose

    def reward_function(self, params):

        ################## HELPER FUNCTIONS ###################

        def dist_2_points(x1, x2, y1, y2):
            return abs(abs(x1-x2)**2 + abs(y1-y2)**2)**0.5

        def closest_2_racing_points_index(racing_coords, car_coords):

            # Calculate all distances to racing points
            distances = []
            for i in range(len(racing_coords)):
                distance = dist_2_points(x1=racing_coords[i][0], x2=car_coords[0],
                                         y1=racing_coords[i][1], y2=car_coords[1])
                distances.append(distance)

            # Get index of the closest racing point
            closest_index = distances.index(min(distances))

            # Get index of the second closest racing point
            distances_no_closest = distances.copy()
            distances_no_closest[closest_index] = 999
            second_closest_index = distances_no_closest.index(
                min(distances_no_closest))

            return [closest_index, second_closest_index]

        def dist_to_racing_line(closest_coords, second_closest_coords, car_coords):
            
            # Calculate the distances between 2 closest racing points
            a = abs(dist_2_points(x1=closest_coords[0],
                                  x2=second_closest_coords[0],
                                  y1=closest_coords[1],
                                  y2=second_closest_coords[1]))

            # Distances between car and closest and second closest racing point
            b = abs(dist_2_points(x1=car_coords[0],
                                  x2=closest_coords[0],
                                  y1=car_coords[1],
                                  y2=closest_coords[1]))
            c = abs(dist_2_points(x1=car_coords[0],
                                  x2=second_closest_coords[0],
                                  y1=car_coords[1],
                                  y2=second_closest_coords[1]))

            # Calculate distance between car and racing line (goes through 2 closest racing points)
            # try-except in case a=0 (rare bug in DeepRacer)
            try:
                distance = abs(-(a**4) + 2*(a**2)*(b**2) + 2*(a**2)*(c**2) -
                               (b**4) + 2*(b**2)*(c**2) - (c**4))**0.5 / (2*a)
            except:
                distance = b

            return distance

        # Calculate which one of the closest racing points is the next one and which one the previous one
        def next_prev_racing_point(closest_coords, second_closest_coords, car_coords, heading):

            # Virtually set the car more into the heading direction
            heading_vector = [math.cos(math.radians(
                heading)), math.sin(math.radians(heading))]
            new_car_coords = [car_coords[0]+heading_vector[0],
                              car_coords[1]+heading_vector[1]]

            # Calculate distance from new car coords to 2 closest racing points
            distance_closest_coords_new = dist_2_points(x1=new_car_coords[0],
                                                        x2=closest_coords[0],
                                                        y1=new_car_coords[1],
                                                        y2=closest_coords[1])
            distance_second_closest_coords_new = dist_2_points(x1=new_car_coords[0],
                                                               x2=second_closest_coords[0],
                                                               y1=new_car_coords[1],
                                                               y2=second_closest_coords[1])

            if distance_closest_coords_new <= distance_second_closest_coords_new:
                next_point_coords = closest_coords
                prev_point_coords = second_closest_coords
            else:
                next_point_coords = second_closest_coords
                prev_point_coords = closest_coords

            return [next_point_coords, prev_point_coords]

        def racing_direction_diff(closest_coords, second_closest_coords, car_coords, heading):

            # Calculate the direction of the center line based on the closest waypoints
            next_point, prev_point = next_prev_racing_point(closest_coords,
                                                            second_closest_coords,
                                                            car_coords,
                                                            heading)

            # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
            track_direction = math.atan2(
                next_point[1] - prev_point[1], next_point[0] - prev_point[0])

            # Convert to degree
            track_direction = math.degrees(track_direction)

            # Calculate the difference between the track direction and the heading direction of the car
            direction_diff = abs(track_direction - heading)
            if direction_diff > 180:
                direction_diff = 360 - direction_diff

            return direction_diff

        # Gives back indexes that lie between start and end index of a cyclical list 
        # (start index is included, end index is not)
        def indexes_cyclical(start, end, array_len):

            if end < start:
                end += array_len

            return [index % array_len for index in range(start, end)]

        # Calculate how long car would take for entire lap, if it continued like it did until now
        def projected_time(first_index, closest_index, step_count, times_list):

            # Calculate how much time has passed since start
            current_actual_time = (step_count-1) / 15

            # Calculate which indexes were already passed
            indexes_traveled = indexes_cyclical(first_index, closest_index, len(times_list))

            # Calculate how much time should have passed if car would have followed optimals
            current_expected_time = sum([times_list[i] for i in indexes_traveled])

            # Calculate how long one entire lap takes if car follows optimals
            total_expected_time = sum(times_list)

            # Calculate how long car would take for entire lap, if it continued like it did until now
            try:
                projected_time = (current_actual_time/current_expected_time) * total_expected_time
            except:
                projected_time = 9999

            return projected_time

        #################### RACING LINE ######################

        # Optimal racing line for the Oval track
        # Each row: [x,y,speed,timeFromPreviousPoint]
        racing_track = [[4.30966, 0.43176, 4.0, 0.03164],
                [4.43611, 0.43785, 4.0, 0.03165],
                [4.56268, 0.4431, 4.0, 0.03167],
                [4.68937, 0.4474, 4.0, 0.03169],
                [4.8162, 0.45069, 3.52117, 0.03603],
                [4.94317, 0.45289, 2.91198, 0.04361],
                [5.07029, 0.45396, 2.5505, 0.04984],
                [5.19756, 0.45386, 2.31535, 0.05497],
                [5.32499, 0.45257, 2.12619, 0.05994],
                [5.45248, 0.45004, 1.98558, 0.06422],
                [5.57923, 0.44997, 1.87704, 0.06752],
                [5.70474, 0.45341, 1.79461, 0.06996],
                [5.82849, 0.46126, 1.73197, 0.0716],
                [5.94998, 0.47424, 1.66603, 0.07334],
                [6.06867, 0.493, 1.60462, 0.07489],
                [6.18404, 0.51801, 1.54454, 0.07643],
                [6.29559, 0.54958, 1.4848, 0.07808],
                [6.40288, 0.58787, 1.42655, 0.07986],
                [6.50555, 0.63291, 1.3673, 0.082],
                [6.60313, 0.68481, 1.3673, 0.08083],
                [6.69514, 0.74364, 1.3673, 0.07987],
                [6.781, 0.80952, 1.3673, 0.07915],
                [6.86003, 0.8826, 1.3673, 0.07872],
                [6.93135, 0.96306, 1.3673, 0.07864],
                [6.99383, 1.05114, 1.45316, 0.07431],
                [7.048, 1.14563, 1.57635, 0.0691],
                [7.09471, 1.24541, 1.66612, 0.06612],
                [7.13434, 1.34983, 1.76519, 0.06327],
                [7.16732, 1.45835, 1.8799, 0.06033],
                [7.19408, 1.57048, 2.016, 0.05718],
                [7.21513, 1.68578, 2.17747, 0.05383],
                [7.23102, 1.80385, 2.38497, 0.04995],
                [7.2424, 1.92431, 2.66012, 0.04549],
                [7.24999, 2.04682, 3.05701, 0.04015],
                [7.2546, 2.1711, 3.68912, 0.03371],
                [7.25709, 2.29694, 4.0, 0.03147],
                [7.25838, 2.4243, 3.17307, 0.04014],
                [7.25943, 2.55235, 2.66026, 0.04813],
                [7.2606, 2.68039, 2.33994, 0.05472],
                [7.26185, 2.80844, 2.11633, 0.06051],
                [7.26261, 2.93587, 1.96014, 0.06501],
                [7.26164, 3.06181, 1.82781, 0.0689],
                [7.25778, 3.18586, 1.71608, 0.07232],
                [7.25001, 3.30766, 1.61816, 0.07542],
                [7.23744, 3.42679, 1.5317, 0.07821],
                [7.21932, 3.54284, 1.44808, 0.08111],
                [7.19514, 3.65538, 1.36996, 0.08403],
                [7.16442, 3.76396, 1.3, 0.0868],
                [7.12678, 3.86807, 1.3, 0.08516],
                [7.08187, 3.96713, 1.3, 0.08367],
                [7.02939, 4.06052, 1.3, 0.0824],
                [6.96899, 4.14743, 1.3, 0.08141],
                [6.90032, 4.22687, 1.3, 0.08077],
                [6.8231, 4.2976, 1.36408, 0.07677],
                [6.73858, 4.35996, 1.43115, 0.07339],
                [6.64775, 4.41428, 1.50481, 0.07033],
                [6.55141, 4.46092, 1.58213, 0.06765],
                [6.4502, 4.50017, 1.6767, 0.06474],
                [6.34475, 4.53243, 1.78038, 0.06194],
                [6.23558, 4.55811, 1.90716, 0.0588],
                [6.1232, 4.57771, 2.05822, 0.05542],
                [6.0081, 4.59179, 2.27662, 0.05094],
                [5.89076, 4.60112, 2.59945, 0.04528],
                [5.77171, 4.60666, 3.12612, 0.03812],
                [5.65146, 4.60949, 4.0, 0.03007],
                [5.53056, 4.61081, 4.0, 0.03023],
                [5.40757, 4.61241, 4.0, 0.03075],
                [5.28464, 4.61446, 4.0, 0.03074],
                [5.16176, 4.61709, 4.0, 0.03073],
                [5.03896, 4.62043, 4.0, 0.03071],
                [4.91626, 4.6246, 4.0, 0.03069],
                [4.79367, 4.62967, 4.0, 0.03067],
                [4.6712, 4.63567, 4.0, 0.03065],
                [4.54886, 4.64264, 4.0, 0.03063],
                [4.42668, 4.65054, 4.0, 0.03061],
                [4.30467, 4.65936, 4.0, 0.03058],
                [4.18289, 4.66901, 4.0, 0.03054],
                [4.06142, 4.67943, 4.0, 0.03048],
                [3.94036, 4.69048, 4.0, 0.03039],
                [3.81981, 4.70205, 4.0, 0.03027],
                [3.69985, 4.71401, 4.0, 0.03014],
                [3.58176, 4.72545, 4.0, 0.02966],
                [3.46347, 4.73648, 4.0, 0.0297],
                [3.34493, 4.74699, 4.0, 0.02975],
                [3.2261, 4.75693, 4.0, 0.02981],
                [3.10692, 4.76623, 4.0, 0.02989],
                [2.98734, 4.77484, 4.0, 0.02997],
                [2.86731, 4.78274, 4.0, 0.03007],
                [2.74679, 4.7899, 4.0, 0.03018],
                [2.62572, 4.79632, 4.0, 0.03031],
                [2.50407, 4.80198, 4.0, 0.03045],
                [2.3818, 4.8069, 4.0, 0.03059],
                [2.25886, 4.81108, 4.0, 0.03075],
                [2.1352, 4.81454, 4.0, 0.03093],
                [2.01077, 4.8173, 3.47117, 0.03586],
                [1.88549, 4.81937, 2.95795, 0.04236],
                [1.75928, 4.82075, 2.61304, 0.0483],
                [1.63198, 4.82144, 2.37326, 0.05364],
                [1.50328, 4.82145, 2.18029, 0.05903],
                [1.37488, 4.81984, 2.02408, 0.06344],
                [1.24705, 4.81568, 1.89434, 0.06752],
                [1.12012, 4.80807, 1.78481, 0.07125],
                [0.99447, 4.79614, 1.69191, 0.0746],
                [0.87053, 4.77911, 1.61402, 0.07751],
                [0.74879, 4.75624, 1.55776, 0.07952],
                [0.6298, 4.72685, 1.51536, 0.08088],
                [0.51419, 4.69033, 1.47909, 0.08197],
                [0.40264, 4.64617, 1.44807, 0.08285],
                [0.2959, 4.59391, 1.42182, 0.08358],
                [0.19482, 4.53323, 1.40238, 0.08407],
                [0.10022, 4.46404, 1.38465, 0.08464],
                [0.01299, 4.38641, 1.37397, 0.08499],
                [-0.06593, 4.30059, 1.37397, 0.08486],
                [-0.13549, 4.20697, 1.37397, 0.08488],
                [-0.1946, 4.10624, 1.37397, 0.08501],
                [-0.24221, 3.99935, 1.37397, 0.08516],
                [-0.27732, 3.8876, 1.37397, 0.08525],
                [-0.29919, 3.77259, 1.37604, 0.08508],
                [-0.30759, 3.65609, 1.37718, 0.08481],
                [-0.30253, 3.53985, 1.38442, 0.08404],
                [-0.2844, 3.42547, 1.40041, 0.0827],
                [-0.25389, 3.31429, 1.42566, 0.08087],
                [-0.2119, 3.20734, 1.46321, 0.07853],
                [-0.15944, 3.1053, 1.51148, 0.07591],
                [-0.09755, 3.00856, 1.57747, 0.0728],
                [-0.02727, 2.91724, 1.67877, 0.06864],
                [0.05024, 2.83113, 1.78843, 0.06478],
                [0.13401, 2.74996, 1.93415, 0.06031],
                [0.22309, 2.67325, 2.13024, 0.05518],
                [0.3165, 2.60043, 2.39172, 0.04952],
                [0.41336, 2.53083, 2.77991, 0.04291],
                [0.51276, 2.46369, 2.78931, 0.043],
                [0.61383, 2.39821, 2.61447, 0.04606],
                [0.71238, 2.33301, 2.48558, 0.04754],
                [0.80972, 2.26653, 2.40108, 0.0491],
                [0.90548, 2.19835, 2.35457, 0.04992],
                [0.99927, 2.12809, 2.31847, 0.05055],
                [1.09077, 2.05544, 2.28342, 0.05117],
                [1.17967, 1.98009, 2.26105, 0.05154],
                [1.26568, 1.90179, 2.23965, 0.05193],
                [1.3486, 1.82037, 2.22572, 0.05221],
                [1.42827, 1.73572, 2.19283, 0.05301],
                [1.50457, 1.64776, 2.06788, 0.05631],
                [1.57733, 1.55638, 1.95545, 0.05973],
                [1.64642, 1.46153, 1.8669, 0.06286],
                [1.71168, 1.3631, 1.78926, 0.066],
                [1.77295, 1.26103, 1.72097, 0.06918],
                [1.82924, 1.15655, 1.65939, 0.07152],
                [1.88984, 1.05678, 1.59945, 0.07298],
                [1.95493, 0.96217, 1.53973, 0.07458],
                [2.02461, 0.87313, 1.48036, 0.07638],
                [2.09895, 0.79005, 1.42014, 0.0785],
                [2.17797, 0.71332, 1.42014, 0.07756],
                [2.26169, 0.64337, 1.42014, 0.07682],
                [2.35014, 0.58073, 1.42014, 0.07632],
                [2.44336, 0.52606, 1.42014, 0.0761],
                [2.54138, 0.48017, 1.42014, 0.07621],
                [2.64422, 0.44416, 1.59739, 0.06822],
                [2.75062, 0.41629, 1.70527, 0.0645],
                [2.86, 0.39585, 1.83522, 0.06063],
                [2.9719, 0.38209, 1.9938, 0.05655],
                [3.08594, 0.37423, 2.19213, 0.05215],
                [3.20185, 0.37147, 2.4586, 0.04715],
                [3.31936, 0.37294, 2.8429, 0.04134],
                [3.43823, 0.37768, 3.49139, 0.03407],
                [3.5581, 0.38468, 4.0, 0.03002],
                [3.6785, 0.39282, 4.0, 0.03017],
                [3.80465, 0.40126, 4.0, 0.03161],
                [3.93081, 0.40953, 4.0, 0.03161],
                [4.05702, 0.41747, 4.0, 0.03162],
                [4.1833, 0.42491, 4.0, 0.03162]]

        ################## INPUT PARAMETERS ###################

        # Read all input parameters
        all_wheels_on_track = params['all_wheels_on_track']
        x = params['x']
        y = params['y']
        distance_from_center = params['distance_from_center']
        is_left_of_center = params['is_left_of_center']
        heading = params['heading']
        progress = params['progress']
        steps = params['steps']
        speed = params['speed']
        steering_angle = params['steering_angle']
        track_width = params['track_width']
        waypoints = params['waypoints']
        closest_waypoints = params['closest_waypoints']
        is_offtrack = params['is_offtrack']

        ############### OPTIMAL X,Y,SPEED,TIME ################

        # Get closest indexes for racing line (and distances to all points on racing line)
        closest_index, second_closest_index = closest_2_racing_points_index(
            racing_track, [x, y])

        # Get optimal [x, y, speed, time] for closest and second closest index
        optimals = racing_track[closest_index]
        optimals_second = racing_track[second_closest_index]

        # Save first racingpoint of episode for later
        if self.verbose == True:
            self.first_racingpoint_index = 0 # this is just for testing purposes
        if steps == 1:
            self.first_racingpoint_index = closest_index

        ################ REWARD AND PUNISHMENT ################

        ## Define the default reward ##
        reward = 1

        ## Reward if car goes close to optimal racing line ##
        DISTANCE_MULTIPLE = 1
        dist = dist_to_racing_line(optimals[0:2], optimals_second[0:2], [x, y])
        distance_reward = max(1e-3, 1 - (dist/(track_width*0.5)))
        reward += distance_reward * DISTANCE_MULTIPLE

        ## Reward if speed is close to optimal speed ##
        SPEED_DIFF_NO_REWARD = 1
        SPEED_MULTIPLE = 2
        speed_diff = abs(optimals[2]-speed)
        if speed_diff <= SPEED_DIFF_NO_REWARD:
            # we use quadratic punishment (not linear) bc we're not as confident with the optimal speed
            # so, we do not punish small deviations from optimal speed
            speed_reward = (1 - (speed_diff/(SPEED_DIFF_NO_REWARD))**2)**2
        else:
            speed_reward = 0
        reward += speed_reward * SPEED_MULTIPLE

        # Reward if less steps
        REWARD_PER_STEP_FOR_FASTEST_TIME = 1 
        STANDARD_TIME = 13
        FASTEST_TIME = 11
        times_list = [row[3] for row in racing_track]
        projected_time = projected_time(self.first_racingpoint_index, closest_index, steps, times_list)
        try:
            steps_prediction = projected_time * 15 + 1
            reward_prediction = max(1e-3, (-REWARD_PER_STEP_FOR_FASTEST_TIME*(FASTEST_TIME) /
                                           (STANDARD_TIME-FASTEST_TIME))*(steps_prediction-(STANDARD_TIME*15+1)))
            steps_reward = min(REWARD_PER_STEP_FOR_FASTEST_TIME, reward_prediction / steps_prediction)
        except:
            steps_reward = 0
        reward += steps_reward

        # Zero reward if obviously wrong direction (e.g. spin)
        direction_diff = racing_direction_diff(
            optimals[0:2], optimals_second[0:2], [x, y], heading)
        if direction_diff > 30:
            reward = 1e-3
            
        # Zero reward of obviously too slow
        speed_diff_zero = optimals[2]-speed
        if speed_diff_zero > 0.5:
            reward = 1e-3
            
        ## Incentive for finishing the lap in less steps ##
        REWARD_FOR_FASTEST_TIME = 1500 # should be adapted to track length and other rewards
        STANDARD_TIME = 13  # seconds (time that is easily done by model)
        FASTEST_TIME = 10  # seconds (best time of 1st place on the track)
        if progress == 100:
            finish_reward = max(1e-3, (-REWARD_FOR_FASTEST_TIME /
                      (15*(STANDARD_TIME-FASTEST_TIME)))*(steps-STANDARD_TIME*15))
        else:
            finish_reward = 0
        reward += finish_reward
        
        ## Zero reward if off track ##
        if all_wheels_on_track == False:
            reward = 1e-3

        ####################### VERBOSE #######################
        
        if self.verbose == True:
            print("Closest index: %i" % closest_index)
            print("Distance to racing line: %f" % dist)
            print("=== Distance reward (w/out multiple): %f ===" % (distance_reward))
            print("Optimal speed: %f" % optimals[2])
            print("Speed difference: %f" % speed_diff)
            print("=== Speed reward (w/out multiple): %f ===" % speed_reward)
            print("Direction difference: %f" % direction_diff)
            print("Predicted time: %f" % projected_time)
            print("=== Steps reward: %f ===" % steps_reward)
            print("=== Finish reward: %f ===" % finish_reward)
            
        #################### RETURN REWARD ####################
        
        # Always return a float value
        return float(reward)


reward_object = Reward(verbose=True) # add parameter verbose=True to get noisy output for testing


def reward_function(params):
    return reward_object.reward_function(params)