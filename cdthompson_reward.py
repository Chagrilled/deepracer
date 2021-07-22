"""
AWS DeepRacer reward function
"""
import math
import time
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import LinearRing, LineString
from numpy import array

# Constants
DEBUG_LOG_ENABLED = True

# Action space constants
MAX_SPEED = 9.0
MAX_STEERING_ANGLE = 40.0

# Raceline track
RACE_LINE_WAYPOINTS = EMPIRE_TRACK_RACE_LINE 

# TUNING: Adjust these to find tune factors affect on reward
#
# Reward weights, always 0..1.  These are relative to one another
SPEED_FACTOR_WEIGHT = 0.0
SPEED_FACTOR_EASING = 'linear'
WHEEL_FACTOR_WEIGHT = 0.0
WHEEL_FACTOR_EASING = 'linear'
HEADING_FACTOR_WEIGHT = 0.0
HEADING_FACTOR_EASING = 'linear'
STEERING_FACTOR_WEIGHT = 0.0
STEERING_FACTOR_EASING = 'linear'
PROGRESS_FACTOR_WEIGHT = 0.0
PROGRESS_FACTOR_EASING = 'linear'
LANE_FACTOR_WEIGHT = 0.0
LANE_FACTOR_EASING = 'linear'
RACE_LINE_FACTOR_WEIGHT = 1.0
RACE_LINE_FACTOR_EASING = 'linear'

# Globals
g_last_progress_value = 0.0
g_last_progress_time = 0.0
g_last_speed_value = 0.0
g_last_steering_angle = 0.0

#===============================================================================
#
# REWARD
#
#===============================================================================

def reward_function(params):
  """Reward function is:

  f(s,w,h,t,p) = 1.0 * W(s,Ks) * W(w,Kw) * W(h,Kh) * W(t,Kt) * W(p,Kp) * W(l,Kl)

  s: speed factor, linear 0..1 for range of speed from 0 to MAX_SPEED
  w: wheel factor, non-linear 0..1 for wheels being off the track and
     vehicle in danger of going off the track.  We want to use the full
     width of the track for smoothing curves so we only apply wheel
     factor if the car is hanging off the track.
  h: heading factor, 0..1 for range of angle between car heading vector
     and the track direction vector.  This is the current heading
     based on the immediate direction of the car regardless of steering.
  t: steering factor, 0..1 for steering pressure if steering the wrong
     direction to correct the heading.
  p: progress factor
  l: lane factor

  W: Weighting function: (1.0 - (1.0 - f) * Kf)
  Kx: Weight of respective factor

      Example 1:
        s = 0
        Ks = 0.5
        reward = (1.0 - ((1.0 - s) * Ks)) = 1.0 - (1.0 - 0) * 0.5 = 0.5

      Example 2:
        s = 0.25
        Ks = 1.0
        reward = (1.0 - ((1.0 - s) * Ks)) = 1.0 - (1.0 - 0.25) * 1.0 = 0.25

      Example 2:
        s = 1.0
        Ks = 0.1
        reward = (1.0 - ((1.0 - s) * Ks)) = 1.0 - (1.0 - 1.0) * 1.0 = 1.0

  params:

  from https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-input.html

    Name                  Type                    Value(s)
    ----                  ----                    --------
    track_width           float                   0..Dtrack (varies)
    distance_from_center  float                   0..~track_width/2
    speed                 float                   0.0..5.0
    steering_angle        float                   -30..30
    all_wheels_on_track   Boolean                 True|False
    heading               float                   -180..+180
    waypoints             list of [float, float]  [[xw,0,yw,0] ... [xw,Max-1, yw,Max-1]]
    closest_waypoints     [int, int]              [0..Max-2, 1..Max-1]
    steps                 int                     0..Nstep
    progress              float                   0..100
    is_left_of_center     Boolean                 True|False
    is_reversed           Boolean                 True|False
    x                     float
    y                     float

  """

  # s: Speed Factor: ideal speed is max
  speed_factor = calculate_speed_factor(params)

  # w: Wheel Factor: apply pressure when wheels are off the track
  wheel_factor = calculate_wheel_factor(params)

  # h: Heading Factor
  heading_factor = calculate_heading_factor(params)

  # t: Steering Factor
  steering_factor = calculate_steering_factor(params)

  # p: Progress Factor: TBD
  progress_factor = calculate_progress_factor(params)

  # l: Lane Factor
  lane_factor = calculate_lane_factor(params)

  # r: Race line factor (distance from)
  race_line_factor = calculate_race_line_factor(params)

  # Log for validation
  if DEBUG_LOG_ENABLED:
    print("s: %0.2f, w: %0.2f, h: %0.2f, t: %0.2f, p: %0.2f, l: %0.2f r: %0.2f" %
          (speed_factor, wheel_factor, heading_factor, steering_factor,
           progress_factor, lane_factor, race_line_factor))

  reward = 1.0
  reward *= apply_weight(speed_factor, SPEED_FACTOR_WEIGHT, SPEED_FACTOR_EASING)
  reward *= apply_weight(wheel_factor, WHEEL_FACTOR_WEIGHT, WHEEL_FACTOR_EASING)
  reward *= apply_weight(heading_factor, HEADING_FACTOR_WEIGHT,
                         HEADING_FACTOR_EASING)
  reward *= apply_weight(steering_factor, STEERING_FACTOR_WEIGHT,
                         STEERING_FACTOR_EASING)
  reward *= apply_weight(progress_factor, PROGRESS_FACTOR_WEIGHT)
  reward *= apply_weight(lane_factor, LANE_FACTOR_WEIGHT, LANE_FACTOR_EASING)
  reward *= apply_weight(race_line_factor, RACE_LINE_FACTOR_WEIGHT, RACE_LINE_FACTOR_EASING)

  return float(max(reward, 1e-3)) # make sure we never return exactly zero



#===============================================================================
#
# RACE LINE
#
#===============================================================================
def calculate_race_line_factor(params):
    # Reward for track position
    current_position = Point(params['x'], params['y'])
    race_line = LineString(RACE_LINE_WAYPOINTS)
    distance = current_position.distance(race_line)
    # clamp reward to range (0..1) mapped to distance (track_width..0).
    # This could be negative since the car center can be off the track but
    # still not disqualified.

    factor = 1.0 - distance / params['track_width']
    print("x %0.2f y %0.2f distance %0.2f track_width %0.2f factor %0.7f" % (params['x'], params['y'], distance, params['track_width'], factor))
    return float(max(factor, 0.0))
  

#===============================================================================
#
# SPEED
#
#===============================================================================

def penalize_downshifting(speed):
  global g_last_speed_value
  if g_last_speed_value > speed:
    speed_factor = 1e-3
  else:
    speed_factor = 1.0
  g_last_speed_value = speed
  return speed_factor
  
def reward_upshifting(speed):
  global g_last_speed_value
  if g_last_speed_value < speed:
    speed_factor = 1.0
  else:
    speed_factor = 0.5
  g_last_speed_value = speed
  return speed_factor

def speed_or_acceleration(speed):
  """ Reward top speed AND any acceleration as well """
  global g_last_speed_value
  if speed > g_last_speed_value:
    speed_factor = 1.0
  else:
    speed_factor = percentage_speed(speed)
  return speed_factor

def percentage_speed(speed):
  return speed / MAX_SPEED

def calculate_speed_factor(params):
  """ Calculate the speed factor """

  # make calls here not affect each other
  speed_factor = percentage_speed(params['speed'])
  return min(speed_factor, 1.0)


#===============================================================================
#
# PROGRESS
#
#===============================================================================

def progress_over_time(progress):
  """ Calculate the progress per time.  Note that
  we rely on how the simulation code calculates
  progress which is an unknown algorithm.

  The nice thing about this algorithm is that is scales
  up rewards exponentially, as the differences in lower
  lap times are more valueable than at higher lap times.
  """
  global g_last_progress_value
  global g_last_progress_time
  current_time = time.time()
  # progress is 0..100
  if g_last_progress_value == 0:
    progress_factor = 1.0 # arbitrary but positive enough to promote going
  else:
    # time can be anything, but probably ~20s/lap, 15fps:
    #       1s/15frames = 67ms/frame = 0.067s
    #
    #   for 30s/lap: 30s*15f/s = 400 frames
    #         => expected progress of 100/400 = 0.25 per frame
    #         => 3.7
    #
    #   assuming 20s/lap: 20s*15f/s = 300 frames
    #         => expected progress of 100/300 = 0.3 progress per frame / 0.067s
    #         => 4.47
    #
    #   for 13s/lap: 13s*15f/s = 195 frames
    #         => expected progress of 100/195 = 0.51 per frame
    #         => 7.6
    #
    #   for 12s/lap: 12s*15f/s = 180 frames
    #         => expected progress of 100/180 = 0.55 per frame
    #         => 8.2
    #
    #   for 10s/lap: 10s*15f/s = 150 frames
    #         => expected progress of 100/150 = 0.67 per frame
    #         => 10
    #
    #   for 9s/lap: 9s*15f/s = 135 frames
    #         => expected progress of 100/135 = 0.74 per frame
    #         => 11.04
    #
    #   for 8s/lap: 8s*15f/s = 120 frames
    #         => expected progress of 100/120 = 0.83 per frame
    #         => 12.39
    #
    progress_factor = (progress - g_last_progress_value) / (current_time - g_last_progress_time)

  g_last_progress_value = progress
  g_last_progress_time = current_time
  return max(progress_factor, 0.0)  #make sure not going backwards

def progress_since_last(progress):
  global g_last_progress_value
  # progress is 0..100. The logic in DR environment code ensures this always
  # increases for the episode, regardless if the car is going backward.
  if g_last_progress_value > progress:
    g_last_progress_value = 0
  progress_factor = (progress - g_last_progress_value) / 100 # divide by 100 to get percentage of track
  g_last_progress_value = progress
  return progress_factor

def calculate_progress_factor(params):
  progress_factor = 1.0
  return min(progress_factor, 1.0)

#===============================================================================
#
# WHEELS
#
#===============================================================================


def all_wheels_must_be_on_track(all_wheels_on_track):
  """ Return low factor if car doesn't have all its wheels on the track """
  if not all_wheels_on_track:
    wheel_factor = 1e-3   # hard code multiplier rather than making it
                          # continuous since we don't know the width of
                          # the car wheelbase
  else:
    wheel_factor = 1.0
  return wheel_factor


def calculate_wheel_factor(params):
  """ Calculate the wheel factor """
  wheel_factor = all_wheels_must_be_on_track(params['all_wheels_on_track'])
  return min(wheel_factor, 1.0)


#===============================================================================
#
# HEADING
#
#===============================================================================

def look_ahead_heading(waypoints, current_waypoint, heading):
  """ Apply pressure based on upcoming track heading """
  
  track_headings = []
  v_init = current_waypoint
  for i in range(3):
    v1 = waypoints[(current_waypoint + 2*i) % len(waypoints)]
    v2 = waypoints[(current_waypoint + 2*i + 1) % len(waypoints)]
    track_heading = angle_of_vector([v1,v2])
    track_headings.append(track_heading)
  print(track_headings)
  return 1.0

def calculate_heading_factor(params):
  """ Calculate the heading factor """
  """
  # SUPRESS: This is too experimental while we haven't finished tracks yet
  closest_waypoints = params['closest_waypoints']
  waypoints = params['waypoints']
  heading = params['heading']

  # Calculate the immediate track angle
  wp1 = waypoints[closest_waypoints[0]]
  wp2 = waypoints[closest_waypoints[1]]
  ta1 = angle_of_vector([wp1,wp2])
  print("track angle 1: %i" % ta1)

  # h: Heading Factor: apply pressure as heading is different than track angle

  # Find closest angle, accounting for possibility of wrapping
  a = abs(ta1 - heading)
  b = abs(ta1 - (heading + 360))
  heading_delta = min(a,b)
  # hard fail if going backwards
  if heading_delta > 90:
    heading_factor = 1e-3
  elif heading_delta > 45:
    heading_factor = 0.5
  else:
    heading_factor = 1.0
  """
  heading_factor = 1.0
  heading_factor = look_ahead_heading(params['waypoints'],
                                      params['closest_waypoints'][0],
                                      params['heading'])
  return min(heading_factor, 1.0)


#===============================================================================
#
# STEERING
#
#===============================================================================

def penalize_steering_change(steering_angle, greater=True, less=True):
  ''' 
  Penalize steering changes 
    
  @greater: penalize sharper turning
  @less: penalize straightening
  '''
  global g_last_steering_angle
  if abs(steering_angle) > g_last_steering_angle and greater:
    # turning sharper
    steering_penalty = 1.0
  elif abs(steering_angle) < g_last_steering_angle and less:
    # straightening
    steering_penalty = 1.0
  else:
    steering_penalty = 0.0
  g_last_steering_angle = abs(steering_angle)
  return 1.0 - steering_penalty

def percentage_steering_angle(steering_angle):
  steering_severity = abs(steering_angle) / MAX_STEERING_ANGLE
  return max(min(1.0 - steering_severity, 1.0), 0.0)

def calculate_steering_factor(params):
  """ Calculate the steering factor """
  steering_factor = percentage_steering_angle(params['steering_angle'])
  return min(steering_factor, 1.0)


#===============================================================================
#
# LANE
#
#===============================================================================


def percentage_distance_from_track_center(track_width, distance_from_center):
  """ Return a linear percentage distance along the track width from
  the center to the outside
  """
  # make sure not negative, in case distance_from_center is over the track_width
  distance = distance_from_center / (track_width/2.0)
  return max(min(1.0 - distance, 1.0), 0.0)

def penalize_off_track(track_width, distance_from_center):
  if distance_from_center >= (track_width/2.0):
    penalty = 1.0
  else:
    penalty = 0.0
  return (1.0 - penalty)

def calculate_lane_factor(params):
  """ Calulcate the reward for the position on the track.
  Be careful to account for the wheel factor here, possibly merge
  the two later.
  """
  lane_factor = penalize_off_track(params['track_width'],
                                   params['distance_from_center'])
  return min(lane_factor, 1.0)


#===============================================================================
#
# HELPER METHODS
#
#===============================================================================

def apply_weight(factor, weight, easing='linear'):
  """Apply a weight to factor, clamping both arguments at 1.0

  Factor values will be 0..1. This function will cause the range of the
  factor values to be reduced according to:

    f = 1 - weight * (1 - factor)^easing

  In simple terms, a weight of 0.5 will cause the factor to only have weighted
  values of 0.5..1.0. If we further apply an easing, the decay from 1.0 toward
  the weighted minimum will be along a curve.
  """

  f_clamp = min(factor, 1.0)
  w_clamp = min(weight, 1.0)
  if EASING_FUNCTIONS[easing]:
    ease = EASING_FUNCTIONS[easing]
  else:
    ease = EASING_FUNCTIONS['linear']

  return 1.0 - w_clamp * ease(1.0 - f_clamp)


def vector_of_angle(angle):
  """ Unit vector of an angle in degrees. """
  return [[0.0, 0.0], [math.sin(math.radians(angle)), math.cos(math.radians(angle))]]


def angle_of_vector(vector):
  """ Calculate the angle of the vector in degrees relative to
  a normal 2d coordinate system.  This is useful for finding the
  angle between two waypoints.

    vector: [[x0,y0],[x1,y1]]

  """
  rad = math.atan2(vector[1][1] - vector[0][1], vector[1][0] - vector[0][0])
  return math.degrees(rad)

#
# SCALING FUNCTIONS
#

def ease_linear(x):
  return x

def ease_quadratic(x):
  return x*x

def ease_cubic(x):
  return abs(x*x*x)

def ease_quartic(x):
  return x*x*x*x

def ease_quintic(x):
  return abs(x*x*x*x*x)

def ease_septic(x):
  return abs(x*x*x*x*x*x*x)

def ease_nonic(x):
  return abs(x*x*x*x*x*x*x*x*x)

EASING_FUNCTIONS = {
    'linear': ease_linear,
    'quadratic': ease_quadratic,
    'cubic': ease_cubic,
    'quartic': ease_quartic,
    'quintic': ease_quintic,
    'septic': ease_septic,
    'nonic': ease_nonic
}


EMPIRE_TRACK_RACE_LINE = \
array([[ 4.3096576 ,  0.43175543],
       [ 4.43611328,  0.43785095],
       [ 4.56268127,  0.4430984 ],
       [ 4.68937377,  0.44740419],
       [ 4.81620111,  0.45068841],
       [ 4.94317132,  0.45288768],
       [ 5.07028955,  0.45396007],
       [ 5.19756089,  0.45386345],
       [ 5.3249889 ,  0.45256655],
       [ 5.45248219,  0.45004143],
       [ 5.57922872,  0.44997046],
       [ 5.70473551,  0.45340521],
       [ 5.82848844,  0.46125673],
       [ 5.94998153,  0.4742394 ],
       [ 6.06867127,  0.49299966],
       [ 6.1840361 ,  0.5180072 ],
       [ 6.29558642,  0.54957619],
       [ 6.40288468,  0.58786664],
       [ 6.505548  ,  0.63291458],
       [ 6.60312872,  0.68480974],
       [ 6.69513749,  0.74363862],
       [ 6.78100267,  0.80951546],
       [ 6.86002729,  0.88259531],
       [ 6.93134802,  0.9630628 ],
       [ 6.99382781,  1.05114055],
       [ 7.04800216,  1.14563434],
       [ 7.09470755,  1.24540977],
       [ 7.13434495,  1.34982873],
       [ 7.16731956,  1.45834806],
       [ 7.19407904,  1.57048021],
       [ 7.21512891,  1.68577936],
       [ 7.23102149,  1.80384644],
       [ 7.24240087,  1.92430699],
       [ 7.24998967,  2.04682236],
       [ 7.25459616,  2.17109691],
       [ 7.25708781,  2.29693672],
       [ 7.25837876,  2.42430013],
       [ 7.25943043,  2.55234755],
       [ 7.2605975 ,  2.68039397],
       [ 7.26184782,  2.80843966],
       [ 7.26261412,  2.93587442],
       [ 7.26163697,  3.06181009],
       [ 7.25778219,  3.18586319],
       [ 7.25001308,  3.3076566 ],
       [ 7.23743968,  3.42679008],
       [ 7.21932381,  3.54283508],
       [ 7.19513958,  3.65538007],
       [ 7.16442177,  3.76395933],
       [ 7.12677991,  3.86806602],
       [ 7.08187002,  3.96713366],
       [ 7.02938743,  4.06052266],
       [ 6.96898881,  4.14743379],
       [ 6.90032424,  4.2268709 ],
       [ 6.82309927,  4.2975968 ],
       [ 6.73858222,  4.35995703],
       [ 6.64775097,  4.41428421],
       [ 6.55140567,  4.46091809],
       [ 6.45019902,  4.50016662],
       [ 6.34474533,  4.53242882],
       [ 6.23557712,  4.55810935],
       [ 6.12320446,  4.57770788],
       [ 6.00809793,  4.59178724],
       [ 5.89076131,  4.60112383],
       [ 5.77170881,  4.60666449],
       [ 5.65146179,  4.60949238],
       [ 5.53055796,  4.61080671],
       [ 5.40757473,  4.61241498],
       [ 5.28463656,  4.61445773],
       [ 5.16176129,  4.61709032],
       [ 5.03896443,  4.62043252],
       [ 4.91626304,  4.62460189],
       [ 4.79367067,  4.62966767],
       [ 4.67120018,  4.63567395],
       [ 4.54886429,  4.64263531],
       [ 4.42667954,  4.6505417 ],
       [ 4.30467315,  4.65935599],
       [ 4.18289338,  4.6690128 ],
       [ 4.06142065,  4.67942527],
       [ 3.94035913,  4.69047878],
       [ 3.81981416,  4.70204861],
       [ 3.69985043,  4.7140087 ],
       [ 3.58175652,  4.72545028],
       [ 3.46346962,  4.73647602],
       [ 3.34493479,  4.74699273],
       [ 3.22610199,  4.75693048],
       [ 3.10691849,  4.76622691],
       [ 2.98733663,  4.77484073],
       [ 2.86730909,  4.78274094],
       [ 2.74678656,  4.78990292],
       [ 2.6257234 ,  4.79631586],
       [ 2.504074  ,  4.80197608],
       [ 2.38180198,  4.8068968 ],
       [ 2.25885772,  4.81108132],
       [ 2.13519562,  4.81454473],
       [ 2.01076506,  4.81730435],
       [ 1.88549101,  4.81936733],
       [ 1.7592844 ,  4.82074804],
       [ 1.63198446,  4.82144477],
       [ 1.50328499,  4.82145143],
       [ 1.37487785,  4.81983978],
       [ 1.24704853,  4.81568235],
       [ 1.12011735,  4.80807435],
       [ 0.99446806,  4.79614114],
       [ 0.87052743,  4.77910969],
       [ 0.74878841,  4.75623776],
       [ 0.62980464,  4.72684782],
       [ 0.51419327,  4.69033355],
       [ 0.40263991,  4.64616652],
       [ 0.29590352,  4.59390799],
       [ 0.19481756,  4.53323213],
       [ 0.10022233,  4.46403835],
       [ 0.01298869,  4.38641311],
       [-0.06592831,  4.30058605],
       [-0.13549093,  4.20697385],
       [-0.19460407,  4.10623738],
       [-0.24221381,  3.99935358],
       [-0.27732069,  3.88760452],
       [-0.29919141,  3.77258987],
       [-0.30758752,  3.65608899],
       [-0.30253104,  3.53984713],
       [-0.28440159,  3.42546742],
       [-0.25389318,  3.31429133],
       [-0.21190016,  3.20734007],
       [-0.15944394,  3.1052964 ],
       [-0.0975509 ,  3.00855624],
       [-0.02727009,  2.9172388 ],
       [ 0.05023855,  2.8311338 ],
       [ 0.13401145,  2.74995593],
       [ 0.22308528,  2.67325273],
       [ 0.31650483,  2.60043296],
       [ 0.41336286,  2.53082899],
       [ 0.51276253,  2.46368868],
       [ 0.61383038,  2.39821011],
       [ 0.71237719,  2.3330141 ],
       [ 0.80972434,  2.26652993],
       [ 0.90547601,  2.19834549],
       [ 0.99927148,  2.12809409],
       [ 1.09077249,  2.05544153],
       [ 1.17966781,  1.98009151],
       [ 1.26568066,  1.90179326],
       [ 1.34859593,  1.82036836],
       [ 1.42826978,  1.73571743],
       [ 1.50456645,  1.64775503],
       [ 1.57733088,  1.55638375],
       [ 1.64642357,  1.46152887],
       [ 1.71168343,  1.36310369],
       [ 1.77294801,  1.26102974],
       [ 1.82923903,  1.15654637],
       [ 1.88983885,  1.05677517],
       [ 1.95493106,  0.96217268],
       [ 2.02461243,  0.87313312],
       [ 2.0989461 ,  0.79004787],
       [ 2.17796603,  0.71331677],
       [ 2.26168869,  0.64337218],
       [ 2.35013912,  0.58073463],
       [ 2.44335506,  0.52605671],
       [ 2.54137635,  0.48016873],
       [ 2.64422274,  0.44416486],
       [ 2.75062191,  0.41629278],
       [ 2.86000348,  0.39584727],
       [ 2.97190058,  0.38208622],
       [ 3.0859431 ,  0.37423186],
       [ 3.20184544,  0.37147286],
       [ 3.31936177,  0.37293519],
       [ 3.43822909,  0.37768058],
       [ 3.55809618,  0.38467689],
       [ 3.67850171,  0.39281976],
       [ 3.80464567,  0.40126336],
       [ 3.93081273,  0.40953316],
       [ 4.05702454,  0.41746628],
       [ 4.18330097,  0.42491294],
       [ 4.3096576 ,  0.43175543]])