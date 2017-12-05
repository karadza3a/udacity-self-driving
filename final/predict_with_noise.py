# Now we'll make the scenario a bit more realistic. Now Traxbot's
# sensor measurements are a bit noisy (though its motions are still
# completetly noise-free and it still moves in an almost-circle).
# You'll have to write a function that takes as input the next
# noisy (x, y) sensor measurement and outputs the best guess
# for the robot's next position.
#
# ----------
# YOUR JOB
#
# Complete the function estimate_next_pos. You will be considered
# correct if your estimate is within 0.01 stepsizes of Traxbot's next
# true position.
#
# ----------
# GRADING
#
# We will make repeated calls to your estimate_next_pos function. After
# each call, we will compare your estimated position to the robot's true
# position. As soon as you are within 0.01 stepsizes of the true position,
# you will be marked correct and we will tell you how many steps it took
# before your function successfully located the target bot.

# These import steps give you access to libraries which you may (or may
# not) want to use.
from final.robot import *
from final.matrix import *


# noinspection PyPep8Naming
class EKF:
    def __init__(self, initial_x, initial_y):
        self.x_mat = matrix([[initial_x], [initial_y], [0.], [0.], [1.]])  # initial state (location and velocity)
        self.u = matrix([[0.], [0.], [0.], [0.], [0.]])  # external motion

        self.P = matrix([[10, 0, 0, 0, 0],
                         [0, 10, 0, 0, 0],
                         [0, 0, 1000, 0, 0],
                         [0, 0, 0, 1000, 0],
                         [0, 0, 0, 0, 1000]])
        self.H = matrix([[1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0]])
        self.R = matrix([[measurement_noise, 0., 0.],
                         [0., measurement_noise, 0.],
                         [0., 0., measurement_noise]])
        self.I = matrix([[]])
        self.I.identity(5)
        self.last_pred = (0, 0)

    def F_predict(self):  # next state function
        x, y, th, dth, dist = self.x_mat.transpose().value[0]
        return matrix([[x + dist * cos(th + dth),
                        y + dist * sin(th + dth),
                        th + dth,
                        dth,
                        dist]]).transpose()

    def F_jacobian(self):
        x, y, th, dth, dist = self.x_mat.transpose().value[0]
        return matrix([[1, 0, -dist * sin(th + dth), -dist * sin(th + dth), cos(th + dth)],
                       [0, 1, dist * cos(th + dth), dist * cos(th + dth), sin(th + dth)],
                       [0, 0, 1, 1, 0],
                       [0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 1]])

    def filter(self, x, y, th):
        # measurement update
        Z = matrix([[x, y, th]])
        Y = Z.transpose() - (self.H * self.x_mat)
        S = self.H * self.P * self.H.transpose() + self.R
        K = self.P * self.H.transpose() * S.inverse()
        self.x_mat = self.x_mat + (K * Y)
        self.P = (self.I - (K * self.H)) * self.P

        # predict
        self.x_mat = self.F_predict() + self.u
        Fj = self.F_jacobian()
        self.P = Fj * self.P * Fj.transpose()

        # print(distance_between(self.last_pred, (x, y)))
        # self.last_pred = (float(self.x_mat.value[0][0]), float(self.x_mat.value[1][0]))


# This is the function you have to write. Note that measurement is a
# single (x, y) point. This function will have to be called multiple
# times before you have enough information to accurately predict the
# next position. The OTHER variable that your function returns will be
# passed back to your function the next time it is called. You can use
# this to keep track of important information over time.
def estimate_next_pos(measurement, OTHER=None):
    """Estimate the next (x, y) position of the wandering Traxbot
    based on noisy (x, y) measurements."""

    if OTHER is None:
        x, y = measurement
        OTHER = (EKF(x, y), x, y)
        ekf, last_x, last_y = OTHER
        xy_estimate = 0, 0
    else:
        ekf, last_x, last_y = OTHER
        x, y = measurement
        th = atan2(y - last_y, x - last_x)
        ekf_th = ekf.x_mat.value[2][0]
        while th < ekf_th:
            th += 2 * pi
        while th >= ekf_th + pi:
            th -= 2 * pi
        ekf.filter(x, y, th)
        xy_estimate = ekf.x_mat.value[0][0], ekf.x_mat.value[1][0]
    return xy_estimate, (ekf, x, y)


# A helper function you may find useful.
def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def demo_grading(estimate_next_pos_fcn, target_bot, OTHER=None):
    localized = False
    distance_tolerance = 0.01 * target_bot.distance
    ctr = 0
    # if you haven't localized the target bot, make a guess about the next
    # position, then we move the bot and compare your guess to the true
    # next position. When you are close enough, we stop checking.
    # For Visualization
    import turtle  # You need to run this locally to use the turtle module
    window = turtle.Screen()
    window.bgcolor('white')
    window._root.attributes('-topmost', 1)
    size_multiplier = 25.0  # change Size of animation
    broken_robot = turtle.Turtle()
    broken_robot.shape('turtle')
    broken_robot.color('green')
    broken_robot.resizemode('user')
    broken_robot.shapesize(0.1, 0.1, 0.1)
    measured_broken_robot = turtle.Turtle()
    measured_broken_robot.shape('circle')
    measured_broken_robot.color('red')
    measured_broken_robot.resizemode('user')
    measured_broken_robot.shapesize(0.1, 0.1, 0.1)
    prediction = turtle.Turtle()
    prediction.shape('arrow')
    prediction.color('blue')
    prediction.resizemode('user')
    prediction.shapesize(0.1, 0.1, 0.1)
    prediction.penup()
    broken_robot.penup()
    measured_broken_robot.penup()
    # End of Visualization
    while not localized and ctr <= 1000:
        ctr += 1
        measurement = target_bot.sense()
        position_guess, OTHER = estimate_next_pos_fcn(measurement, OTHER)
        target_bot.move_in_circle()
        true_position = (target_bot.x, target_bot.y)
        error = distance_between(position_guess, true_position)
        if error <= distance_tolerance:
            print("You got it right! It took you ", ctr, " steps to localize.")
            localized = True
        if ctr == 1000:
            print("Sorry, it took you too many steps to localize the target.")
        # More Visualization
        measured_broken_robot.setheading(target_bot.heading * 180 / pi)
        measured_broken_robot.goto(measurement[0] * size_multiplier, measurement[1] * size_multiplier - 200)
        measured_broken_robot.stamp()
        broken_robot.setheading(target_bot.heading * 180 / pi)
        broken_robot.goto(target_bot.x * size_multiplier, target_bot.y * size_multiplier - 200)
        broken_robot.stamp()
        prediction.setheading(target_bot.heading * 180 / pi)
        prediction.goto(position_guess[0] * size_multiplier, position_guess[1] * size_multiplier - 200)
        prediction.stamp()
        # End of Visualization
    window.exitonclick()
    return localized


# This is a demo for what a strategy could look like. This one isn't very good.
def naive_next_pos(measurement, OTHER=None):
    """This strategy records the first reported position of the target and
    assumes that eventually the target bot will eventually return to that
    position, so it always guesses that the first position will be the next."""
    if not OTHER:  # this is the first measurement
        OTHER = measurement
    xy_estimate = OTHER
    return xy_estimate, OTHER


# This is how we create a target bot. Check the robot.py file to understand
# How the robot class behaves.
test_target = robot(2.1, 4.3, 0.5, 2 * pi / 34.0, 1.5)
measurement_noise = 0.05 * test_target.distance
test_target.set_noise(0.0, 0.0, measurement_noise)

demo_grading(estimate_next_pos, test_target)
