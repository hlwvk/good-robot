import random
import time
import numpy as np
import math

class Maze(object):

    def __init__(self, session):
        # for testing without session
        if session is None:
            self.behavior_mng_service = None
            self.motion_service = None
            self.posture_service = None
            self.tracker_service = None
            self.leds_service = None
            self.tts_service = None
            self.autonomous_life_service = None
            self.laser_service = None

        else:
            self.behavior_mng_service = session.service("ALBehaviorManager")
            self.motion_service = session.service("ALMotion")
            self.posture_service = session.service("ALRobotPosture")
            self.tracker_service = session.service("ALTracker")
            self.leds_service = session.service("ALLeds")
            self.tts_service = session.service("ALTextToSpeech")
            self.autonomous_life_service = session.service("ALAutonomousLife")
            self.laser_service = session.service("ALLaser")
        self.state0 = 10
        self.terminal_state = 24
        self.pick_up_state = 4
        #self.curr_state = []
        self.curr_state = self.state0
        self.actions = [0, 1, 2, 3]
        self.gameDone = False
        self.max_time_steps = 50
        self.score = 0
        self.state_space = list(range(25))
        self.labels_action_space = ["FORWARD", "BACKWARD", "TURN_LEFT", "TURN_RIGHT"]
        self.action_space = [i for i in range(len(self.labels_action_space))]
        self.dim_a = len(self.action_space)

        self.discount = 0.6
        self.n_iter = 1000
        self.epsilon = 1
        self.timestep_counter = 0

        self.target_active = False
        self.say_target_active = False
        self.game_is_on = True
        self.run_with_luring_bool = False
        self.compute_way_back_bool = False
        self.go_back_bool = False
        self.run_without_luring_bool = False
        self.check_if_should_be_moving_bool = False
        self.ready_to_move_bool = False
        self.start_tracking_bool = True

        self.initial_position = None

        self.curr_trial = 0
        self.max_trials = 2

        self.dim_s = len(self.state_space)
        self.transitions = np.zeros((self.dim_a, self.dim_s, self.dim_s))
        self.rewards = np.zeros((self.dim_s, self.dim_a))
        self.q_table = np.zeros([self.dim_s, self.dim_a])
        self.rewards_backwards = np.zeros((self.dim_s, self.dim_a))
        self.human_feedback = np.zeros((self.dim_s, self.dim_a))

        # init transition function AxSxS
        # moving forward transitions
        for i in range(0, 25):
            # edge cases
            if i in [4, 9, 14, 19, 24]:
                self.transitions[0][i][i] = 1  # stay in place
            else:
                self.transitions[0][i][i + 1] = 1

        # moving backward transitions
        for i in range(0, 25):
            # edge cases
            if i in [0, 5, 10, 15, 20]:
                self.transitions[1][i][i] = 1  # stay in place
            else:
                self.transitions[1][i][i - 1] = 1
        # moving right transitions
        for i in range(0, 20):
            self.transitions[2][i][i + 5] = 1
        # edge cases
        for i in range(20, 25):
            self.transitions[2][i][i] = 1  # stay in place

        # moving left transitions
        # edge cases
        for i in range(0, 5):
            self.transitions[3][i][i] = 1  # stay in place
        # facing south-west-north
        for i in range(5, 25):
            self.transitions[3][i][i - 5] = 1

        # init reward function SxA
        self.rewards = np.full((self.dim_s, self.dim_a), -1)
        self.rewards[self.terminal_state][0] = 100
        self.rewards[self.terminal_state][1] = 100
        self.rewards[self.terminal_state][2] = 100
        self.rewards[self.terminal_state][3] = 100
        self.rewards[self.pick_up_state][0] = 50
        self.rewards[self.pick_up_state][1] = 50
        self.rewards[self.pick_up_state][2] = 50
        self.rewards[self.pick_up_state][3] = 50

        self.rewards_backwards = np.full((self.dim_s, self.dim_a), -1)


    def is_terminal_state(self, state):
        return state == self.terminal_state

    def is_not_a_terminal_state(self, state):
        return state != self.terminal_state

    def is_pick_up_state(self, state):
        return state == self.pick_up_state

    def end_game(self):
        self.gameDone = True
        # log score
        # exit()

    def pick_random_action(self):
        return random.randint(0, self.dim_a - 1)

    def reset(self):
        self.curr_state = self.state0
        self.timestep_counter = 0

    def step(self, action):
        self.timestep_counter += 1
        next_state = np.argmax(self.transitions[action][self.curr_state])
        self.curr_state = next_state
        reward = self.rewards[self.curr_state][action]
        done = self.is_terminal_state(next_state)
        return next_state, reward, done

    def step_backwards(self, action, state):
        next_state = np.argmax(self.transitions[action][state])
        #self.curr_state = next_state
        reward = self.rewards_backwards[next_state][action]
        done = next_state == self.state0
        return next_state, reward, done

    def action_selection_via_feedback(self, state):
        probs = []
        for i in range(0, self.dim_a):
            y = self.human_feedback[state, i]
            if int(y) == 0:
                probs.append(0.0)
            else:
                if y > 50:
                    y = 50
                if y < -50:
                    y = -50
                z = math.pow(0.95, y) / (math.pow(0.95, y) + math.pow(0.05, y))
                probs.append(z)
        probs = np.array(probs)
        if probs.max() == 0.0:
            action = random.randint(0, 3)
        else:
            action = probs.argmax()
        return action

    def step_with_feedback(self):
        # Hyperparameters
        epsilon = 0

        total_timesteps = 0
        training_episodes = 10
        for i in range(1, training_episodes + 1):
            self.reset()
            state = self.state0
            epochs = 0
            done = False

            while not done:
                action = self.action_selection_via_feedback(state)
                state, reward, done = self.step(action)

                epochs += 1
                print(action, state)

                total_timesteps += 1
                if total_timesteps % 100 == 0:
                    print(self.human_feedback)

            if i % 100 == 0:
                print("Episode: %d" % i)

        print("Training finished")
        print('Average timesteps: ', total_timesteps / training_episodes)

    def do_q_learning(self, alpha=0.1, gamma=0.5, epsilon=0.1, training_episodes=1000):

        total_timesteps = 0
        for i in range(1, training_episodes + 1):
            self.reset()
            state = self.state0
            epochs = 0
            done = False

            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = self.pick_random_action()  # Explore action space
                else:
                    action = np.argmax(self.q_table[state])  # Exploit learned values

                next_state, reward, done = self.step(action)

                old_value = self.q_table[state, action]

                next_max = np.max(self.q_table[next_state])

                #reward += self.human_feedback[state, action]

                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                self.q_table[state, action] = new_value

                state = next_state
                epochs += 1

                total_timesteps += 1

            if i % 100 == 0:
                print("Episode: %d" % i)

        print("Training finished")
        print('Average timesteps: ', total_timesteps / training_episodes)
        print(self.q_table)

    def do_q_learning_backwards(self, alpha=0.1, gamma=0.5, epsilon=0.001, training_episodes=1000):
        # reset q table
        self.q_table = np.zeros([self.dim_s, self.dim_a])
        total_timesteps = 0
        epsilon = 0
        for i in range(1, training_episodes + 1):
            #self.reset()
            state = self.curr_state
            epochs = 0
            done = False

            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = self.pick_random_action()  # Explore action space
                else:
                    action = np.argmax(self.q_table[state])  # Exploit learned values

                next_state, reward, done = self.step_backwards(action, state)

                old_value = self.q_table[state, action]

                next_max = np.max(self.q_table[next_state])

                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                self.q_table[state, action] = new_value

                state = next_state
                epochs += 1

                total_timesteps += 1

            if i % 100 == 0:
                print("Episode: %d" % i)
                print('Average timesteps: ', total_timesteps / training_episodes)
        print("Training finished")

        #print(self.q_table)

    def do_test_run(self):
        print("Test run:")
        self.reset()
        state = self.state0
        done = False
        total_timesteps = 0

        while not done:
            action = np.argmax(self.q_table[state])
            state, reward, done = self.step(action)
            total_timesteps += 1
        print('Timesteps: ', total_timesteps)

    def do_test_run_in_reality(self):
        print("Test run:")
        self.reset()
        state = self.state0
        done = False
        total_timesteps = 0

        while not done:
            action = np.argmax(self.q_table[state])
            state, reward, done = self.step(action)
            self.do_one_action_in_reality(action=action)
            total_timesteps += 1

            #time.sleep(2)

        print('Timesteps: ', total_timesteps)

    def do_one_action_in_reality(self, action):
        #action = self.pick_random_action()
        name = "behavior_%d" % action
        behavior_path = ".lastUploadedChoregrapheBehavior/" + name
        if self.behavior_mng_service.isBehaviorRunning(behavior_path):
            self.behavior_mng_service.stopBehavior(behavior_path)
        self.behavior_mng_service.startBehavior(behavior_path)
        #self.behavior_mng_service.runBehavior(behavior_path)
        print("Running behaviors: ", self.behavior_mng_service.getRunningBehaviors())
        #self.behavior_mng_service.stopBehavior(behavior_path)

    def do_choregraphe_action(self, action):
        #action = self.pick_random_action()
        # change new sideways actions
        if action == 0:
            name = "behavior_%d" % action
        if action == 1:
            name = "behavior_%d" % action
        if action == 2:
            name = "behavior_%d" % 5
        if action == 3:
            name = "behavior_%d" % 6
        behavior_path = ".lastUploadedChoregrapheBehavior/" + name
        #if self.behavior_mng_service.isBehaviorRunning(behavior_path):
        #    self.behavior_mng_service.stopBehavior(behavior_path)
        #self.behavior_mng_service.stopAllBehaviors()
        #self.behavior_mng_service.startBehavior(behavior_path)
        self.behavior_mng_service.runBehavior(behavior_path)
        print("Running behaviors: ", self.behavior_mng_service.getRunningBehaviors())
        #self.behavior_mng_service.stopBehavior(behavior_path)

    def check_if_should_be_moving(self):
        #print("rel pos", tracker_service.getRelativePosition())
        #print("robot pos", tracker_service.getRobotPosition())
        #print("target pos", tracker_service.getTargetPosition())
        target = self.tracker_service.getTargetPosition()
        targetLost = self.tracker_service.isTargetLost()
        move = False
        action = None
        #print(targetLost)
        # if target detected, say so once
        if len(target) >= 1 and self.say_target_active:
            #self.tts_service.say("Got it!")
            self.say_target_active = False
        if targetLost is True:
            #print("targetLost")
            return move, action
            #self.tts_service.say("Target lost")
        # move only if target in sight
        else:
            # check if list is empty
            if not target:
                x_target = 0.0
                y_target = 0.0
            else:
                x_target = target[0]
                y_target = target[1]
            if y_target >= 0.5:
                action = 2
                move = True
            elif y_target <= -0.5:
                action = 3
                move = True
            elif x_target >= 0.5:
                action = 0
                move = True
        return move, action

    def do_action_in_reality(self, action, use_choregraphe=False):
        print("action is", action)
        last_position = self.get_robot_position()
        if use_choregraphe:
            self.do_choregraphe_action(action)
        else:
            if action == 0:
                if self.curr_state in [4, 9, 14, 19, 24]:
                    print("can't move there")
                    #self.tts_service.say("Can't move there")
                    return
                else:
                    x = 0.5
                    y = 0.0
                    theta = 0.0
                    self.motion_service.moveInit()
                    #motion = self.motion_service.moveTo(x, y, theta, _async=True)
                    #motion.wait()

                    self.motion_service.moveTo(x, y, theta)
                    self.motion_service.waitUntilMoveIsFinished()

            if action == 1:
                if self.curr_state in [0, 5, 10, 15, 20]:
                    print("can't move there")
                    #self.tts_service.say("Can't move there")
                    return
                else:
                    x = -0.5
                    y = 0.0
                    theta = 0.0
                    self.motion_service.moveInit()
                    self.motion_service.moveTo(x, y, theta)
                    self.motion_service.waitUntilMoveIsFinished()

            if action == 2:
                if self.curr_state in [20, 21, 22, 23, 24]:
                    print("can't move there")
                    #self.tts_service.say("Can't move there")
                    return
                else:
                    # change turn to walk sideways
                    #x = 0.0
                    #y = 0.0
                    #theta = math.pi / 2
                    x = 0.0
                    y = 0.5
                    theta = 0.0
                    self.motion_service.moveInit()
                    self.motion_service.moveTo(x, y, theta)
                    self.motion_service.waitUntilMoveIsFinished()

            if action == 3:
                if self.curr_state in [0, 1, 2, 3, 4]:
                    print("can't move there")
                    #self.tts_service.say("Can't move there")
                    return
                else:
                    # change turn to walk sideways
                    #x = 0.0
                    #y = 0.0
                    #theta = - math.pi / 2
                    x = 0.0
                    y = -0.5
                    theta = 0.0
                    self.motion_service.moveInit()
                    self.motion_service.moveTo(x, y, theta)
                    self.motion_service.waitUntilMoveIsFinished()

        curr_position = self.get_robot_position()
        delta_position = self.get_difference_in_position(last_position, curr_position)
        # if difference in position big enough, change state, and, if applicable, change feedback matrix
        if delta_position > 0.05:
            if self.run_with_luring_bool:
                self.human_feedback[self.curr_state, action] += 100
            self.curr_state = np.argmax(self.transitions[action][self.curr_state])
            print(curr_position, 'curr_position', last_position, 'last_position')
            print(self.curr_state, 'curr_state', action, 'action', delta_position, 'delta_position')

        self.check_if_should_be_moving_bool = True

    def get_difference_in_position(self, last_position, curr_position):
        # input: lists of the form [x, y, z]
        return np.abs(np.subtract(last_position[0], curr_position[0])) + np.abs(np.subtract(last_position[1], curr_position[1])) + np.abs(np.subtract(last_position[2], curr_position[2]))

    def adjust_position(self):
        # adjust position using MRE sensors
        curr_position = self.get_robot_position()
        delta_position = self.get_difference_in_position(self.initial_position, curr_position)
        print(curr_position, 'curr_position')
        print(self.initial_position, 'initial_position')
        x = np.subtract(self.initial_position[0], curr_position[0])
        y = np.subtract(self.initial_position[1], curr_position[1])
        z = np.subtract(self.initial_position[2], curr_position[2])
        self.motion_service.moveTo(x.item(), y.item(), z.item())

    def set_ball_tracking(self, ballSize=0.06):
        targetName = "RedBall"
        diameterOfBall = ballSize
        self.tracker_service.registerTarget(targetName, diameterOfBall)
        mode = "Head"
        self.tracker_service.setMode(mode)
        self.tracker_service.track(targetName)
        self.say_target_active = True
        self.leds_service.fadeRGB('FaceLeds', 1, 0.8, 0, 0.5)
        self.tts_service.say("Tracking")

    def set_states(self):
        self.state0 = random.choice([0, 1, 5, 6, 10, 11, 15, 16, 20, 21])
        self.terminal_state = 24
        self.pick_up_state = 4
        print("new state0", self.state0)

    def go_back_to_state0(self):
        self.rewards_backwards = np.full((self.dim_s, self.dim_a), -1)
        self.rewards_backwards[self.state0][0] = 100
        self.rewards_backwards[self.state0][1] = 100
        self.rewards_backwards[self.state0][2] = 100
        self.rewards_backwards[self.state0][3] = 100
        self.do_q_learning_backwards()
        while self.curr_state != self.state0:
            action = np.argmax(self.q_table[self.curr_state])
            self.curr_state = np.argmax(self.transitions[action][self.curr_state])

    def get_robot_position(self, use_sensor_values=True):
        #useSensorValues = False
        result = self.motion_service.getRobotPosition(use_sensor_values)
        print("Robot Position", result, use_sensor_values)
        return result

    def run_with_luring(self):
        if self.start_tracking_bool:
            self.start_tracking_bool = False
            self.set_ball_tracking()
        should_be_moving = False
        action = None
        if self.check_if_should_be_moving_bool:
            should_be_moving, action = self.check_if_should_be_moving()

            # doesn't work that way
            while self.motion_service.moveIsActive():
                # do something
                print("Move is active")

        if should_be_moving:
            self.do_action_in_reality(action, use_choregraphe=False)

        if self.is_pick_up_state(self.curr_state):
            print("pick-up state reached")

            #self.behavior_mng_service.startBehavior('animations/Stand/Emotions/Positive/Excited_3')
            #self.behavior_mng_service.runBehavior('animations/Stand/Emotions/Positive/Interested_2')
            self.tts_service.say("Exhilarating")
            self.leds_service.fadeRGB('FaceLeds', 1, 0.8, 0, 0.1)

        if self.is_terminal_state(self.curr_state):
            self.run_with_luring_bool = False
            self.compute_way_back_bool = True
            print("run done")
            #self.tts_service.say("Goal")
            print(self.human_feedback)
            self.behavior_mng_service.runBehavior('animations/Stand/Emotions/Positive/Happy_1')
            # yellow eyes -> white eyes
            self.leds_service.fadeRGB('FaceLeds', 0, 0, 0, 0.1)
            self.curr_trial += 1
            self.tracker_service.stopTracker()
            self.posture_service.goToPosture("StandInit", 1.0)

    def compute_way_back(self):
        self.compute_way_back_bool = False
        self.set_states()
        # for testing set state0 to 10
        self.state0 = 10
        # self.go_back_to_state0()
        self.rewards_backwards = np.full((self.dim_s, self.dim_a), -1)
        self.rewards_backwards[self.state0][0] = 100
        self.rewards_backwards[self.state0][1] = 100
        self.rewards_backwards[self.state0][2] = 100
        self.rewards_backwards[self.state0][3] = 100
        self.do_q_learning_backwards()
        self.go_back_bool = True
        self.tts_service.say("Going back")


    def go_back(self):
        action = np.argmax(self.q_table[self.curr_state])
        self.do_action_in_reality(action)
        if self.curr_state == self.state0:
            self.go_back_bool = False

            self.tts_service.say("Adjusting")
            self.adjust_position()

            self.tts_service.say("Ready")
            print("Back at starting position", self.state0)

            if self.curr_trial == self.max_trials:
                self.run_without_luring_bool = True
            else:
                self.start_tracking_bool = True
                self.run_with_luring_bool = True


    def run_without_luring(self):
        #self.tts_service.say("On my own now...")
        action = self.action_selection_via_feedback(self.curr_state)
        self.do_action_in_reality(action)
        #self.curr_state = np.argmax(self.transitions[action][self.curr_state])
        if self.is_terminal_state(self.curr_state):
            self.run_without_luring_bool = False
            print("Run Done")
            #self.tts_service.say("Goal")
            self.behavior_mng_service.runBehavior('animations/Stand/Emotions/Positive/Happy_1')
            # yellow eyes -> white eyes
            self.leds_service.fadeRGB('FaceLeds', 0, 0, 0, 0.1)
            self.posture_service.goToPosture("StandInit", 1.0)
            self.compute_way_back_bool = True


    def keep_running(self):
        self.posture_service.goToPosture("StandInit", 1.0)
        self.motion_service.setOrthogonalSecurityDistance(0.01)
        self.motion_service.setTangentialSecurityDistance(0.01)
        self.autonomous_life_service.setAutonomousAbilityEnabled("All", False)
        #self.motion_service.setFallManagerEnabled(False)
        #self.motion_service.setExternalCollisionProtectionEnabled("Arms", False)
        #self.motion_service.setExternalCollisionProtectionEnabled("All", False)
        self.laser_service.laserOFF()

        self.set_states()

        self.state0 = 10
        self.curr_state = 10
        print("curr_state", self.curr_state)
        #self.luring = True
        self.curr_trial = 0
        self.max_trials = 2
        #do_run_without_luring = False

        self.run_with_luring_bool = True

        # set eye-LEDs to white
        self.leds_service.fadeRGB('FaceLeds', 0, 0, 0, 0.1)

        self.initial_position = self.get_robot_position()

        try:
            while True:
                time.sleep(0.1)
                #self.get_robot_position()
                if self.run_with_luring_bool:
                    self.run_with_luring()
                if self.compute_way_back_bool:
                    self.compute_way_back()
                if self.go_back_bool:
                    self.go_back()
                if self.run_without_luring_bool:
                    self.run_without_luring()

        except KeyboardInterrupt:
            print "Interrupted by user"
            print "Stopping..."
        self.tts_service.say("goodbye")
        self.tracker_service.stopTracker()
        self.tracker_service.unregisterAllTargets()


def main():
    maze = Maze(session=None)
    maze.keep_running()

if __name__ == '__main__':
    main()

