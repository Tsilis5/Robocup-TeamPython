import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .player import Player
from .ball import Ball
from .team import Team
from .circle import Circle
from .small_circle import Small_Circle
import gym
from gym import spaces
import numpy as np
import random
import math
from pymunk.vec2d import Vec2d
import pymunk.matplotlib_util
import pymunk
import matplotlib.pyplot as plt

WIDTH = 105

HEIGHT = 68
CIRCLE_WEIGHT = 20
# goalkeeper_restricted_area = WIDTH * 0.4/8

# # Define the constant
# GOALKEEPER_RESTRICTED_AREA = goalkeeper_restricted_area

GOAL_SIZE = 20

TOTAL_TIME = 80000  # 30 s

TIME_STEP = 0.1  # 0.1 s

# player number each team, less than 10
NUMBER_OF_PLAYER = 5

# BALL_MAX_VELOCITY = 25
BALL_MAX_VELOCITY = 1
PLAYER_MAX_VELOCITY = 10
PLAYER_SPEED = 10
TACKLE_DISTANCE = 1

BALL_WEIGHT = 10
PLAYER_WEIGHT = 20
PLAYER_RADIUS = 2.5

PLAYER_FORCE_LIMIT = 40
BALL_FORCE_LIMIT = 120

BALL_max_arr = np.array([WIDTH, HEIGHT, BALL_MAX_VELOCITY, BALL_MAX_VELOCITY])
BALL_min_arr = np.array([0, 0, -BALL_MAX_VELOCITY, -BALL_MAX_VELOCITY])
BALL_avg_arr = (BALL_max_arr + BALL_min_arr) / 2
BALL_range_arr = (BALL_max_arr - BALL_min_arr) / 2

padding = 3
PLAYER_max_arr = np.array(
    [WIDTH + padding, HEIGHT, PLAYER_MAX_VELOCITY, PLAYER_MAX_VELOCITY]
)
PLAYER_min_arr = np.array([0 - padding, 0, -PLAYER_MAX_VELOCITY, -PLAYER_MAX_VELOCITY])
PLAYER_avg_arr = (PLAYER_max_arr + PLAYER_min_arr) / 2
PLAYER_range_arr = (PLAYER_max_arr - PLAYER_min_arr) / 2

# get the vector pointing from [coor2] to [coor1] and
# its magnitude


def get_vec(coor_t, coor_o):
    vec = [coor_t[0] - coor_o[0], coor_t[1] - coor_o[1]]
    vec_mag = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    return vec, vec_mag


class Robosoccer(gym.Env):
    def __init__(
        self,
        width=WIDTH,
        height=HEIGHT,
        total_time=TOTAL_TIME,
        debug=False,
        number_of_player=NUMBER_OF_PLAYER,
    ):
        self.width = width
        self.height = height
        self.total_time = total_time
        self.debug = debug
        self.number_of_player = number_of_player
        self.score_team_a = 0
        self.score_team_b = 0
        self.is_pass = False
        self.is_upper_goal = False
        self.is_lower_goal = False
        self.is_team_B_pass = False
        self.is_upper_goal_team_B = False
        self.is_lower_goal_team_B = False
        self.midfielder_pass = False
        self.defender_pass = False
        self.PLAYER_avg_arr = np.tile(PLAYER_avg_arr, number_of_player)
        self.PLAYER_range_arr = np.tile(PLAYER_range_arr, number_of_player)

        # action space
        # 1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        # 2) Action Keys: Discrete 5  - noop[0], dash[1], shoot[2], press[3], pass[4] - params: min: 0, max: 4
        self.action_space = spaces.MultiDiscrete([5, 5] * self.number_of_player)

        # observation space (normalized)
        # [0] x position
        # [1] y position
        # [2] x velocity
        # [3] y velocity
        self.observation_space = spaces.Box(
            low=np.array(
                [-1.0, -1.0, -1.0, -1.0] * (1 + self.number_of_player * 2),
                dtype=np.float32,
            ),
            high=np.array(
                [1.0, 1.0, 1.0, 1.0] * (1 + self.number_of_player * 2), dtype=np.float32
            ),
            dtype=np.float32,
        )

        # create space
        self.space = pymunk.Space()
        self.space.gravity = 0, 0

        # Amount of simple damping to apply to the space.
        # A value of 0.9 means that each body will lose 10% of its velocity per second.
        self.space.damping = 0.95

        # create walls
        self._setup_walls(width, height)

        # Teams
        self.team_A = Team(
            self.space,
            width,
            height,
            player_weight=PLAYER_WEIGHT,
            player_max_velocity=PLAYER_MAX_VELOCITY,
            color=(1, 0, 0, 1),  # red
            side="left",
            player_number=self.number_of_player,
        )

        self.team_B = Team(
            self.space,
            width,
            height,
            player_weight=PLAYER_WEIGHT,
            player_max_velocity=PLAYER_MAX_VELOCITY,
            color=(0, 0, 1, 1),  # red
            side="right",
            player_number=self.number_of_player,
        )

        self.player_arr = self.team_A.player_array + self.team_B.player_array

        # Ball
        self.ball = Ball(
            self.space,
            self.width * 0.5,
            self.height * 0.5,
            mass=BALL_WEIGHT,
            max_velocity=BALL_MAX_VELOCITY,
            elasticity=0.2,
        )
        self.circle = Circle(
            self.space, self.width * 0.5, self.height * 0.5, mass=CIRCLE_WEIGHT
        )

        self.small_circle = Small_Circle(
            self.space, self.width * 0.5, self.height * 0.5, mass=CIRCLE_WEIGHT
        )

        self.observation = self.reset()

    def _position_to_initial(self, goal=""):
        self.ball.set_position(self.width * 0.5, self.height * 0.5)

        self.team_A.set_position_to_initial()

        self.team_B.set_position_to_initial()

        self.circle.set_position(self.width * 0.5, self.height * 0.5)  # new circle
        self.small_circle.set_position(self.width * 0.5, self.height * 0.5)

        # set the ball velocity to zero
        self.ball.body.velocity = 0, 0

        # after set position, need to step the space so that the object
        # move to the target position
        self.space.step(0.0001)

        self.observation = self._get_observation()

        if goal == "team_b":
            player = self.team_A.player_array[8]
            ball_position_x, ball_position_y = self.ball.get_position()
            player.set_position(ball_position_x - 3, ball_position_y)
            closest_teammate = self.get_closest_teammate(
                player, self.team_A.player_array
            ).get_position()
            self.shot_ball_to_goal(
                goal_target_x=closest_teammate[0],
                goal_target_y=closest_teammate[1],
                ball_position=self.ball.get_position(),
            )
            # self.pass_ball_to_player(player, closest_teammate)

        elif goal == "team_a":
            player = self.team_B.player_array[8]
            ball_position_x, ball_position_y = self.ball.get_position()
            player.set_position(ball_position_x + 3, ball_position_y)
            closest_teammate = self.get_closest_teammate(
                player, self.team_B.player_array
            ).get_position()
            self.shot_ball_to_goal(
                goal_target_x=closest_teammate[0],
                goal_target_y=closest_teammate[1],
                ball_position=self.ball.get_position(),
            )

    def reset(self):
        self.current_time = 0
        self.ball_owner_side = random.choice(["left", "right"])
        self._position_to_initial()
        return self._get_observation()

    # normalize ball observation

    def _normalize_ball(self, ball_observation):
        ball_observation = (ball_observation - BALL_avg_arr) / BALL_range_arr
        return ball_observation

    # normalize player observation

    def _normalize_player(self, player_observation):
        player_observation = (
            player_observation - self.PLAYER_avg_arr
        ) / self.PLAYER_range_arr
        return player_observation

    # normalized observation
    def _get_observation(self):
        ball_observation = self._normalize_ball(np.array(self.ball.get_observation()))

        team_A_observation = self._normalize_player(self.team_A.get_observation())

        team_B_observation = self._normalize_player(self.team_B.get_observation())

        obs = np.concatenate((ball_observation, team_A_observation, team_B_observation))

        return obs

    def _setup_walls(self, width, height):
        # Create walls.
        static = [
            pymunk.Segment(
                self.space.static_body, (0, 0), (0, height / 2 - GOAL_SIZE / 2), 1
            ),
            pymunk.Segment(
                self.space.static_body, (0, height / 2 + GOAL_SIZE / 2), (0, height), 1
            ),
            pymunk.Segment(self.space.static_body, (0, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (width, 0),
                (width, height / 2 - GOAL_SIZE / 2),
                1,
            ),
            pymunk.Segment(
                self.space.static_body,
                (width, height / 2 + GOAL_SIZE / 2),
                (width, height),
                1,
            ),
            pymunk.Segment(self.space.static_body, (0, 0), (width, 0), 1),
        ]
        for i in static:
            i.color = (1, 1, 1)

        under_lines = [

            pymunk.Segment(
                self.space.static_body, (width / 2, height), (width / 2, 0), 1
            ),  # new center line
            pymunk.Segment(
                self.space.static_body,
                (8, height / 2 - GOAL_SIZE),
                (8, height / 2 + GOAL_SIZE),
                1,
            ),  # new goal indoor line

            pymunk.Segment(
                self.space.static_body,
                (8, height / 2 - GOAL_SIZE),
                (0, height / 2 - GOAL_SIZE),
                1,
            ),  # new  indoor horizntal line

            pymunk.Segment(
                self.space.static_body,
                (8, height / 2 + GOAL_SIZE),
                (0, height / 2 + GOAL_SIZE),
                1,
            ),  # new2 indoor horizonatl line
            pymunk.Segment(
                self.space.static_body,
                (width - 8, height / 2 - GOAL_SIZE),
                (width - 8, height / 2 + GOAL_SIZE),
                1,
            ),  # new small goal vertical

            pymunk.Segment(
                self.space.static_body,
                (width, height / 2 - GOAL_SIZE),
                (width - 8, height / 2 - GOAL_SIZE),
                1,
            ),  # new2 right indoor horizontal

            pymunk.Segment(
                self.space.static_body,
                (width, height / 2 + GOAL_SIZE),
                (width - 8, height / 2 + GOAL_SIZE),
                1,
            ),  # new right indoor horizontal
        ]
        for i in under_lines:
            i.color = (1, 1, 1)
            i.sensor = True

        static_goal = [
            pymunk.Segment(
                self.space.static_body,
                (-0.2, height / 2 - GOAL_SIZE / 2),
                (-0.2, height / 2 + GOAL_SIZE / 2),
                4,
            ),  # small goal vrtical
            pymunk.Segment(
                self.space.static_body,
                (width, height / 2 - GOAL_SIZE / 2),
                (width, height / 2 + GOAL_SIZE / 2),
                4,
            ),  # small goal vertical
        ]

        for i in static_goal:
            i.color = (0, 0, 0, 1)

        for s in static + static_goal:  # goal side add krna ha yaha
            s.friction = 1.0
            s.group = 1
            s.collision_type = 1

        self.under_lines = under_lines
        self.space.add(under_lines)
        self.static = static
        self.space.add(static)
        self.static_goal = static_goal
        self.space.add(static_goal)

    def render(self):
        padding = 5
        ax = plt.axes(
            xlim=(0 - padding, self.width + padding),
            ylim=(0 - padding, self.height + padding),
        )
        ax.set_aspect("equal")
        ax.set_facecolor("green")  # Set the background color to green
        o = pymunk.matplotlib_util.DrawOptions(ax)
        self.space.debug_draw(o)
        plt.show()

    # return true and wall index if the ball is in contact with the walls

    def ball_contact_wall(self):
        wall_index, i = -1, 0
        for wall in self.static:
            if self.ball.shape.shapes_collide(wall).points != []:
                wall_index = i
                return True, wall_index
            i += 1
        return False, wall_index

    def check_and_fix_out_bounds(self):
        out, wall_index = self.ball_contact_wall()
        if out:
            bx, by = self.ball.get_position()
            dbx, dby, dpx, dpy = 0, 0, 0, 0

            if wall_index == 1 or wall_index == 0:  # left bound
                dbx, dpx = 3.5, 1
            elif wall_index == 3 or wall_index == 4:
                dbx, dpx = -3.5, -1
            elif wall_index == 2:
                dby, dpy = -3.5, -1
            else:
                dby, dpy = 3.5, 1

            self.ball.set_position(bx + dbx, by + dby)
            self.ball.body.velocity = 0, 0

            if self.ball_owner_side == "right":
                get_ball_player = random.choice(self.team_A.player_array)
                self.ball_owner_side = "left"
            elif self.ball_owner_side == "left":
                get_ball_player = random.choice(self.team_B.player_array)
                self.ball_owner_side = "right"
            else:
                print("invalid side")

            get_ball_player.set_position(bx + dpx, by + dpy)
            get_ball_player.body.velocity = 0, 0
        else:
            pass
        return out

    # return true if score

    def ball_contact_goal(self):
        goal = False
        for goal_wall in self.static_goal:
            goal = goal or self.ball.shape.shapes_collide(goal_wall).points != []

        if goal:
            self.is_pass = False
            self.is_upper_goal = False
            self.is_lower_goal = False
            self.midfielder_pass = False
            self.defender_pass = False
            self.is_team_B_pass = False
            self.is_upper_goal_team_B = False
            self.is_lower_goal_team_B = False

        return goal

    # if player has contact with ball and move, let the ball move with the player.

    def _ball_move_with_player(self, player):
        if self.ball.has_contact_with(player):
            self.ball.body.velocity = player.body.velocity
        else:
            pass

    def get_distance(self, pos1, pos2):
        """Returns the Euclidean distance between two positions."""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def get_closest_opponent(self, player, opponent_array):
        """Returns the closest opponent to the given player."""
        closest_opponent = None
        closest_distance = float("inf")
        for opponent in opponent_array:
            distance = self.get_distance(player.get_position(), opponent.get_position())
            if distance < closest_distance:
                closest_opponent = opponent
                closest_distance = distance
        return closest_opponent

    def get_closest_teammate(self, player, players_array):
        """Returns the closest opponent to the given player."""
        closest_teammate = None
        closest_distance = float("inf")
        for teammate in players_array:
            if teammate != player:
                distance = self.get_distance(
                    player.get_position(), teammate.get_position()
                )
                if distance < closest_distance:
                    closest_teammate = teammate
                    closest_distance = distance
        return closest_teammate

    def get_opponent_goalie_player(self, opponent_array):
        """Returns the closest opponent to the given player."""
        for opponent in opponent_array:
            if opponent.role == "goalie":
                return opponent

    def get_opponent_goal_target(self, player):
        if player.side == "left":
            goal = [self.width, self.height / 2]
        elif player.side == "right":
            goal = [0, self.height / 2]

        return goal[0], goal[1]

    def get_closest_teammate_attacker(self, player, players_array):
        """Returns the closest opponent to the given player."""
        closest_teammate = None
        closest_distance = float("inf")
        for teammate in players_array:
            if teammate != player and teammate.role == "attacker":
                distance = self.get_distance(
                    player.get_position(), teammate.get_position()
                )
                if distance < closest_distance:
                    closest_teammate = teammate
                    closest_distance = distance
        return closest_teammate

    def _attacker_shoot(self, player, force_x, force_y, target_x, target_y):
        """
        This function calculates the direction and force of a shot and applies it to the ball, while
        reducing the player's stamina.
        
        :param player: The player object that is shooting the ball
        :param force_x: The x-component of the force to apply to the ball. However, it is not used in
        the function and can be removed
        :param force_y: force_y is the vertical force to apply to the ball during the shot. It is a
        scalar value that determines how high or low the ball will travel during the shot
        :param target_x: The x-coordinate of the target position where the player wants to shoot the
        ball
        :param target_y: The y-coordinate of the target position where the player wants to shoot the
        ball
        """
        ball_pos = self.ball.get_position()
        player_pos = player.get_position()

        # Calculate the direction of the shot
        target_direction = np.array([target_x, target_y]) - np.array(ball_pos)
        target_direction /= np.linalg.norm(target_direction)

        # Calculate the force to apply to the ball
        shot_force = BALL_FORCE_LIMIT * np.array([target_direction[0], force_y])

        # Apply the force to the ball
        self.ball.apply_force_to_ball(*shot_force)

        # Reduce the player's stamina
        self.ball.body.velocity /= 2

    def _move_towards_ball(self, player):
        """
        The function moves a player towards the ball and stops the player if they are within a certain
        distance of the ball.
        
        :param player: The "player" parameter is an instance of a class representing a player in a
        soccer game. It is passed to the "_move_towards_ball" method as an argument
        """
        ball_pos = self.ball.get_position()
        player_pos = player.get_position()
        distance_to_ball = math.sqrt(
            (ball_pos[0] - player_pos[0]) ** 2 + (ball_pos[1] - player_pos[1]) ** 2
        )
        # print(distance_to_ball)
        if distance_to_ball > TACKLE_DISTANCE:
            # Move towards the ball
            target_x = ball_pos[0] - player_pos[0]
            target_y = ball_pos[1] - player_pos[1]
            norm = math.sqrt(target_x**2 + target_y**2)
            # print(norm)
            if norm > 0:
                target_x *= PLAYER_SPEED / norm
                target_y *= PLAYER_SPEED / norm

            # print(target_x, target_y)
            player.apply_force_to_player(target_x, target_y)
        else:
            player_velocity_x, player_velocity_y = player.get_velocity()
            # Stop moving and wait for the ball to come closer
            player.apply_force_to_player(-player_velocity_x, -player_velocity_y)

    def _move_towards_target(self, player, target_x, target_y):
        """
        This function moves a player towards a target position by applying a force in the direction of
        the target.
        
        :param player: The player parameter is an object representing the player in the game. It likely
        has attributes such as position, velocity, and acceleration, and methods for applying forces and
        updating its position
        :param target_x: The x-coordinate of the target position that the player is trying to move
        towards
        :param target_y: `target_y` is the y-coordinate of the target position towards which the player
        is moving
        :return: If the distance between the player and the target is less than 0.01, nothing is
        returned. Otherwise, the function applies a force to the player towards the target and there is
        no explicit return statement. Therefore, the function does not return anything.
        """
        player_pos = player.get_position()
        dx, dy = target_x - player_pos[0], target_y - player_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        if distance < 0.01:
            return
        force_x = dx / distance
        force_y = dy / distance
        player.apply_force_to_player(PLAYER_WEIGHT * force_x, PLAYER_WEIGHT * force_y)

    def _get_shooting_direction(self, player_position, target_position):
        """
        This function calculates the shooting direction from a player's position to a target's position.
        
        :param player_position: The position of the player, represented as a tuple of (x, y) coordinates
        :param target_position: The position of the target that the player is shooting at. It is a tuple
        containing the x and y coordinates of the target
        :return: a tuple of two values, which represent the shooting direction as x and y components of
        a unit vector. If the distance between the player and the target is less than 0.01, the function
        returns (0, 0) to avoid division by zero.
        """
        dx, dy = (
            target_position[0] - player_position[0],
            target_position[1] - player_position[1],
        )
        distance = np.sqrt(dx**2 + dy**2)
        if distance < 0.01:
            return 0, 0
        force_x = dx / distance
        force_y = dy / distance
        return force_x, force_y

    def _player_towards_goal(self, player, goal_x, goal_y):
        """
        This function calculates the force vector needed for a player to move towards a goal and applies
        it to the player.
        
        :param player: The player object that we want to move towards the goal
        :param goal_x: The x-coordinate of the goal towards which the player is moving
        :param goal_y: The y-coordinate of the goal towards which the player is moving
        """
        player_x, player_y = player.get_position()

        # Calculate the angle between the player and the ball
        angle = math.atan2(goal_y - player_y, goal_x - player_x)

        # Calculate the force vector in the x and y directions
        force_x = PLAYER_FORCE_LIMIT * math.cos(angle)
        force_y = PLAYER_FORCE_LIMIT * math.sin(angle)

        # Apply the calculated force vector to the player
        player.apply_force_to_player(force_x, force_y)

    def _player_towards_ball(self, player, ball_x, ball_y):
        """
        This function calculates the force vector needed for a player to move towards a ball and applies
        it to the player.
        
        :param player: This parameter is an object representing the player in the game. It likely has
        properties such as position, velocity, and acceleration, as well as methods for applying forces
        and updating its state
        :param ball_x: The x-coordinate of the ball's position
        :param ball_y: The y-coordinate of the ball's position on the field
        """
        player_x, player_y = player.get_position()

        # Calculate the angle between the player and the ball
        angle = math.atan2(ball_y - player_y, ball_x - player_x)

        # Calculate the force vector in the x and y directions
        force_x = PLAYER_FORCE_LIMIT * math.cos(angle)
        force_y = PLAYER_FORCE_LIMIT * math.sin(angle)

        # Apply the calculated force vector to the player
        player.apply_force_to_player(force_x, force_y)

    def should_move_towards_ball(self, teammates):
        """
        This function checks if any teammate has contact with the ball and returns a boolean value
        accordingly.
        
        :param teammates: a list of Player objects representing the teammates of the current player
        :return: a boolean value indicating whether there is a teammate with the ball or not. If there
        is at least one teammate with the ball, the function returns True. Otherwise, it returns False.
        """
        teammate_with_ball = False
        for player in teammates:  # or self.team_A.players, depending on the team
            if self.ball.has_contact_with(player):
                teammate_with_ball = True
                break
        return teammate_with_ball

    def is_opponent_with_ball(self, player_array):
        """
        This function checks if any opponent in a given array has contact with a ball.
        
        :param player_array: The player_array parameter is a list or array of objects representing
        players in a game or simulation
        :return: a boolean value indicating whether there is an opponent with the ball or not. If there
        is at least one player in the player_array that has contact with the ball, the function returns
        True. Otherwise, it returns False.
        """
        opponent_with_ball = False
        for player in player_array:
            if self.ball.has_contact_with(player):
                opponent_with_ball = True
                break
        return opponent_with_ball

    def opponent_moving_towards_ball(self, opponent, ball_x, ball_y):
        """
        This function checks if any opponent player is closer than 5 units to the ball's position.
        
        :param opponent: A list of objects representing the opponent players in a game
        :param ball_x: The x-coordinate of the ball's current position on the field
        :param ball_y: The parameter ball_y represents the y-coordinate of the ball's position on the
        field
        :return: a boolean value indicating whether at least one player from the opponent team is closer
        than 5 units to the ball's position.
        """
        opponent_closer = False
        for player in opponent:
            opponent_x, opponent_y = player.get_position()
            distance = math.sqrt(
                (opponent_x - ball_x) ** 2 + (opponent_y - ball_y) ** 2
            )
            if distance < 5:
                opponent_closer = True
                break
        return opponent_closer

    def check_shoot_goal(self, player, goalie):
        """
        This function checks if the player should shoot the ball towards the goal and shoots it
        accordingly.
        
        :param player: The player object representing the player who is attempting to shoot the ball
        towards the goal
        :param goalie: The goalie parameter is an object representing the goalkeeper in a soccer game.
        It is used in the check_shoot_goal method to determine whether the player should shoot the ball
        towards the upper, lower, or center part of the goal. The method calculates the positions of the
        goal and the player, and uses
        """
        goal_up = [WIDTH, (HEIGHT / 2) + (GOAL_SIZE / 2)]
        goal_down = [WIDTH, (HEIGHT / 2) - (GOAL_SIZE / 2)]
        goal_center = [WIDTH, HEIGHT / 2]
        player_x, player_y = player.get_position()
        goalie_x, goalie_y = goalie.get_position()

        if self.should_goal_shoot(goalie_x, goalie_y, goal_up[1], player_y):
            self.shot_ball_to_goal(player, goal_up[0], goal_up[1])
        elif self.should_goal_shoot(goalie_x, goalie_y, goal_down[1], player_y):
            self.shot_ball_to_goal(player, goal_down[0], goal_down[1])
        elif self.should_goal_shoot(goalie_x, goalie_y, goal_center[1], player_y):
            self.shot_ball_to_goal(player, goal_center[0], goal_center[1])

    def should_goal_shoot(self, goalie_x, goalie_y, goal, player_y):
        """
        This function determines whether a player should shoot towards the goal based on the position of
        the goalie and the goal.
        
        :param goalie_x: The x-coordinate of the goalie's position on the field
        :param goalie_y: The y-coordinate of the goalie's position on the field
        :param goal: The y-coordinate of the center of the opponent's goal
        :param player_y: The y-coordinate of the player who is considering shooting towards the goal
        :return: a boolean value indicating whether the player should shoot towards the goal or not.
        """
        shoot = True
        # Check if goalie is between the player and the goal
        if (goalie_y - player_y) * (goalie_y - goal) < 0:
            shoot = False

        return shoot

    def pass_ball_to_nearest_player(self, defender_with_ball, all_players):
        """
        This function passes the ball to the nearest player who is at least 20 units away and is either
        a midfielder or an attacker.
        
        :param defender_with_ball: The player object representing the defender who currently has
        possession of the ball
        :param all_players: A list of all the players on the field, including the defender with the ball
        :return: If no player is within the minimum distance, the function returns nothing (i.e., None).
        Otherwise, the function passes the ball to the nearest player and does not return anything.
        """
        # Get the positions of the defender with the ball and all the players on the field
        defender_x, defender_y = defender_with_ball.get_position()
        player_distances = []
        for player in all_players:
            if player != defender_with_ball:
                player_x, player_y = player.get_position()
                distance = math.sqrt(
                    (defender_x - player_x) ** 2 + (defender_y - player_y) ** 2
                )
                if distance >= 20 and (
                    player.role == "midfielder" or player.role == "attacker"
                ):
                    player_distances.append((player, distance))

        if not player_distances:
            return  # No player is within the minimum distance, do not pass the ball

        # Find the player with the smallest distance to the defender with the ball
        nearest_player, nearest_distance = min(player_distances, key=lambda x: x[1])

        # Pass the ball to the nearest player
        #  Add force to the ball to pass it to the nearest player
        ball_x, ball_y = self.ball.get_position()
        nearest_player_x, nearest_player_y = nearest_player.get_position()
        force_x = (nearest_player_x - ball_x) / 10
        force_y = (nearest_player_y - ball_y) / 10

        self.ball.apply_force_to_ball(
            BALL_FORCE_LIMIT * force_x, BALL_FORCE_LIMIT * force_y
        )

    def shot_ball_to_goal(self, goal_target_x, goal_target_y, ball_position):
        """
        The function calculates the force and angle needed to shoot a ball towards a goal target and
        applies it to the ball.
        
        :param goal_target_x: The x-coordinate of the target position of the goal that the player wants
        to shoot the ball towards
        :param goal_target_y: The y-coordinate of the target position of the goal
        :param ball_position: The current position of the ball
        """
        ball_to_goal_vec, ball_to_goal_vec_mag = get_vec(
            [goal_target_x, goal_target_y], ball_position
        )

        # Increase the force of the shot by multiplying with a factor
        ball_force_x = BALL_FORCE_LIMIT * 2 * ball_to_goal_vec[0] / ball_to_goal_vec_mag
        ball_force_y = BALL_FORCE_LIMIT * 2 * ball_to_goal_vec[1] / ball_to_goal_vec_mag

        # Add some randomness to the shot angle
        ball_force_x += random.uniform(-BALL_FORCE_LIMIT / 2, BALL_FORCE_LIMIT / 2)
        ball_force_y += random.uniform(-BALL_FORCE_LIMIT / 2, BALL_FORCE_LIMIT / 2)

        # Decrease the velocity influence on shoot
        self.ball.body.velocity /= 2

        self.ball.apply_force_to_ball(2 * ball_force_x, 2 * ball_force_y)

    def pass_ball_to_player(self, player, closest_teammate):
        """
        This function calculates the force needed to pass a ball to a teammate and applies it to the
        ball.
        
        :param player: The player parameter is an object representing the player who is passing the ball
        :param closest_teammate: The parameter closest_teammate is an object representing the closest
        teammate to the player who is passing the ball
        """
        goal = closest_teammate.get_position()

        ball_pos = self.ball.get_position()
        ball_to_goal_vec, ball_to_goal_vec_mag = get_vec(goal, ball_pos)

        ball_force_x = (
            (BALL_FORCE_LIMIT - 60) * ball_to_goal_vec[0] / ball_to_goal_vec_mag
        )
        ball_force_y = (
            (BALL_FORCE_LIMIT - 60) * ball_to_goal_vec[1] / ball_to_goal_vec_mag
        )

        # decrease the velocity influence on pass
        self.ball.body.velocity /= 10
        self.pass_force_x, self.pass_force_y = ball_force_x, ball_force_y
        self.ball_owner_side = player.side
        self.is_pass = True
        self.ball.apply_force_to_ball(ball_force_x, ball_force_y)

    def ball_closer_to_teammate(self, teammates, ball_x, ball_y):
        """
        This function checks if any teammate is closer than 5 units to the ball's position.
        
        :param teammates: A list of objects representing the teammates of a player in a game
        :param ball_x: The x-coordinate of the ball's current position on the field
        :param ball_y: The parameter ball_y represents the y-coordinate of the ball's position on the
        field
        :return: a boolean value indicating whether there is a teammate closer to the ball than the
        opponent.
        """
        teammate_closer = False
        for player in teammates:
            player_x, player_y = player.get_position()
            distance = math.sqrt((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2)
            if distance < 5:
                teammate_closer = True
                break
        return teammate_closer

    def ball_closer_to_attacker(self, teammates, ball_x, ball_y):
        """
        This function checks if any attacker teammate is closer than 5 units to the ball.
        
        :param teammates: A list of Player objects representing the teammates of the current player
        :param ball_x: The x-coordinate of the ball's current position on the field
        :param ball_y: The parameter ball_y represents the y-coordinate of the ball's current position
        on the field
        :return: a boolean value indicating whether there is an attacker teammate closer to the ball
        than any other teammate within a distance of 5 units.
        """
        attacker_closer = False
        for player in teammates:
            player_x, player_y = player.get_position()
            distance = math.sqrt((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2)
            if distance < 5 and player.role == "attackers":
                attacker_closer = True
                break
        return attacker_closer

    def set_players_goal_target(self, ball_y, goalie_player_y):
        """
        This function sets the target for the player to shoot the ball towards the goal based on the
        position of the ball and the goalie player.
        
        :param ball_y: The y-coordinate of the ball's current position on the game screen
        :param goalie_player_y: The y-coordinate of the player controlling the goalie position in the
        game
        :return: a list containing the coordinates of the target position for the player to shoot the
        ball towards the opponent's goal. The target position is determined based on the position of the
        ball and the goalie player on the field. If the ball is closer to the lower half of the field
        and below the goalie player, the function returns a target position at the lower end of the
        goal. If the ball
        """
        if ball_y > HEIGHT / 2 and ball_y < goalie_player_y:
            lower_target = [WIDTH, (HEIGHT / 2) - (GOAL_SIZE / 2)]
            return lower_target
        elif ball_y < HEIGHT / 2 and ball_y < goalie_player_y:
            upper_target = [WIDTH, (HEIGHT / 2) + (GOAL_SIZE / 2)]
            return upper_target
        else:
            return random.choice(
                [
                    [WIDTH, (HEIGHT / 2) - (GOAL_SIZE / 2)],
                    [WIDTH, (HEIGHT / 2) + (GOAL_SIZE / 2)],
                ]
            )

    def random_action(self):
        return self.action_space.sample()

    def _process_action(self, player, action):
        # Define arrow key mappings for readability
        arrow_keys = {
            0: (0, 0),  # NOOP
            1: (0, 1),  # UP
            2: (1, 0),  # RIGHT
            3: (0, -1),  # DOWN
            4: (-1, 0),  # LEFT
        }

        # Arrow Keys
        if action[0] not in arrow_keys:
            print("Invalid arrow key")
            return

        force_x, force_y = arrow_keys[action[0]]

        if player.side == "left" and self.ball.has_contact_with(player):
            self.ball_owner_side = "left"
            other_teammate_has_ball = True
            self.midfielder_pass = False
            self.defender_pass = False
            self.is_team_B_pass = False
            self.is_upper_goal_team_B = False
            self.is_lower_goal_team_B = False
            if player.role != "attacker":
                self.is_pass = False
                self.is_upper_goal = False
                self.is_lower_goal = False

        if self.ball.has_contact_with(player):
            self.ball_owner_side = "right"
            self.is_pass = False
            self.is_upper_goal = False
            self.is_lower_goal = False
            self.midfielder_pass = False
            self.defender_pass = False
            self.is_team_B_pass = False
            self.is_upper_goal_team_B = False
            self.is_lower_goal_team_B = False

            other_teammate_has_ball = False


        if player.role == "attacker" and player.side == "left":
            player_position = player.get_position()

            ball_position = self.ball.get_position()

            opponent_goalie_player = self.team_B.player_array[0]

            (
                opponent_goalie_player_x,
                opponent_goalie_player_y,
            ) = opponent_goalie_player.get_position()

            player_to_ball_distance = self.get_distance(player_position, ball_position)

            goal_target_x, goal_target_y = self.set_players_goal_target(
                ball_y=ball_position[0], goalie_player_y=opponent_goalie_player_y
            )

            player_to_goal_distance = self.get_distance(
                player_position, (goal_target_x, goal_target_y)
            )

            closest_opponent = self.get_closest_opponent(
                player, self.team_B.player_array
            )
            closest_teammate = self.get_closest_teammate(
                player, self.team_A.player_array
            )

            distance_to_opponent_player = self.get_distance(
                player.get_position(), closest_opponent.get_position()
            )
            other_player_has_ball = False

            if self.is_pass:
                self.ball_owner_side = player.side
                # print(self.pass_force_x, self.pass_force_y)
                self.ball.apply_force_to_ball(self.pass_force_x, self.pass_force_y)

                if self.ball.has_contact_with(player):
                    self.is_pass = False

            if self.is_upper_goal:
                target_x, target_y = [
                    WIDTH,
                    HEIGHT / 2 + (GOAL_SIZE / 2 - 3),
                ]
                self.shot_ball_to_goal(
                    goal_target_x=target_x,
                    goal_target_y=target_y,
                    ball_position=ball_position,
                )

            if self.is_lower_goal:
                target_x, target_y = [
                    WIDTH,
                    HEIGHT / 2 - (GOAL_SIZE / 2 - 3),
                ]
                self.shot_ball_to_goal(
                    goal_target_x=target_x,
                    goal_target_y=target_y,
                    ball_position=ball_position,
                )

            for teammate in self.team_A.player_array:
                if teammate != player and self.ball.has_contact_with(teammate):
                    other_player_has_ball = True
                    break

            if self.ball.has_contact_with(player):
                if (
                    ball_position[1] > opponent_goalie_player_y + 3
                    and player_to_goal_distance < 25
                ):
                    self.is_upper_goal = True
                    # means the ball is above the goalie than shot into ball upward

                elif (
                    ball_position[1] < opponent_goalie_player_y - 3
                    and player_to_goal_distance < 25
                ):
                    self.is_lower_goal = True
                    # means the ball is down by the goalie than shot into downward side

                elif distance_to_opponent_player < 6:
                    # If no one else has the ball and the target goal is far away, move towards the goal

                    self.pass_ball_to_player(player, closest_teammate)

                else:
                    if player.sub_role == "left_forward":
                        upper_target_x, upper_target_y = [
                            WIDTH,
                            HEIGHT / 2 + GOAL_SIZE / 2,
                        ]

                        ball_pos = self.ball.get_position()
                        ball_to_goal_vec, ball_to_goal_vec_mag = get_vec(
                            [upper_target_x, upper_target_y], ball_pos
                        )

                        ball_force_x = (
                            (BALL_FORCE_LIMIT - 20)
                            * ball_to_goal_vec[0]
                            / ball_to_goal_vec_mag
                        )
                        ball_force_y = (
                            (BALL_FORCE_LIMIT - 20)
                            * ball_to_goal_vec[1]
                            / ball_to_goal_vec_mag
                        )
                        if ball_pos[0] + 2 < player_position[0]:
                            self.ball.apply_force_to_ball(ball_force_x, ball_force_y)
                        else:
                            self.ball.apply_force_to_ball(
                                0.8 * ball_force_x, 0.8 * ball_force_y
                            )

                        self._player_towards_goal(
                            player, goal_x=upper_target_x, goal_y=upper_target_y
                        )
                    elif player.sub_role == "striker":
                        middle_target_x, middle_target_y = [
                            WIDTH,
                            HEIGHT / 2,
                        ]
                        ball_pos = self.ball.get_position()
                        ball_to_goal_vec, ball_to_goal_vec_mag = get_vec(
                            [middle_target_x, middle_target_y], ball_pos
                        )

                        ball_force_x = (
                            (BALL_FORCE_LIMIT)
                            * ball_to_goal_vec[0]
                            / ball_to_goal_vec_mag
                        )
                        ball_force_y = (
                            (BALL_FORCE_LIMIT)
                            * ball_to_goal_vec[1]
                            / ball_to_goal_vec_mag
                        )
                        if ball_pos[0] + 2 < player_position[0]:
                            self.ball.apply_force_to_ball(
                                1.5 * ball_force_x, 1.5 * ball_force_y
                            )
                        else:
                            self.ball.apply_force_to_ball(
                                0.8 * ball_force_x, 0.8 * ball_force_y
                            )

                        self._player_towards_goal(
                            player, goal_x=middle_target_x, goal_y=middle_target_y
                        )
                    else:
                        lower_target_x, lower_target_y = [
                            WIDTH,
                            HEIGHT / 2 - GOAL_SIZE / 2,
                        ]
                        ball_pos = self.ball.get_position()
                        ball_to_goal_vec, ball_to_goal_vec_mag = get_vec(
                            [lower_target_x, lower_target_y], ball_pos
                        )

                        ball_force_x = (
                            (BALL_FORCE_LIMIT - 20)
                            * ball_to_goal_vec[0]
                            / ball_to_goal_vec_mag
                        )
                        ball_force_y = (
                            (BALL_FORCE_LIMIT - 20)
                            * ball_to_goal_vec[1]
                            / ball_to_goal_vec_mag
                        )
                        if ball_pos[0] + 2 < player_position[0]:
                            self.ball.apply_force_to_ball(ball_force_x, ball_force_y)
                        else:
                            self.ball.apply_force_to_ball(
                                0.8 * ball_force_x, 0.8 * ball_force_y
                            )

                        self._player_towards_goal(
                            player, goal_x=lower_target_x, goal_y=lower_target_y
                        )

            elif not other_player_has_ball:
                if (
                    ball_position[1] >= HEIGHT / 2 + GOAL_SIZE / 2
                    and player.sub_role == "left_forward"
                    or player_to_ball_distance < 8
                ):
                    self._player_towards_ball(
                        player, ball_position[0], ball_position[1]
                    )
                elif (
                    ball_position[1] <= HEIGHT / 2 - GOAL_SIZE / 2
                    and player.sub_role == "right_forward"
                    or player_to_ball_distance < 8
                ):
                    self._player_towards_ball(
                        player, ball_position[0], ball_position[1]
                    )
                elif player.sub_role == "striker":
                    self._player_towards_ball(
                        player, ball_position[0], ball_position[1]
                    )
            elif other_player_has_ball:

                if player.sub_role == "left_forward":
                    upper_target_x, upper_target_y = [
                        WIDTH - 15,
                        HEIGHT / 2 + GOAL_SIZE / 2 + 5,
                    ]
                    if player_position[0] > ball_position[0] + 10:
                        self._player_towards_goal(
                            player, goal_x=-upper_target_x, goal_y=-upper_target_y
                        )
                    else:
                        self._player_towards_goal(
                            player, goal_x=upper_target_x, goal_y=upper_target_y
                        )
                elif player.sub_role == "striker":
                    middle_target_x, middle_target_y = [
                        WIDTH - 15,
                        HEIGHT / 2,
                    ]
                    if player_position[0] > ball_position[0] + 10:
                        self._player_towards_goal(
                            player, goal_x=-middle_target_x, goal_y=-middle_target_y
                        )
                    else:
                        self._player_towards_goal(
                            player, goal_x=middle_target_x, goal_y=middle_target_y
                        )

                else:
                    lower_target_x, lower_target_y = [
                        WIDTH - 15,
                        HEIGHT / 2 - GOAL_SIZE / 2 + 5,
                    ]
                    if player_position[0] > ball_position[0] + 10:
                        self._player_towards_goal(
                            player, goal_x=-lower_target_x, goal_y=-lower_target_y
                        )
                    else:
                        self._player_towards_goal(
                            player, goal_x=lower_target_x, goal_y=lower_target_y
                        )



        # right side attackers strategy
        if player.role == "attacker" and player.side == "right":

            player_position = player.get_position()
            ball_position = self.ball.get_position()
            opponent_goalie_player = self.team_A.player_array[0]

            (
                opponent_goalie_player_x,
                opponent_goalie_player_y,
            ) = opponent_goalie_player.get_position()

            player_to_ball_distance = self.get_distance(player_position, ball_position)

            goal_target_x, goal_target_y = 0, HEIGHT / 2

            player_to_goal_distance = self.get_distance(
                player_position, (goal_target_x, goal_target_y)
            )

            closest_opponent = self.get_closest_opponent(
                player, self.team_A.player_array
            )

            closest_teammate = self.get_closest_teammate(
                player, self.team_B.player_array
            )

            distance_to_opponent_player = self.get_distance(
                player.get_position(), closest_opponent.get_position()
            )

            other_player_has_ball = False

            if self.is_team_B_pass:
                self.ball_owner_side = player.side
                self.ball.apply_force_to_ball(self.pass_force_x, self.pass_force_y)

                if self.ball.has_contact_with(player):
                    self.is_team_B_pass = False

            if self.is_upper_goal_team_B:
                target_x, target_y = [
                    0,
                    HEIGHT / 2 + (GOAL_SIZE / 2 - 3),
                ]
                self.shot_ball_to_goal(
                    goal_target_x=target_x,
                    goal_target_y=target_y,
                    ball_position=ball_position,
                )

            if self.is_lower_goal_team_B:
                target_x, target_y = [
                    0,
                    HEIGHT / 2 - (GOAL_SIZE / 2 - 3),
                ]
                self.shot_ball_to_goal(
                    goal_target_x=target_x,
                    goal_target_y=target_y,
                    ball_position=ball_position,
                )

            for teammate in self.team_B.player_array:
                if teammate != player and self.ball.has_contact_with(teammate):
                    other_player_has_ball = True
                    break

            if self.ball.has_contact_with(player):
                if (
                    ball_position[1] > opponent_goalie_player_y + 3
                    and player_to_goal_distance < 25
                ):
                    self.is_upper_goal_team_B = True
                    # means the ball is above the goalie than shot into ball upward

                elif (
                    ball_position[1] < opponent_goalie_player_y - 3
                    and player_to_goal_distance < 25
                ):
                    self.is_lower_goal_team_B = True
                    # means the ball is down by the goalie than shot into downward side

                elif distance_to_opponent_player < 6:
                    # If no one else has the ball and the target goal is far away, move towards the goal

                    goal = closest_teammate.get_position()

                    ball_pos = self.ball.get_position()
                    ball_to_goal_vec, ball_to_goal_vec_mag = get_vec(goal, ball_pos)

                    ball_force_x = (
                        (BALL_FORCE_LIMIT - 60) * ball_to_goal_vec[0] / ball_to_goal_vec_mag
                    )
                    ball_force_y = (
                        (BALL_FORCE_LIMIT - 60) * ball_to_goal_vec[1] / ball_to_goal_vec_mag
                    )

                    # decrease the velocity influence on pass
                    self.ball.body.velocity /= 10
                    self.pass_force_x, self.pass_force_y = ball_force_x, ball_force_y
                    self.ball_owner_side = player.side
                    self.is_team_B_pass = True
                    self.ball.apply_force_to_ball(ball_force_x, ball_force_y)

                else:
                    if player.sub_role == "left_forward":
                        upper_target_x, upper_target_y = [
                            0,
                            HEIGHT / 2 + GOAL_SIZE / 2,
                        ]

                        ball_pos = self.ball.get_position()
                        ball_to_goal_vec, ball_to_goal_vec_mag = get_vec(
                            [upper_target_x, upper_target_y], ball_pos
                        )

                        ball_force_x = (
                            (BALL_FORCE_LIMIT - 20)
                            * ball_to_goal_vec[0]
                            / ball_to_goal_vec_mag
                        )
                        ball_force_y = (
                            (BALL_FORCE_LIMIT - 20)
                            * ball_to_goal_vec[1]
                            / ball_to_goal_vec_mag
                        )
                        # means the ball is behind the player for team_B perspective
                        if ball_pos[0] + 2 > player_position[0]:
                            # apply more force to get the ball ahead
                            self.ball.apply_force_to_ball(ball_force_x, ball_force_y)

                        # means the ball is ahead
                        else:
                            # apply force to carry the ball to goal
                            self.ball.apply_force_to_ball(
                                0.8 * ball_force_x, 0.8 * ball_force_y
                            )

                        # move player towards the upper goal target
                        self._player_towards_goal(
                            player, goal_x=upper_target_x, goal_y=upper_target_y
                        )

                    elif player.sub_role == "striker":
                        middle_target_x, middle_target_y = [
                            0,
                            HEIGHT / 2,
                        ]
                        ball_pos = self.ball.get_position()
                        ball_to_goal_vec, ball_to_goal_vec_mag = get_vec(
                            [middle_target_x, middle_target_y], ball_pos
                        )

                        ball_force_x = (
                            (BALL_FORCE_LIMIT)
                            * ball_to_goal_vec[0]
                            / ball_to_goal_vec_mag
                        )
                        ball_force_y = (
                            (BALL_FORCE_LIMIT)
                            * ball_to_goal_vec[1]
                            / ball_to_goal_vec_mag
                        )
                        if ball_pos[0] + 2 > player_position[0]:
                            self.ball.apply_force_to_ball(
                                1.5 * ball_force_x, 1.5 * ball_force_y
                            )
                        else:
                            self.ball.apply_force_to_ball(
                                0.8 * ball_force_x, 0.8 * ball_force_y
                            )

                        self._player_towards_goal(
                            player, goal_x=middle_target_x, goal_y=middle_target_y
                        )
                    else:
                        lower_target_x, lower_target_y = [
                            0,
                            HEIGHT / 2 - GOAL_SIZE / 2,
                        ]
                        ball_pos = self.ball.get_position()
                        ball_to_goal_vec, ball_to_goal_vec_mag = get_vec(
                            [lower_target_x, lower_target_y], ball_pos
                        )

                        ball_force_x = (
                            (BALL_FORCE_LIMIT - 20)
                            * ball_to_goal_vec[0]
                            / ball_to_goal_vec_mag
                        )
                        ball_force_y = (
                            (BALL_FORCE_LIMIT - 20)
                            * ball_to_goal_vec[1]
                            / ball_to_goal_vec_mag
                        )
                        if ball_pos[0] + 2 > player_position[0]:
                            self.ball.apply_force_to_ball(ball_force_x, ball_force_y)
                        else:
                            self.ball.apply_force_to_ball(
                                0.8 * ball_force_x, 0.8 * ball_force_y
                            )

                        self._player_towards_goal(
                            player, goal_x=lower_target_x, goal_y=lower_target_y
                        )

            elif not other_player_has_ball:
                if (
                    ball_position[1] >= HEIGHT / 2 + GOAL_SIZE / 2
                    and player.sub_role == "left_forward"
                    or player_to_ball_distance < 8
                ):
                    self._player_towards_ball(
                        player, ball_position[0], ball_position[1]
                    )
                elif (
                    ball_position[1] <= HEIGHT / 2 - GOAL_SIZE / 2
                    and player.sub_role == "right_forward"
                    or player_to_ball_distance < 8
                ):
                    self._player_towards_ball(
                        player, ball_position[0], ball_position[1]
                    )
                elif player.sub_role == "striker":
                    self._player_towards_ball(
                        player, ball_position[0], ball_position[1]
                    )
            elif other_player_has_ball:

                if player.sub_role == "left_forward":
                    upper_target_x, upper_target_y = [
                        0 + 15,
                        HEIGHT / 2 + GOAL_SIZE / 2 + 5,
                    ]
                    # means player is ahead and ball is behind so move backword to positions
                    if player_position[0] < ball_position[0] + 8:
                        self._player_towards_goal(
                            player, goal_x=-upper_target_x, goal_y=-upper_target_y
                        )
                    else:
                        self._player_towards_goal(
                            player, goal_x=upper_target_x, goal_y=upper_target_y
                        )
                elif player.sub_role == "striker":
                    middle_target_x, middle_target_y = [
                        0 + 15,
                        HEIGHT / 2,
                    ]
                    if player_position[0] < ball_position[0] + 8:
                        self._player_towards_goal(
                            player, goal_x=-middle_target_x, goal_y=-middle_target_y
                        )
                    else:
                        self._player_towards_goal(
                            player, goal_x=middle_target_x, goal_y=middle_target_y
                        )

                else:
                    lower_target_x, lower_target_y = [
                        0 + 15,
                        HEIGHT / 2 - GOAL_SIZE / 2 + 5,
                    ]
                    if player_position[0] > ball_position[0] + 8:
                        self._player_towards_goal(
                            player, goal_x=-lower_target_x, goal_y=-lower_target_y
                        )
                    else:
                        self._player_towards_goal(
                            player, goal_x=lower_target_x, goal_y=lower_target_y
                        )




        if player.role == "goalie" :
            ball_y = self.ball.get_position()[1]
            goalie_x, goalie_y = player.get_position()
            force_x, force_y = 0, 0

            # Move goalie up or down based on ball position
            if ball_y > goalie_y:
                force_y = 1
            elif ball_y < goalie_y:
                force_y = -1

            # Restrict goalie to penalty box
            if goalie_y < HEIGHT / 2 - GOAL_SIZE:
                if force_y == -1:
                    force_y = 0
                    # Add force to bring goalie back inside penalty box
                    force_y += 1
            elif goalie_y > HEIGHT / 2 + GOAL_SIZE:
                if force_y == 1:
                    force_y = 0
                    # Add force to bring goalie back inside penalty box
                    force_y -= 1

            if self.ball.has_contact_with(player):
                if player.side == "left":
                    goal = [self.width, self.height / 2]
                elif player.side == "right":
                    goal = [0, self.height / 2]
                else:
                    print("invalid side")

                ball_pos = self.ball.get_position()
                ball_to_goal_vec, ball_to_goal_vec_mag = get_vec(goal, ball_pos)

                ball_force_x = (
                    BALL_FORCE_LIMIT * ball_to_goal_vec[0] / ball_to_goal_vec_mag
                )
                ball_force_y = (
                    BALL_FORCE_LIMIT * ball_to_goal_vec[1] / ball_to_goal_vec_mag
                )
                self.ball_owner_side = player.side
                self.ball.apply_force_to_ball(ball_force_x, ball_force_y)

            player.apply_force_to_player(
                PLAYER_FORCE_LIMIT * force_x, PLAYER_FORCE_LIMIT * force_y
            )


        if player.role == "defender" and player.side == "right":
            ball_x, ball_y = self.ball.get_position()
            defender_x, defender_y = player.get_position()

            initial_pos = self.team_B.get_defender_initial_position()

            # Calculate the distance between the ball and player
            distance = math.sqrt(
                (ball_x - defender_x) ** 2 + (ball_y - defender_y) ** 2
            )

            closest_teammate = self.get_closest_teammate(
                player, self.team_A.player_array
            )

            if self.defender_pass:
                self.ball.apply_force_to_ball(self.pass_force_x, self.pass_force_y)
                if self.ball.has_contact_with(player):
                    self.defender_pass = False

            # Move defender forward in the x direction if the ball is on the other side of the center line
            if self.ball.has_contact_with(player):
                player_pass = closest_teammate.get_position()
                ball_pos = self.ball.get_position()
                ball_to_goal_vec, ball_to_goal_vec_mag = get_vec(player_pass, ball_pos)

                ball_force_x = (
                    (BALL_FORCE_LIMIT - 60) * ball_to_goal_vec[0] / ball_to_goal_vec_mag
                )
                ball_force_y = (
                    (BALL_FORCE_LIMIT - 60) * ball_to_goal_vec[1] / ball_to_goal_vec_mag
                )

                # decrease the velocity influence on pass
                self.ball.body.velocity /= 10
                self.pass_force_x, self.pass_force_y = ball_force_x, ball_force_y
                self.ball_owner_side = player.side
                self.defender_pass = True
                self.ball.apply_force_to_ball(ball_force_x, ball_force_y)

            else:
                force_x, force_y = 0, 0
                if distance < 16 and not self.should_move_towards_ball(
                    self.team_B.player_array
                ):
                    if not self.ball_closer_to_teammate(
                        self.team_B.player_array, ball_x, ball_y
                    ):
                        self._player_towards_ball(player, ball_x, ball_y)
                elif ball_x < WIDTH / 2 and defender_x > ((WIDTH / 2) + 20):
                    force_x = -1
                    player.set_velocity(-PLAYER_MAX_VELOCITY, 0)
                elif ball_x > WIDTH / 2 and defender_x < initial_pos:
                    player.set_velocity(PLAYER_MAX_VELOCITY, 0)
                    force_x = 1
                else:
                    force_x = 0
                    player.set_velocity(0, 0)

            player.apply_force_to_player(
                PLAYER_FORCE_LIMIT * force_x, PLAYER_FORCE_LIMIT * force_y
            )


        if player.role == "defender" and player.side == "left":
            ball_x, ball_y = self.ball.get_position()
            defender_x, defender_y = player.get_position()

            initial_pos = self.team_A.get_defender_initial_position()

            # Calculate the distance between the ball and player
            distance = math.sqrt(
                (ball_x - defender_x) ** 2 + (ball_y - defender_y) ** 2
            )
            mid = self.team_B.player_array[5].get_position()

            closest_teammate = self.get_closest_teammate_attacker(
                player, self.team_A.player_array
            )

            if self.defender_pass:
                self.ball.apply_force_to_ball(self.pass_force_x, self.pass_force_y)
                if self.ball.has_contact_with(player):
                    self.defender_pass = False

            # Move defender forward in the x direction if the ball is on the other side of the center line
            if self.ball.has_contact_with(player):
                player_pass = closest_teammate.get_position()
                ball_pos = self.ball.get_position()
                ball_to_goal_vec, ball_to_goal_vec_mag = get_vec(player_pass, ball_pos)

                ball_force_x = (
                    (BALL_FORCE_LIMIT - 60) * ball_to_goal_vec[0] / ball_to_goal_vec_mag
                )
                ball_force_y = (
                    (BALL_FORCE_LIMIT - 60) * ball_to_goal_vec[1] / ball_to_goal_vec_mag
                )

                # decrease the velocity influence on pass
                self.ball.body.velocity /= 10
                self.pass_force_x, self.pass_force_y = ball_force_x, ball_force_y
                self.ball_owner_side = player.side
                self.defender_pass = True
                self.ball.apply_force_to_ball(ball_force_x, ball_force_y)
            else:
                force_x, force_y = 0, 0
                if distance < 15 and not self.should_move_towards_ball(
                    self.team_A.player_array
                ):
                    self._player_towards_ball(player, ball_x, ball_y)
                elif ball_x > WIDTH / 2 and defender_x < ((WIDTH / 2) - 20):
                    force_x = 1
                    player.set_velocity(PLAYER_MAX_VELOCITY, 0)
                elif (
                    ball_x < WIDTH / 2 or defender_x > mid[0]
                ) and defender_x > initial_pos:
                    player.set_velocity(-PLAYER_MAX_VELOCITY, 0)
                    force_x = -1
                else:
                    force_x = 0
                    player.set_velocity(0, 0)

            player.apply_force_to_player(
                PLAYER_FORCE_LIMIT * force_x, PLAYER_FORCE_LIMIT * force_y
            )

        if player.role == "midfielder" and player.side == "left":
            ball_x, ball_y = self.ball.get_position()
            midfielder_x, midfielder_y = player.get_position()

            goal = [self.width, self.height / 2]
            initial_pos = 26.25

            goal_distance = math.sqrt(
                (goal[0] - midfielder_x) ** 2 + (goal[1] - midfielder_y) ** 2
            )

            closest_teammate = self.get_closest_teammate_attacker(
                player, self.team_A.player_array
            )

            # Calculate the distance between the ball and player
            distance = math.sqrt(
                (ball_x - midfielder_x) ** 2 + (ball_y - midfielder_y) ** 2
            )
            goalie = self.team_B.player_array[0]

            
            if self.midfielder_pass:
                self.ball.apply_force_to_ball(self.pass_force_x, self.pass_force_y)
                if self.ball.has_contact_with(player):
                    self.midfielder_pass = False

            if self.ball.has_contact_with(goalie) and midfielder_x > (
                (3 * WIDTH / 4) + 20
            ):
                force_x = -1
                player.set_velocity(-PLAYER_MAX_VELOCITY, 0)

            elif self.ball.has_contact_with(player):
                # mid fielder to move towards goal with ball
                self._player_towards_ball(player, goal[0], goal[1])

                # pass the ball to a teammate if opponent is closer to stealing the ball
                if self.opponent_moving_towards_ball(
                    self.team_B.player_array, ball_x, ball_y
                ) or midfielder_x > ((3 * WIDTH / 4) + 10):
                    player_pass = closest_teammate.get_position()
                    ball_pos = self.ball.get_position()
                    ball_to_goal_vec, ball_to_goal_vec_mag = get_vec(
                        player_pass, ball_pos
                    )

                    ball_force_x = (
                        (BALL_FORCE_LIMIT - 60)
                        * ball_to_goal_vec[0]
                        / ball_to_goal_vec_mag
                    )
                    ball_force_y = (
                        (BALL_FORCE_LIMIT - 60)
                        * ball_to_goal_vec[1]
                        / ball_to_goal_vec_mag
                    )

                    # decrease the velocity influence on pass
                    self.ball.body.velocity /= 10
                    self.pass_force_x, self.pass_force_y = ball_force_x, ball_force_y
                    self.ball_owner_side = player.side
                    self.midfielder_pass = True
                    self.ball.apply_force_to_ball(ball_force_x, ball_force_y)

                self._ball_move_with_player(player)
            else:
                force_x, force_y = 0, 0
                # player to steal the ball if the ball is in its range and is with an opponent
                if distance < 10 and not self.should_move_towards_ball(
                    self.team_A.player_array
                ):
                    if not (
                        (
                            midfielder_x > (3 * WIDTH / 4)
                            and not self.ball_closer_to_attacker(
                                self.team_A.player_array, ball_x, ball_y
                            )
                        )
                    ):
                        self._player_towards_ball(player, ball_x, ball_y)
                # player to move back to their if opponent is advancing towards your goal
                elif self.is_opponent_with_ball(self.team_B.player_array):
                    if midfielder_x >= (initial_pos + 10) and (
                        ball_x < ((3 * WIDTH / 4) - 10) or midfielder_x > ball_x
                    ):
                        force_x = -1
                        player.set_velocity(-PLAYER_MAX_VELOCITY, 0)
                # player to move forward if ball is ahead of it or ahead of center line
                elif (
                    ball_x > WIDTH / 2
                    and midfielder_x < ((WIDTH / 2) + 10)
                    and midfielder_x < ball_x
                ):
                    force_x = 1
                    player.set_velocity(PLAYER_MAX_VELOCITY, 0)
                elif (
                    ball_x < WIDTH / 2
                    and midfielder_x >= (initial_pos + 10)
                    or midfielder_x > ball_x
                ):
                    force_x = -1
                    player.set_velocity(-PLAYER_MAX_VELOCITY, 0)
                elif ball_x > ((3 * WIDTH / 4) - 10) and midfielder_x < ball_x:
                    if (
                        player.sub_role == "center_forward"
                        and ball_y > ((HEIGHT / 4) + 10)
                        and ball_y < ((3 * HEIGHT / 4) - 10)
                    ):
                        self._player_towards_ball(player, ball_x, ball_y)
                    elif player.sub_role == "center_up" and ball_y < ((HEIGHT / 4) + 5):
                        self._player_towards_ball(player, ball_x, ball_y)
                    elif player.sub_role == "center_down" and ball_y > (
                        (3 * HEIGHT / 4) - 5
                    ):
                        self._player_towards_ball(player, ball_x, ball_y)
                    else:
                        force_x = 0
                        player.set_velocity(0, 0)
                else:
                    force_x = 0
                    player.set_velocity(0, 0)

            player.apply_force_to_player(
                PLAYER_FORCE_LIMIT * force_x, PLAYER_FORCE_LIMIT * force_y
            )

        if player.role == "midfielder" and player.side == "right":
            ball_x, ball_y = self.ball.get_position()
            midfielder_x, midfielder_y = player.get_position()

            goal = [0, self.height / 2]
            initial_pos = 78.75

            goal_distance = math.sqrt(
                (goal[0] - midfielder_x) ** 2 + (goal[1] - midfielder_y) ** 2
            )

            closest_teammate = self.get_closest_teammate(
                player, self.team_B.player_array
            )

            # Calculate the distance between the ball and player
            distance = math.sqrt(
                (ball_x - midfielder_x) ** 2 + (ball_y - midfielder_y) ** 2
            )
            goalie = self.team_A.player_array[0]

            if self.midfielder_pass:
                self.ball.apply_force_to_ball(self.pass_force_x, self.pass_force_y)
                if self.ball.has_contact_with(player):
                    self.midfielder_pass = False

            if self.ball.has_contact_with(goalie) and midfielder_x < (
                (1 * self.width / 4) - 20
            ):
                force_x = 1
                player.set_velocity(PLAYER_MAX_VELOCITY, 0)

            elif self.ball.has_contact_with(player):
                # midfielder to move towards goal with ball
                self._player_towards_ball(player, goal[0], goal[1])

                # pass the ball to a teammate if opponent is closer to stealing the ball
                if self.opponent_moving_towards_ball(
                    self.team_A.player_array, ball_x, ball_y
                ) or midfielder_x < ((1 * self.width / 4) - 10):
                    player_pass = closest_teammate.get_position()
                    ball_pos = self.ball.get_position()
                    ball_to_goal_vec, ball_to_goal_vec_mag = get_vec(
                        player_pass, ball_pos
                    )

                    ball_force_x = (
                        (BALL_FORCE_LIMIT - 60)
                        * ball_to_goal_vec[0]
                        / ball_to_goal_vec_mag
                    )
                    ball_force_y = (
                        (BALL_FORCE_LIMIT - 60)
                        * ball_to_goal_vec[1]
                        / ball_to_goal_vec_mag
                    )

                    # decrease the velocity influence on pass
                    self.ball.body.velocity /= 10
                    self.pass_force_x, self.pass_force_y = ball_force_x, ball_force_y
                    self.ball_owner_side = player.side
                    self.midfielder_pass = True
                    self.ball.apply_force_to_ball(ball_force_x, ball_force_y)

                self._ball_move_with_player(player)
            else:
                force_x, force_y = 0, 0
                # player to steal the ball if the ball is in its range and is with an opponent
                if distance < 10 and not self.should_move_towards_ball(
                    self.team_B.player_array
                ):
                    if not (
                        (
                            midfielder_x < (1 * self.width / 4)
                            and not self.ball_closer_to_attacker(
                                self.team_B.player_array, ball_x, ball_y
                            )
                        )
                    ):
                        self._player_towards_ball(player, ball_x, ball_y)
                # player to move back to their if opponent is advancing towards your goal
                elif self.is_opponent_with_ball(self.team_A.player_array):
                    if midfielder_x <= (WIDTH - initial_pos - 10) and (
                        ball_x > (WIDTH / 4) + 10 or midfielder_x < ball_x
                    ):
                        force_x = 1
                        player.set_velocity(PLAYER_MAX_VELOCITY, 0)
                # player to move forward if ball is ahead of it or ahead of center line
                elif (
                    ball_x < WIDTH / 2
                    and midfielder_x > (WIDTH / 2 - 10)
                    and midfielder_x > ball_x
                ):
                    force_x = -1
                    player.set_velocity(-PLAYER_MAX_VELOCITY, 0)
                elif (
                    ball_x > WIDTH / 2
                    and midfielder_x <= (WIDTH - initial_pos - 10)
                    or midfielder_x < ball_x
                ):
                    force_x = 1
                    player.set_velocity(PLAYER_MAX_VELOCITY, 0)
                elif ball_x < (WIDTH / 4) + 10 and midfielder_x > ball_x:
                    if (
                        player.sub_role == "center_forward"
                        and ball_y > ((HEIGHT / 4) + 10)
                        and ball_y < ((3 * HEIGHT / 4) - 10)
                    ):
                        self._player_towards_ball(player, ball_x, ball_y)
                    elif player.sub_role == "center_up" and ball_y < ((HEIGHT / 4) + 5):
                        self._player_towards_ball(player, ball_x, ball_y)
                    elif player.sub_role == "center_down" and ball_y > (
                        (3 * HEIGHT / 4) - 5
                    ):
                        self._player_towards_ball(player, ball_x, ball_y)
                    else:
                        force_x = 0
                        player.set_velocity(0, 0)
                else:
                    force_x = 0
                    player.set_velocity(0, 0)

                player.apply_force_to_player(
                    PLAYER_FORCE_LIMIT * force_x, PLAYER_FORCE_LIMIT * force_y
                )

    # action space
    # 1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
    # 2) Action Keys: Discrete 5  - noop[0], dash[1], shoot[2], press[3], pass[4] - params: min: 0, max: 4
    def step(self, left_player_action):
        """
        The function takes in the action of the left player and returns the observation, reward, done and
        info

        :param left_player_action: The action of the left player
        :return: The observation, reward, done, and info are being returned.
        """
        right_player_action = np.reshape(self.random_action(), (-1, 2))
        left_player_action = np.reshape(left_player_action, (-1, 2))
        init_distance_arr = self._ball_to_team_distance_arr(self.team_A)
        ball_init = self.ball.get_position()
        done = False
        reward = 0

        action_arr = np.concatenate((left_player_action, right_player_action))

        for player, action in zip(self.player_arr, action_arr):
            self._process_action(player, action)
            if self.ball.has_contact_with(player):
                self.ball_owner_side = player.side
            else:
                pass

        out = self.check_and_fix_out_bounds()

        self.space.step(TIME_STEP)
        self.observation = self._get_observation()

        if not out:
            ball_after = self.ball.get_position()
            reward += self.get_team_reward(init_distance_arr, self.team_A)
            reward += self.get_ball_reward(ball_init, ball_after)

        if self.ball_contact_goal():
            bx, _ = self.ball.get_position()
            goal = ""
            if bx > self.width - 10:
                self.score_team_a += 1  # increment score for Team A
                reward += 30
                goal = "team_a"
                # give 30 reward for scoring a goal against opponent team

            elif bx < self.width - 2:
                self.score_team_b += 1  # increment score for Team B
                reward -= 30
                goal = "team_b"
            self._position_to_initial(goal)
            self.ball_owner_side = random.choice(["left", "right"])
        self.current_time += TIME_STEP

        if self.current_time > self.total_time:
            done = True

        info = {"score_team_a": self.score_team_a, "score_team_b": self.score_team_b}

        return self.observation, reward, done, info

    def get_team_reward(self, init_distance_arr, team):
        """
        The function returns the maximum difference between the initial distance of the ball to the team
        and the final distance of the ball to the team

        :param init_distance_arr: the distance between the ball and each player at the beginning of the
        episode
        :param team: the team that the player is on
        :return: The maximum difference between the initial distance and the final distance.
        """

        after_distance_arr = self._ball_to_team_distance_arr(team)

        difference_arr = init_distance_arr - after_distance_arr

        run_to_ball_reward_coefficient = 10

        if self.number_of_player == 5:
            # check if any player has passed the ball to their own teammate
            for player, after_distance in zip(
                self.player_arr[:3], after_distance_arr[:3]
            ):
                if after_distance < np.min(after_distance_arr[:3]):
                    reward += 10  # give 10 reward to player who passed the ball to their own teammate

            return (
                np.max([difference_arr[3], difference_arr[4]])
                * run_to_ball_reward_coefficient
            )
        else:
            # check if any player has passed the ball to their own teammate
            for player, after_distance in zip(self.player_arr, after_distance_arr):
                if after_distance < np.min(after_distance_arr):
                    reward += 10  # give 10 reward to player who passed the ball to their own teammate

            return np.max(difference_arr) * run_to_ball_reward_coefficient

    def get_ball_reward(self, ball_init, ball_after):
        """
        If the ball is shoot towards the goal, give a reward. If the ball is passed to a teammate, give a
        reward. If the ball is neither shot nor passed, return zero reward

        :param ball_init: the initial position of the ball
        :param ball_after: the position of the ball after the action
        :return: The reward is being returned.
        """

        ball_to_goal_reward_coefficient = 10

        goal = [self.width, self.height / 2]

        _, ball_a_to_goal = get_vec(ball_after, goal)
        _, ball_i_to_goal = get_vec(ball_init, goal)
        # calculate the distance from the ball to the center of the opponent's goal
        ball_to_goal_dist = np.sqrt(
            (ball_after[0] - goal[0]) ** 2 + (ball_after[1] - goal[1]) ** 2
        )

        # calculate the distance change of the ball to the center of the opponent's goal
        ball_dist_change = (
            np.sqrt((ball_init[0] - goal[0]) ** 2 + (ball_init[1] - goal[1]) ** 2)
            - ball_to_goal_dist
        )

        # if the ball is shot towards the goal, give a reward
        if ball_a_to_goal > ball_i_to_goal:
            return ball_dist_change * ball_to_goal_reward_coefficient + 20
        # if the ball is passed to a teammate, give a reward
        elif self.ball_owner_side == "left" and ball_init[0] < ball_after[0]:
            return 10
        elif self.ball_owner_side == "right" and ball_init[0] > ball_after[0]:
            return -10
        # if the ball is neither shot nor passed, return zero reward
        else:
            return 0

    def _ball_to_team_distance_arr(self, team):
        distance_arr = []
        bx, by = self.ball.get_position()
        for player in team.player_array:
            px, py = player.get_position()
            distance_arr.append(math.sqrt((px - bx) ** 2 + (py - by) ** 2))
        return np.array(distance_arr)