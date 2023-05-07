import pymunk
from pymunk.vec2d import Vec2d
from .player import Player
import numpy as np
import random
import math


class Team:
    def __init__(
        self,
        space,
        width,
        height,
        player_weight,
        player_max_velocity,
        color=(1, 0, 0, 1),
        side="left",
        player_number=2,
        elasiticity=0.2,
    ):
        self.space = space
        self.width = width
        self.height = height
        self.side = side

        self.player_number = player_number

        self._create_pos_array(player_number, side, width, height)
        self._create_color_array(color, player_number)

        self.player_array = []
        for x, y, c in zip(self.x_pos_array, self.y_pos_array, self.color_array):
            self.player_array.append(
                Player(
                    self.space,
                    x,
                    y,
                    mass=player_weight,
                    color=c,
                    max_velocity=player_max_velocity,
                    elasticity=elasiticity,
                    side=side,
                )
            )

        self.set_players_role()

    def set_players_role(self):
        for index, player in enumerate(self.player_array):
            if index == 0:
                player.role = "goalie"

            elif index >= 1 and index <= 4:
                player.role = "defender"

            elif index >= 5 and index <= 7:
                player.role = "midfielder"
                if index == 5:
                    player.sub_role = "center_forward"
                elif index == 6:
                    player.sub_role = "center_down"
                elif index == 7:
                    player.sub_role = "center_up"

            elif index >= 8:
                player.role = "attacker"
                if index == 8:
                    player.sub_role = "right_forward"
                elif index == 9:
                    player.sub_role = "striker"
                else:
                    player.sub_role = "left_forward"

    # only implemented with red and blue
    def _create_color_array(self, color, player_number):
        if player_number == 1:
            self.color_array = [color]
        else:
            if color == (1, 0, 0, 1):  # if color is red
                self.color_array = [
                    (1, 0, 0, 1)
                ] * player_number  # set red color for left team
                self.color_array[0] = (1, 1, 0, 1)  # set goalie color to yellow
            elif color == (0, 0, 1, 1):  # if color is blue
                self.color_array = [
                    (0, 0, 1, 1)
                ] * player_number  # set blue color for right team
                self.color_array[0] = (1, 1, 1, 1)  # set goalie color to yellow
            else:
                self.color_array = [color] * player_number

    def _create_pos_array(self, player_number, side, width, height):
        # implement for 3 players and fewer now
        if player_number <= 3:
            # get x position for each player
            if side == "left":
                self.x_pos_array = [width * 0.25] * player_number
            elif side == "right":
                self.x_pos_array = [width * 0.75] * player_number
            else:
                print("invalid side")
            # get y position for each player
            y_increment = height / (player_number + 1)
            self.y_pos_array = []
            for i in range(player_number):
                self.y_pos_array.append(y_increment * (i + 1))

        elif player_number <= 6:
            # get x position for each player
            if side == "left":
                self.x_pos_array = [width * 1 / 6] * 3 + [width * 2 / 6] * (
                    player_number - 3
                )
            elif side == "right":
                self.x_pos_array = [width * 5 / 6] * 3 + [width * 4 / 6] * (
                    player_number - 3
                )
            else:
                print("invalid side")
            # get y position for each player
            y_increment = height / (3 + 1)
            self.y_pos_array = []
            for i in range(3):
                self.y_pos_array.append(y_increment * (i + 1))
            y_increment = height / (player_number - 3 + 1)
            for i in range(player_number - 3):
                self.y_pos_array.append(y_increment * (i + 1))

        # 7 player might look wired
        elif player_number <= 10:
            # get x position for each player
            if side == "left":
                self.x_pos_array = (
                    [width * 1 / 8] * 4
                    + [width * 2 / 8] * 3
                    + [width * 3 / 8] * (player_number - 7)
                )

            elif side == "right":
                self.x_pos_array = (
                    [width * 7 / 8] * 4
                    + [width * 6 / 8] * 3
                    + [width * 5 / 8] * (player_number - 7)
                )
            else:
                print("invalid side")
            # get y position for each player
            y_increment = height / (4 + 1)
            self.y_pos_array = []
            for i in range(4):
                self.y_pos_array.append(y_increment * (i + 1))

            y_increment = height / (3 + 1)
            for i in range(3):
                self.y_pos_array.append(y_increment * (i + 1))

            y_increment = height / (player_number - 7 + 1)
            for i in range(player_number - 7):
                self.y_pos_array.append(y_increment * (i + 1))

        elif player_number == 11:
            # get x position for each player
            if side == "left":
                self.x_pos_array = (
                    [width * 0.3 / 8] * 1
                    + [width * 1 / 8] * 4
                    + [width * 2 / 8] * 3
                    + [width * 3 / 8] * (player_number - 7)
                )
                # Set the x-coordinate of the goalkeeper
                # Goal Keeper
                self.x_pos_array[0] = width * 0.3 / 8

                # Defenders initial position
                self.x_pos_array[1:5] = [width * 1 / 8] * 4

                # Mid Fielders
                self.x_pos_array[5:8] = [width * 2 / 8] * 3
                

                # Attackers
                self.x_pos_array[8:11] = [width * 3 / 8] * (player_number - 7)

            elif side == "right":
                self.x_pos_array = (
                    [width * 7.7 / 8] * 1
                    + [width * 7 / 8] * 4
                    + [width * 6 / 8] * 4
                    + [width * 5 / 8] * (player_number - 7)
                )

        else:
            print("invalid side")

        # get y position for each player
        y_increment = height / (4 + 1)

        self.y_pos_array = []
        for i in range(4):
            self.y_pos_array.append(y_increment * (i + 1))

        y_increment = height / (3 + 1)
        for i in range(3):
            self.y_pos_array.append(y_increment * (i + 1))

        self.y_pos_array[0] = height / 2
        print(self.y_pos_array[0])

        y_increment = height / (player_number - 7 + 1)
        for i in range(player_number - 7):
            self.y_pos_array.append(y_increment * (i + 1))

    def set_position_to_initial(self):
        for player, x, y in zip(self.player_array, self.x_pos_array, self.y_pos_array):
            player.set_position(x, y)
            # zero velocity
            player.body.velocity = 0, 0

    # return reshpaed numpy array observation
    def get_observation(self):
        obs_array = []
        for player in self.player_array:
            obs_array.append(player.get_observation())
        obs_array = np.reshape(np.array(obs_array), -1)
        return obs_array

    def get_closest_teammate(self, player):
        min_distance = float("inf")
        closest_teammate = None
        for teammate in self.player_array:
            if teammate != player and teammate.side == player.side:
                distance = self.get_distance(
                    player.get_position(), teammate.get_position()
                )
                if distance < min_distance:
                    min_distance = distance
                    closest_teammate = teammate
        return closest_teammate

    def get_distance(self, pos1, pos2):
        """Calculate the Euclidean distance between two positions."""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx**2 + dy**2)

    def get_position_list(self):
        x_pos, y_pos = [], []
        for player in self.player_array:
            x, y = player.get_position()
            x_pos.append(x)
            y_pos.append(y)
        return np.array(x_pos), np.array(y_pos)

    def get_pass_target_teammate(self, player, arrow_keys):
        if self.player_number == 1:
            return player
        else:
            # choose any other player
            target_teammate = random.choices(
                self.player_array,
                weights=(np.array(self.player_array) != player).astype(int),
            )

            # noop
            if arrow_keys == 0:
                pass
            else:
                x_pos, y_pos = self.get_position_list()
                player_x, player_y = player.get_position()
                minus_x, minus_y = x_pos - player_x, y_pos - player_y
                # up
                if arrow_keys == 1:
                    if np.any(minus_y > 0):
                        target_teammate = random.choices(
                            self.player_array, weights=(minus_y > 0).astype(int)
                        )
                    else:
                        pass
                # right
                elif arrow_keys == 2:
                    if np.any(minus_x > 0):
                        target_teammate = random.choices(
                            self.player_array, weights=(minus_x > 0).astype(int)
                        )
                    else:
                        pass
                # down
                elif arrow_keys == 3:
                    if np.any(minus_y < 0):
                        target_teammate = random.choices(
                            self.player_array, weights=(minus_y < 0).astype(int)
                        )
                    else:
                        pass
                # left
                elif arrow_keys == 4:
                    if np.any(minus_x < 0):
                        target_teammate = random.choices(
                            self.player_array, weights=(minus_x < 0).astype(int)
                        )
                    else:
                        pass

            return target_teammate[0]

    def get_defender_initial_position(self):
        return self.x_pos_array[1]

    def get_midfielder_initial_position(self):
        return self.x_pos_array[5]
