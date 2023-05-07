import pymunk
from pymunk.vec2d import Vec2d
import math


class Player:
    def __init__(
        self,
        space,
        x,
        y,
        mass=70,
        radius=2,
        max_velocity=10,
        elasticity=0.2,
        color=(1, 0, 0, 1),
        side="left",
    ):
        self.space = space
        self.max_velocity = max_velocity
        self.color = color
        self.role = ""
        self.sub_role = ""
        self.side = side
        self.has_ball = False
        self.radius = radius
        self.body, self.shape = self._setup_player(
            space, x, y, mass, radius, elasticity
        )

    def get_position(self):
        x, y = self.body.position
        return [x, y]

    def get_velocity(self):
        vx, vy = self.body.velocity
        return [vx, vy]

    def set_velocity(self, vx, vy):
        self.body.velocity = Vec2d(vx, vy)

    def get_observation(self):
        return self.get_position() + self.get_velocity()

    def set_position(self, x, y):
        self.body.position = x, y

    # apply force on the center of the player
    # fx: force in x direction
    # fy: force in y direction

    def apply_force_to_player(self, fx, fy):
        self.body.apply_impulse_at_local_point((fx, fy), point=(0, 0))

    def _setup_player(self, space, x, y, mass, radius, elasticity):
        moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, moment)

        body.position = x, y
        body.start_position = Vec2d(body.position)

        def limit_velocity(body, gravity, damping, dt):
            pymunk.Body.update_velocity(body, gravity, damping, dt)
            l = body.velocity.length
            if l > self.max_velocity:
                scale = self.max_velocity / l
                body.velocity = body.velocity * scale

        body.velocity_func = limit_velocity

        shape = pymunk.Circle(body, radius)
        shape.color = self.color
        shape.elasticity = elasticity
        self.space.add(body, shape)
        return body, shape

    def move_to_ball(self, ball, team_players):
        # calculate the distance and angle between the player and the ball for all players on the team
        distances = []
        angles = []
        for player in team_players:
            distance = math.sqrt(
                (ball.body.position[0] - player.body.position[0]) ** 2
                + (ball.body.position[1] - player.body.position[1]) ** 2
            )
            angle = math.atan2(
                ball.body.position[1] - player.body.position[1],
                ball.body.position[0] - player.body.position[0],
            )
            distances.append(distance)
            angles.append(angle)

        # find the closest player to the ball
        closest_distance = min(distances)
        closest_player_index = distances.index(closest_distance)
        closest_player = team_players[closest_player_index]

        # move the closest player towards the ball
        closest_player.body.position = (
            closest_player.body.position[0]
            + math.cos(angles[closest_player_index]) * closest_distance * 0.1,
            closest_player.body.position[1]
            + math.sin(angles[closest_player_index]) * closest_distance * 0.1,
        )

    def get_distance_to_ball(self, ball):
        dx = self.body.position[0] - ball.body.position[0]
        dy = self.body.position[1] - ball.body.position[1]
        distance = math.sqrt(dx**2 + dy**2)
        return distance

    def move(self, destination_x, destination_y):
        velocity_x = destination_x - self.body.position[0]
        velocity_y = destination_y - self.body.position[1]
        if velocity_x != 0 or velocity_y != 0:
            speed = math.sqrt(velocity_x**2 + velocity_y**2)
            normalized_velocity_x = velocity_x / speed
            normalized_velocity_y = velocity_y / speed
            self.body.velocity = Vec2d(
                normalized_velocity_x * self.max_velocity,
                normalized_velocity_y * self.max_velocity,
            )
