import pymunk
from pymunk.vec2d import Vec2d


class Circle:
    def __init__(
        self,
        space,
        x,
        y,
        mass=10,
        radius=10,
    ):
        self.space = space
        self.body, self.shape = self._setup_circle(space, x, y, mass, radius)


    def set_position(self, x, y):
        self.body.position = x, y

    def _setup_circle(self, space, x, y, mass, radius):
        body = pymunk.Body(body_type=pymunk.Body.STATIC, moment=pymunk.inf)
        body.position = x, y

        circle_shape = pymunk.Circle(body, radius, (0, 0))
        circle_shape.outline = 8
        circle_shape.outline_color = (1, 0, 0, 1)
        circle_shape.color = (0.0, 0.0, 0.0, 0.0)
        circle_shape.sensor = True
        self.space.add(body, circle_shape)

        return body, circle_shape
