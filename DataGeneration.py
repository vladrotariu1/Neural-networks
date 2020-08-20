import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Point:

    def __init__(self):
        self.x = np.random.uniform(low=0, high=200)
        self.y = np.random.uniform(low=0, high=200)

        if self.y > 50 and self.x ** 1 > 40:
            self.label = 1
        else:
            self.label = 0


def data():

    points = []
    label = []
    for i in range(0, 200):
        coordonate = Point()
        points.append([round(coordonate.x, 2), round(coordonate.y, 2)])
        label.append([coordonate.label])
    return np.array(points), np.array(label)


'''
points1 = []
points_x = []
points_y = []
label = []
for i in range(0, 200):
    points1.append(Point())
for point in points1:
    points_x.append(point.x)
for point in points1:
    points_y.append(point.y)
for point in points1:
    label.append(point.label)

plt.xlabel("X")
plt.xlabel("Y")
sns.scatterplot(points_x, points_y, label)
plt.show()
'''