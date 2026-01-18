from cubic_spline import CubicSpline1D
from matplotlib import pyplot as plt

# берём все точки
points = (
    (0, 0),
    (1, 1),
    (2, 3),
    (3, 2),
    (4, 5),
    (5, 3),
    (6, 3),
    (7, 5),
)

# строим сплайн
spline = CubicSpline1D(points)

# генерируем сетку для отображения
xp = [i / 50 for i in range(0, 7 * 50 + 1)]   # шаг 0.02 примерно
ys = spline(xp)                               # значения сплайна

# исходные точки (для проверки интерполяции)
xs_pts = [p[0] for p in points]
ys_pts = [p[1] for p in points]

# рисуем
plt.figure(figsize=(10, 5))
plt.plot(xp, ys, label="Cubic Spline", linewidth=2)
plt.scatter(xs_pts, ys_pts, color="red", s=50, label="Points")
plt.grid(True)
plt.legend()
plt.title("Natural Cubic Spline Interpolation")
plt.xlabel("x")
plt.ylabel("S(x)")
plt.show()

