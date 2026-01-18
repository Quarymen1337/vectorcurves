from bisect import bisect_right
from math import sqrt

class CubicSpline1D:
    """
    Натуральный кубический сплайн по точкам (x_i, y_i).

    - x_i должны быть строго возрастающими.
    - Граничные условия: S''(x_0) = 0, S''(x_n) = 0.
    """

    def __init__(self, points):
        # ---------- 1. Разбор входных данных ----------
        xs, ys = zip(*points)
        self.x = list(xs)
        self.y = list(ys)

        n = len(self.x) - 1  # число интервалов
        if n < 1:
            raise ValueError("Нужно минимум 2 точки для сплайна")

        # ---------- 2. h_i = x_{i+1} - x_i ----------
        h = [self.x[i + 1] - self.x[i] for i in range(n)]
        if any(hi <= 0 for hi in h):
            raise ValueError("x_i должны быть строго возрастающими")

        self.h = h

        # ---------- 3. Сборка трёхдиагональной системы для M_i ----------
        # Неизвестные: M_0 .. M_n, но M_0 = M_n = 0 (натуральный),
        # значит решаем систему только для i = 1..n-1 (m = n-1 уравнений).
        m = n - 1
        if m > 0:
            alpha = [0.0] * m
            beta  = [0.0] * m
            gamma = [0.0] * m
            rhs   = [0.0] * m

            for k in range(m):
                i = k + 1  # реальный индекс узла (1..n-1)

                alpha[k] = h[i - 1]
                beta[k]  = 2.0 * (h[i - 1] + h[i])
                gamma[k] = h[i]

                rhs[k] = 6.0 * (
                    (self.y[i + 1] - self.y[i])   / h[i] -
                    (self.y[i]     - self.y[i-1]) / h[i-1]
                )

            # ---------- 4. Метод прогонки (Thomas) ----------
            # Прямой ход
            c = [0.0] * m
            d = [0.0] * m

            # i = 0
            c[0] = gamma[0] / beta[0]
            d[0] = rhs[0]   / beta[0]

            # i = 1..m-1
            for k in range(1, m):
                denom = beta[k] - alpha[k] * c[k - 1]
                c[k] = gamma[k] / denom
                d[k] = (rhs[k] - alpha[k] * d[k - 1]) / denom

            # Обратный ход: находим M_1..M_{n-1}
            M = [0.0] * (n + 1)  # M_0..M_n, сразу заполняем
            M[n - 1] = d[m - 1]
            for k in range(m - 2, -1, -1):
                i = k + 1  # узел
                M[i] = d[k] - c[k] * M[i + 1]

            # Граничные условия натурального сплайна
            M[0] = 0.0
            M[n] = 0.0
        else:
            # Если всего один интервал (2 точки) – просто прямая
            M = [0.0, 0.0]

        self.M = M

        # ---------- 5. Предвычисляем A_i, B_i для формулы сплайна ----------
        A = [0.0] * n
        B = [0.0] * n
        for i in range(n):
            hi = h[i]
            A[i] = self.y[i]   - M[i]   * hi**2 / 6.0
            B[i] = self.y[i+1] - M[i+1] * hi**2 / 6.0

        self.A = A
        self.B = B

    # ---------- 6. Поиск интервала ----------
    def _find_interval(self, xq):
        """
        Найти i такое, что x[i] <= xq <= x[i+1].
        Используем bisect для логарифмического поиска.
        """
        if xq <= self.x[0]:
            return 0
        if xq >= self.x[-1]:
            return len(self.x) - 2
        # bisect_right вернет позицию > xq, вычитаем 1
        i = bisect_right(self.x, xq) - 1
        return max(0, min(i, len(self.x) - 2))

    # ---------- 7. Вычисление значения сплайна ----------
    def __call__(self, xq):
        """
        Вернуть S(xq).
        xq может быть числом или итерируемым объектом (список, numpy-массив и т.п.).
        """
        # Если это "скаляр" (int/float)
        if isinstance(xq, (int, float)):
            return self._eval_scalar(float(xq))
        # Иначе считаем поэлементно
        return [self._eval_scalar(float(x)) for x in xq]

    def _eval_scalar(self, xq: float) -> float:
        i = self._find_interval(xq)
        xi, xi1 = self.x[i], self.x[i+1]
        hi = self.h[i]
        Mi, Mi1 = self.M[i], self.M[i+1]
        Ai, Bi = self.A[i], self.B[i]

        dxL = xi1 - xq
        dxR = xq - xi

        term1 = Mi  * dxL**3 / (6.0 * hi)
        term2 = Mi1 * dxR**3 / (6.0 * hi)
        term3 = Ai  * dxL     / hi
        term4 = Bi  * dxR     / hi

        return term1 + term2 + term3 + term4

    # ---------- 8. Производная S'(x) (по желанию) ----------
    def derivative(self, xq):
        """
        Вернуть S'(xq).
        """
        if isinstance(xq, (int, float)):
            return self._eval_deriv_scalar(float(xq))
        return [self._eval_deriv_scalar(float(x)) for x in xq]

    def _eval_deriv_scalar(self, xq: float) -> float:
        i = self._find_interval(xq)
        xi, xi1 = self.x[i], self.x[i+1]
        hi = self.h[i]
        Mi, Mi1 = self.M[i], self.M[i+1]
        Ai, Bi = self.A[i], self.B[i]

        dxL = xi1 - xq
        dxR = xq - xi

        # S'(x) =
        # - Mi  * dxL^2 / (2 h_i) + Mi1 * dxR^2 / (2 h_i) + (B_i - A_i) / h_i
        term1 = - Mi  * dxL**2 / (2.0 * hi)
        term2 =   Mi1 * dxR**2 / (2.0 * hi)
        term3 = (Bi - Ai) / hi

        return term1 + term2 + term3


class CubicSpline3D:
    """
    Параметрический 3D-сплайн:
      c(t) = (Sx(t), Sy(t), Sz(t)),  t ∈ [0, 1]

    points: последовательность (x, y, z)
    t-параметр строится по дуговой длине.
    """

    def __init__(self, points):
        pts = list(points)
        if len(pts) < 2:
            raise ValueError("Нужно минимум 2 точки для 3D-сплайна")

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]

        # --- 1. параметр t по дуговой длине ---
        # s_0 = 0, s_k = sum |p_k - p_{k-1}|
        s = [0.0]
        for i in range(1, len(pts)):
            dx = xs[i] - xs[i-1]
            dy = ys[i] - ys[i-1]
            dz = zs[i] - zs[i-1]
            dist = sqrt(dx*dx + dy*dy + dz*dz)
            s.append(s[-1] + dist)

        total_len = s[-1]
        if total_len == 0.0:
            # все точки совпали — делаем равномерный параметр
            t = [i / (len(pts)-1) for i in range(len(pts))]
        else:
            t = [si / total_len for si in s]   # нормируем в [0,1]

        self.t0 = 0.0
        self.t1 = 1.0

        # --- 2. строим три 1D сплайна по одному и тому же t ---
        self.sx = CubicSpline1D(list(zip(t, xs)))
        self.sy = CubicSpline1D(list(zip(t, ys)))
        self.sz = CubicSpline1D(list(zip(t, zs)))

    # ---------- базовая оценка c(t) ----------
    def __call__(self, t):
        """
        Вернуть точку(и) на кривой для параметра t ∈ [0,1].
        t может быть числом или итерируемым объектом.
        """
        if isinstance(t, (int, float)):
            return self._eval_scalar(float(t))
        return [self._eval_scalar(float(u)) for u in t]

    def _eval_scalar(self, u: float):
        # можно чуть поджать t в [t0, t1]
        if u < self.t0:
            u = self.t0
        elif u > self.t1:
            u = self.t1
        return (
            self.sx(u),
            self.sy(u),
            self.sz(u),
        )

    # ---------- производная c'(t) ----------
    def derivative(self, t):
        """
        Вернуть вектор производной c'(t) (по параметру t).
        """
        if isinstance(t, (int, float)):
            return self._eval_deriv_scalar(float(t))
        return [self._eval_deriv_scalar(float(u)) for u in t]

    def _eval_deriv_scalar(self, u: float):
        if u < self.t0:
            u = self.t0
        elif u > self.t1:
            u = self.t1
        return (
            self.sx.derivative(u),
            self.sy.derivative(u),
            self.sz.derivative(u),
        )

    # ---------- единичный тангенс T(t) ----------
    def tangent(self, t):
        """
        Единичный тангенс T(t) = c'(t) / ||c'(t)||.
        """
        if isinstance(t, (int, float)):
            return self._eval_tangent_scalar(float(t))
        return [self._eval_tangent_scalar(float(u)) for u in t]

    def _eval_tangent_scalar(self, u: float):
        dx, dy, dz = self._eval_deriv_scalar(u)
        norm = sqrt(dx*dx + dy*dy + dz*dz)
        if norm == 0.0:
            return (0.0, 0.0, 0.0)
        return (dx / norm, dy / norm, dz / norm)

    # ---------- удобный метод: выборка n точек вдоль кривой ----------
    def sample(self, n=100):
        """
        Вернуть список из n точек вдоль кривой (равномерно по t).
        """
        if n < 2:
            raise ValueError("n >= 2")
        ts = [self.t0 + (self.t1 - self.t0) * i / (n - 1) for i in range(n)]
        return self(ts)

