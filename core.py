"""
Вычислительное ядро для анализа обобщённого отображения параболы:
    x_{n+1} = 1 - mu * |x_n|^z

Все тяжёлые функции векторизованы через NumPy: итерации по всем
значениям mu выполняются одновременно как операции над массивами.
"""

import numpy as np

OVERFLOW = 1e10


def generalized_map(x, mu, z):
    """Обобщённое отображение параболы (скаляр или массив)."""
    return 1.0 - mu * np.abs(x) ** z


def generalized_map_derivative(x, mu, z):
    """Производная отображения по x."""
    return -mu * z * np.abs(x) ** (z - 1) * np.sign(x)


# ======================================================================
#  Бифуркационная диаграмма (векторизована)
# ======================================================================

def bifurcation_data(z, mu_min=0.0, mu_max=2.0, n_mu=1200,
                     n_skip=400, n_plot=200):
    """
    Возвращает (mu_values, x_values) для scatter-plot.
    Все n_mu значений mu итерируются одновременно.
    """
    mu_arr = np.linspace(mu_min, mu_max, n_mu)
    x = np.full(n_mu, 0.1)

    for _ in range(n_skip):
        x = 1.0 - mu_arr * np.abs(x) ** z
        x = np.clip(x, -OVERFLOW, OVERFLOW)

    mu_out = np.empty(n_mu * n_plot)
    x_out = np.empty(n_mu * n_plot)
    for j in range(n_plot):
        x = 1.0 - mu_arr * np.abs(x) ** z
        x = np.clip(x, -OVERFLOW, OVERFLOW)
        mu_out[j * n_mu:(j + 1) * n_mu] = mu_arr
        x_out[j * n_mu:(j + 1) * n_mu] = x

    valid = np.isfinite(x_out) & (np.abs(x_out) < OVERFLOW)
    return mu_out[valid], x_out[valid]


# ======================================================================
#  Показатель Ляпунова (векторизован)
# ======================================================================

def lyapunov_exponent(mu_arr, z, n_iter=1000, n_skip=300):
    """
    lambda(mu) = (1/N) * sum ln|f'(x_n)|  для каждого mu.
    """
    mu_arr = np.atleast_1d(mu_arr).astype(float)
    n = len(mu_arr)
    x = np.full(n, 0.1)

    for _ in range(n_skip):
        x = 1.0 - mu_arr * np.abs(x) ** z

    lyap_sum = np.zeros(n)
    for _ in range(n_iter):
        df = np.abs(-mu_arr * z * np.abs(x) ** (z - 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            lyap_sum += np.where(df > 0, np.log(df), -50.0)
        x = 1.0 - mu_arr * np.abs(x) ** z

    result = lyap_sum / n_iter
    bad = ~np.isfinite(result)
    result[bad] = np.nan
    return result


# ======================================================================
#  Паутинная диаграмма (скалярная — быстрая сама по себе)
# ======================================================================

def cobweb_data(mu, z, x0=0.1, n_iter=80):
    """Возвращает (cobweb_x, cobweb_y) — ломаную для отрисовки."""
    cx = [x0, x0]
    cy = [0.0, 1.0 - mu * abs(x0) ** z]
    x = x0
    for _ in range(n_iter):
        y = 1.0 - mu * abs(x) ** z
        cx.extend([x, y])
        cy.extend([y, y])
        x = y
        if abs(x) > OVERFLOW:
            break
    return np.array(cx), np.array(cy)


# ======================================================================
#  Определение периода — пакетная версия (векторизована)
# ======================================================================

def _detect_periods_batch(mu_arr, z, n_skip=2000, n_check=512, tol=1e-6):
    """
    Определяет устойчивый период для каждого mu из массива.
    Все mu итерируются одновременно. Возвращает int-массив периодов
    (0 = не определён / хаос).
    """
    mu_arr = np.asarray(mu_arr, dtype=float)
    n = len(mu_arr)
    x = np.full(n, 0.1)

    for _ in range(n_skip):
        x = 1.0 - mu_arr * np.abs(x) ** z

    orbit = np.empty((n_check, n))
    orbit[0] = x
    for i in range(1, n_check):
        orbit[i] = 1.0 - mu_arr * np.abs(orbit[i - 1]) ** z

    ref = orbit[-1]
    periods = np.zeros(n, dtype=int)

    for period in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        if period * 8 > n_check:
            break

        undecided = periods == 0
        if not np.any(undecided):
            break

        match = np.ones(n, dtype=bool) & undecided
        for k in range(1, 8):
            diff = np.abs(orbit[-1 - k * period] - ref)
            match &= diff < tol

        if period > 1:
            sub_match = np.ones(n, dtype=bool) & match
            half = period // 2
            for k in range(1, 8):
                diff = np.abs(orbit[-1 - k * half] - ref)
                sub_match &= diff < tol
            match &= ~sub_match

        periods[match] = period

    diverged = ~np.isfinite(ref) | (np.abs(ref) > OVERFLOW)
    periods[diverged] = 0
    return periods


def _detect_period(mu, z, n_skip=2000, n_check=512, tol=1e-6):
    """Скалярная обёртка для совместимости (используется в бисекции)."""
    return int(_detect_periods_batch(np.array([mu]), z, n_skip, n_check, tol)[0])


# ======================================================================
#  Поиск точек бифуркаций
# ======================================================================

def _bisect_bifurcation(z, p_before, mu_lo, mu_hi):
    """Уточняет точку бифуркации бисекцией между mu_lo и mu_hi."""
    lo, hi = mu_lo, mu_hi
    for _ in range(60):
        mid = (lo + hi) / 2.0
        per = _detect_period(mid, z)
        if per == p_before:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    return (lo + hi) / 2.0


def find_bifurcation_points(z, n_bifurcations=8):
    """
    Находит точки бифуркаций удвоения периода.
    Грубый скан (векторизованный) + предсказание + бисекция.
    """
    n_scan = 800
    scan_mus = np.linspace(0.001, 2.0, n_scan)
    periods = _detect_periods_batch(scan_mus, z)

    stable_zones = []
    run_start = 0
    for i in range(1, n_scan):
        if periods[i] != periods[run_start] or i == n_scan - 1:
            length = i - run_start
            if length >= 3 and periods[run_start] > 0:
                stable_zones.append((
                    int(periods[run_start]),
                    scan_mus[run_start],
                    scan_mus[min(i - 1, n_scan - 1)],
                ))
            run_start = i

    bif_points = []
    for zi in range(len(stable_zones) - 1):
        p1, _, mu1_end = stable_zones[zi]
        p2, mu2_start, _ = stable_zones[zi + 1]
        if p2 == p1 * 2:
            bif_mu = _bisect_bifurcation(z, p1, mu1_end, mu2_start)
            bif_points.append(bif_mu)

    while len(bif_points) >= 2 and len(bif_points) < n_bifurcations - 1:
        n = len(bif_points)
        d_last = bif_points[-1] - bif_points[-2]
        ds = feigenbaum_deltas(np.array(bif_points))
        delta_est = ds[-1] if len(ds) > 0 and np.isfinite(ds[-1]) else 4.5
        delta_est = max(2.0, min(delta_est, 20.0))

        d_next = d_last / delta_est
        predicted = bif_points[-1] + d_next
        if predicted > 2.0 or predicted < bif_points[-1]:
            break

        lo_s = bif_points[-1] + d_next * 0.1
        hi_s = min(2.0, predicted + d_next * 2)

        p_before = 1 << n
        n_fine = 300
        fine_mus = np.linspace(lo_s, hi_s, n_fine)
        fine_periods = _detect_periods_batch(fine_mus, z)

        start_idx = -1
        for i in range(n_fine):
            if fine_periods[i] == p_before:
                start_idx = i
                break

        found = False
        if start_idx >= 0:
            for i in range(start_idx + 1, n_fine):
                if fine_periods[i - 1] == p_before and fine_periods[i] != p_before:
                    bif_mu = _bisect_bifurcation(z, p_before,
                                                  fine_mus[i - 1], fine_mus[i])
                    bif_points.append(bif_mu)
                    found = True
                    break

        if not found:
            break

    return np.array(bif_points)


# ======================================================================
#  Константы Фейгенбаума
# ======================================================================

def feigenbaum_deltas(bif_points):
    """delta_n = (mu_n - mu_{n-1}) / (mu_{n+1} - mu_n)"""
    if len(bif_points) < 3:
        return np.array([])
    bp = np.asarray(bif_points)
    num = bp[1:-1] - bp[:-2]
    den = bp[2:] - bp[1:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(den) > 1e-15, num / den, np.nan)


def feigenbaum_alphas(z, bif_points):
    """
    alpha_n через отношение расстояний суперустойчивых орбит
    от критической точки x=0.
    """
    if len(bif_points) < 3:
        return np.array([])

    alphas = []
    for i in range(len(bif_points) - 1):
        period = 1 << (i + 1)
        mu = bif_points[i]
        x = 0.0
        for _ in range(period // 2):
            x = 1.0 - mu * abs(x) ** z
        d_curr = abs(x)

        mu_next = bif_points[i + 1]
        x = 0.0
        next_period = 1 << (i + 2)
        for _ in range(next_period // 2):
            x = 1.0 - mu_next * abs(x) ** z
        d_next = abs(x)

        if abs(d_next) > 1e-15:
            alphas.append(d_curr / d_next)
        else:
            alphas.append(np.nan)

    return np.array(alphas)


def feigenbaum_constants_vs_z(z_values, n_bif=6):
    """Предельные delta и alpha для набора значений z."""
    delta_arr = np.full(len(z_values), np.nan)
    alpha_arr = np.full(len(z_values), np.nan)

    for i, z in enumerate(z_values):
        bp = find_bifurcation_points(z, n_bif)
        if len(bp) >= 4:
            ds = feigenbaum_deltas(bp)
            valid = ds[~np.isnan(ds)]
            if len(valid) > 0:
                delta_arr[i] = valid[-1]
            als = feigenbaum_alphas(z, bp)
            valid_a = als[~np.isnan(als)]
            if len(valid_a) > 0:
                alpha_arr[i] = valid_a[-1]

    return delta_arr, alpha_arr


# ======================================================================
#  Скейлинг
# ======================================================================

def scaling_zoom_regions(z, n_zooms=3):
    """Прямоугольные области для демонстрации самоподобия."""
    bp = find_bifurcation_points(z, n_zooms + 3)
    if len(bp) < 3:
        return [(0.0, 2.0, -1.5, 1.5)]

    regions = [(0.0, bp[-1] * 1.05, -1.5, 1.5)]

    for k in range(min(n_zooms - 1, len(bp) - 2)):
        idx = k + 1
        if idx + 1 >= len(bp):
            break
        mu_width = bp[idx + 1] - bp[idx - 1]
        mu_lo = bp[idx] - mu_width * 0.3
        mu_hi = bp[idx + 1] + mu_width * 0.1

        _, x_data = bifurcation_data(z, mu_lo, mu_hi,
                                     n_mu=400, n_skip=500, n_plot=150)
        if len(x_data) > 0:
            x_lo = np.percentile(x_data, 1) - 0.05
            x_hi = np.percentile(x_data, 99) + 0.05
        else:
            x_lo, x_hi = -1.0, 1.0

        regions.append((mu_lo, mu_hi, x_lo, x_hi))

    return regions
