import os
import numpy as np

def compile_data():
    files = ["aluminum", "copperround", "coppersquare", "steelround"]
    data = {}
    for file in files:
        cwd = os.getcwd()
        file_path = os.path.join(cwd, f"group23_{file}.csv")
        with open(file_path, "r") as f:
            data_slice = f.read()
            headers = data_slice.split('\n')[0].split(',')
            rows = [row.split(',') for row in data_slice.split('\n')[1:] if row.strip()]
            data[file] = {}
            readings = []
            for i, header in enumerate(headers):
                column_data = [float(row[i]) for row in rows if i < len(row) and row[i]]
                column_data = np.average(column_data) if column_data else 0
                readings.append(column_data)
            data[file]["readings"] = np.array(readings[2:-1]) # column 0 is the time column
            data[file]["T0"] = float(readings[1])          # column 1 is the initial temperature column
            data[file]["T_inf"] = float(readings[-1])      # column -1 is the ambient temperature column

    return data

def add_labsheet_data(data):
    for fin in data:
        if fin == "coppersquare":
            data[fin]["W"] = 1.27e-2
            data[fin]["H"] = 1.27e-2
            data[fin]["L"] = 28.50e-2
            data[fin]["k"] = 388
            data[fin]["A"] = data[fin]["W"] * data[fin]["H"]
            data[fin]["P"] = 2 * (data[fin]["W"] + data[fin]["H"])
            data[fin]["locs"] = 1e-2*np.array([3.06, 5.50, 9.82, 17.15, 22.07, 28.07])
            data[fin]["volts"] = 40
            data[fin]["T_b"] = 68
            data[fin]["current"] = 0.216
            data[fin]["power"] = data[fin]["volts"] * data[fin]["current"]
        elif fin == "copperround":
            data[fin]["D"] = 1.27e-2
            data[fin]["L"] = 17.20e-2
            data[fin]["k"] = 388
            data[fin]["A"] = np.pi * (data[fin]["D"] / 2) ** 2
            data[fin]["P"] = np.pi * data[fin]["D"]
            data[fin]["locs"] = 1e-2*np.array([3.09, 5.56, 9.80, 17.15]) # copper round has only 4 measurement points
            data[fin]["volts"] = 40
            data[fin]["T_b"] = 81
            data[fin]["current"] = 0.218
            data[fin]["power"] = data[fin]["volts"] * data[fin]["current"]
        elif fin == "steelround":
            data[fin]["D"] = 0.95e-2
            data[fin]["L"] = 28.50e-2
            data[fin]["k"] = 16
            data[fin]["A"] = np.pi * (data[fin]["D"] / 2) ** 2
            data[fin]["P"] = np.pi * data[fin]["D"]
            data[fin]["locs"] = 1e-2*np.array([3.10, 5.44, 9.81, 17.22, 22.07, 28.45])
            data[fin]["volts"] = 35
            data[fin]["T_b"] = 87
            data[fin]["current"] = 0.182
            data[fin]["power"] = data[fin]["volts"] * data[fin]["current"]
        elif fin == "aluminum":
            data[fin]["D"] = 1.27e-2
            data[fin]["L"] = 28.50e-2
            data[fin]["k"] = 167
            data[fin]["A"] = np.pi * (data[fin]["D"] / 2) ** 2
            data[fin]["P"] = np.pi * data[fin]["D"]
            data[fin]["locs"] = 1e-2*np.array([4.35, 5.41, 9.72, 17.15, 22.07, 28.07])
            data[fin]["volts"] = 45
            data[fin]["T_b"] = 91
            data[fin]["current"] = 0.247
            data[fin]["power"] = data[fin]["volts"] * data[fin]["current"]
    return data

def plot_all_data(data):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    for i, (fin, fin_data) in enumerate(data.items()):
        ax[i//2, i%2].scatter(fin_data["locs"]*1e2, fin_data["readings"], label='Measured Data', color='blue')
        if "h" in fin_data:
            h = fin_data["h"]
            # grid = np.logspace(-3, 3, 100)
            # for h in grid:
            #     loc_range = np.linspace(fin_data["locs"].min(), fin_data["locs"].max(), 100)
            #     model_temps = eq_1_rhs(h, fin_data, loc_range) * (fin_data["T_b"] - fin_data["T_inf"]) + fin_data["T_inf"]
            #     ax[i//2, i%2].plot(loc_range*1e2, model_temps, label='Fitted Model')
            #     #ax[i//2, i%2].legend()
            loc_range = np.linspace(fin_data["locs"].min(), fin_data["locs"].max(), 100)
            model_temps = eq_1_rhs(h, fin_data, loc_range) * (fin_data["T_b"] - fin_data["T_inf"]) + fin_data["T_inf"]
            ax[i//2, i%2].plot(loc_range*1e2, model_temps, label='Fitted Model', color='red')
            ax[i//2, i%2].legend()

        ax[i//2, i%2].set_xlabel('Location (cm)')
        ax[i//2, i%2].set_ylabel('Temperature (°C)')
        ax[i//2, i%2].set_title(f'{fin} - Temperature Distribution')
        ax[i//2, i%2].grid(True)
        ax[i//2, i%2].set_xlim(fin_data["locs"].min()*1e2 - 1, fin_data["locs"].max()*1e2 + 1)
        ax[i//2, i%2].set_ylim(fin_data["readings"].min() - 5, fin_data["readings"].max() + 5)

    plt.tight_layout()
    fig.savefig("temperature_distribution.png")
    plt.show()

def eq_1_rhs(h, fin_data, positions):
    P = fin_data["P"]
    Ac = fin_data["A"]
    k = fin_data["k"]
    L = fin_data["L"]
    if h <= 0:
        return np.zeros_like(positions, dtype=float)

    m = np.sqrt(h * P / (k * Ac))

    if m == 0:
        return np.zeros_like(positions, dtype=float)

    numerator = np.cosh(m * (L - positions)) + (h / (m * k)) * np.sinh(m * (L - positions))
    denominator = np.cosh(m * L) + (h / (m * k)) * np.sinh(m * L)
    return numerator / denominator

def find_h(data, fin):
    fin_data = data[fin]
    positions = fin_data["locs"]
    readings = fin_data["readings"]

    theta_b = fin_data["T_b"] - fin_data["T_inf"]
    if theta_b == 0:
        raise ValueError(f"Cannot fit h for {fin}: T_b and T_inf are equal")

    measured_ratio = (readings - fin_data["T_inf"]) / theta_b

    def objective(h):
        if h < 0:
            return np.inf
        model_ratio = eq_1_rhs(h, fin_data, positions)
        residuals = measured_ratio - model_ratio
        return float(np.sum(residuals ** 2))

    grid = np.logspace(-3, 3, 1000)
    errors = np.array([objective(h) for h in grid])
    best_index = int(np.argmin(errors))
    best_h = float(grid[best_index])

    if best_index == 0 or best_index == len(grid) - 1:
        print(f"Warning: Best h for {fin} is at the edge of the grid. Consider expanding the grid range.")
        return best_h

    lower = float(grid[best_index - 1])
    upper = float(grid[best_index + 1])

    phi = (1 + np.sqrt(5)) / 2
    inv_phi = 1 / phi
    c = upper - inv_phi * (upper - lower)
    d = lower + inv_phi * (upper - lower)
    fc = objective(c)
    fd = objective(d)

    # this is prob overkill lol
    for _ in range(80):
        if abs(upper - lower) < 1e-10:
            break
        if fc < fd:
            upper = d
            d = c
            fd = fc
            c = upper - inv_phi * (upper - lower)
            fc = objective(c)
        else:
            lower = c
            c = d
            fc = fd
            d = lower + inv_phi * (upper - lower)
            fd = objective(d)

    return float((lower + upper) / 2)

def equation_2(fin_data):
    h = fin_data["h"]
    P = fin_data["P"]
    Ac = fin_data["A"]
    k = fin_data["k"]
    L = fin_data["L"]
    th_b = fin_data["T_b"] - fin_data["T_inf"]

    M = th_b * np.sqrt(h * P * k * Ac)
    m = np.sqrt(h * P / (k * Ac))

    numerator = np.sinh(m * L) + (h / (m * k)) * np.cosh(m * L)
    denominator = np.cosh(m * L) + (h / (m * k)) * np.sinh(m * L)
    return M * numerator / denominator



data = compile_data()
data["copperround"]["readings"]=data["copperround"]["readings"][:-2] # copper round has only 4 measurement points
data = add_labsheet_data(data)
#data["copperround"]["T0"] = data["copperround"]["T1"]+3 # We didn't have this problem
thermocouple_depth = 0.64e-2

#plot_all_data(data)

for fin in data:
    h = find_h(data, fin)
    data[fin]["h"] = h
    print(f"Best h for {fin}: {h}")

#plot_all_data(data)

for fin in data:
    fin_data = data[fin]
    q_f = equation_2(fin_data)
    data[fin]["q_f"] = q_f
    print(f"Calculated heat transfer rate for {fin}: {q_f:.4f} W")
    print(f"Given power input for {fin}: {fin_data['power']:.4f} W")