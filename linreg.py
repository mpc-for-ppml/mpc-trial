from mpyc.runtime import mpc
import numpy as np
import matplotlib.pyplot as plt

secfxp = mpc.SecFxp()

async def secure_linear_regression(X_local, y_local):
    await mpc.start()

    # Step 1: Secret-share the local data
    X = [[secfxp(x) for x in row] for row in X_local]
    y = [secfxp(val) for val in y_local]

    # Step 2: Compute X^T X and X^T y securely
    Xt = list(zip(*X))
    XtX = [[sum(xi * xj for xi, xj in zip(row_i, row_j)) for row_j in Xt] for row_i in Xt]
    Xty = [sum(xi * yi for xi, yi in zip(row, y)) for row in Xt]

    # Step 3: Open XtX and Xty
    XtX_flat = [elem for row in XtX for elem in row]
    XtX_open_flat = await mpc.output(XtX_flat)
    n = len(XtX)
    XtX_open = [XtX_open_flat[i * n:(i + 1) * n] for i in range(n)]

    Xty_open = await mpc.output(Xty)

    # Step 4: Solve for theta (plain domain here, for simplicity)
    XtX_np = np.array([[float(x) for x in row] for row in XtX_open])
    Xty_np = np.array([float(v) for v in Xty_open])
    lambda_reg = 1e-4
    XtX_np += lambda_reg * np.identity(XtX_np.shape[0])
    theta = np.linalg.solve(XtX_np, Xty_np)

    await mpc.shutdown()
    return theta


# Each party will have their own data — simulate here
np.random.seed(200)

X_local = np.random.rand(10, 5).tolist()
X_local = [x + [1] for x in X_local]  # Add bias
y_local = np.random.rand(10).tolist()

theta = mpc.run(secure_linear_regression(X_local, y_local))
print("Estimated theta (coefficients):", theta)

X = np.array(X_local)
theta_vec = np.array(theta)
y_pred = X @ theta_vec
print("Predicted y:", y_pred)
print("Actual y:", y_local)

y_true = np.array(y_local)
mse = np.mean((y_pred - y_true) ** 2)
print("Mean Squared Error:", mse)

plt.plot(y_true, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted")
plt.show()