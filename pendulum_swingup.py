import math
import numpy as np
import matplotlib.pyplot as plt

from pydrake.examples.pendulum import (PendulumPlant, PendulumState)
from pydrake.all import (DirectCollocation, PiecewisePolynomial,
                         SolutionResult)
from visualizer import PendulumVisualizer

plant = PendulumPlant()
context = plant.CreateDefaultContext()

timesteps = 501
dt = 0.01

dircol = DirectCollocation(plant, context, num_time_samples=timesteps,
                           minimum_timestep=dt, maximum_timestep=dt)

dircol.AddEqualTimeIntervalsConstraints()

final_state = [np.pi, 0]
xf = dircol.final_state()
err = xf - final_state
dircol.AddFinalCost(10000 * np.dot(err.T, err))

torque_limit = 3.0  # N*m.
u = dircol.input()
dircol.AddConstraintToAllKnotPoints(-torque_limit <= u[0])
dircol.AddConstraintToAllKnotPoints(u[0] <= torque_limit)

initial_state = PendulumState()
initial_state.set_theta(0.0)
initial_state.set_thetadot(0.0)
dircol.AddBoundingBoxConstraint(initial_state.get_value(),
                                initial_state.get_value(),
                                dircol.initial_state())

final_state = PendulumState()
final_state.set_theta(math.pi)
final_state.set_thetadot(0.0)
# dircol.AddBoundingBoxConstraint(final_state.get_value(),
#                                final_state.get_value(),
#                                dircol.final_state())


R = 1  # Cost on input "effort".
dircol.AddRunningCost(R*u[0]**2)

initial_x_trajectory = \
    PiecewisePolynomial.FirstOrderHold([0., 4.],
                                       [initial_state.get_value(),
                                        final_state.get_value()])
dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)

result = dircol.Solve()
assert(result == SolutionResult.kSolutionFound)

x_trajectory = dircol.ReconstructStateTrajectory()

vis = PendulumVisualizer()
ani = vis.animate(x_trajectory, repeat=True)

x_knots = np.hstack([x_trajectory.value(t) for t in
                     np.linspace(x_trajectory.start_time(),
                                 x_trajectory.end_time(), timesteps)])
plt.figure()
plt.plot(x_knots[0, :], x_knots[1, :])

plt.show()

u_trajectory = dircol.ReconstructInputTrajectory()
times = np.linspace(u_trajectory.start_time(), u_trajectory.end_time(), timesteps)
u_lookup = np.vectorize(u_trajectory.value)
u_values = u_lookup(times)

# Print the final cost.
x_f = x_trajectory.value(5).flatten()
print x_f
err = x_f - [np.pi, 0]
print np.sum(u_values**2)*dt*R + 10000*np.dot(err.T, err)