import math
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import (DirectCollocation, FloatingBaseType,
                         PiecewisePolynomial, RigidBodyTree, RigidBodyPlant,
                         SolutionResult)
from pydrake.examples.acrobot import AcrobotPlant
from underactuated import (FindResource, PlanarRigidBodyVisualizer)

plant = AcrobotPlant()
context = plant.CreateDefaultContext()
timesteps = 301
dt = 0.01
dircol = DirectCollocation(plant, context, num_time_samples=timesteps,
                           minimum_timestep=dt, maximum_timestep=dt)


# Add input limits.
u = dircol.input()

initial_state = (0., 0., 0., 0.)
dircol.AddBoundingBoxConstraint(initial_state, initial_state,
                                dircol.initial_state())

final_state = (math.pi, 0., 0., 0.)

xf = dircol.final_state()
err = xf - final_state
dircol.AddFinalCost(10000 * np.dot(err.T, err))

# dircol.AddBoundingBoxConstraint(final_state, final_state,
#                                dircol.final_state())

R = 1.  # Cost on input "effort".
dircol.AddRunningCost(R*u[0]**2)

initial_x_trajectory = \
    PiecewisePolynomial.FirstOrderHold([0., 3.],
                                       np.column_stack((initial_state,
                                                        final_state)))
dircol.SetInitialTrajectory(PiecewisePolynomial(), initial_x_trajectory)

print 'Solving...'
result = dircol.Solve()
assert(result == SolutionResult.kSolutionFound)

x_trajectory = dircol.ReconstructStateTrajectory()

tree = RigidBodyTree(FindResource("acrobot/acrobot.urdf"),
                     FloatingBaseType.kFixed)
vis = PlanarRigidBodyVisualizer(tree, xlim=[-4., 4.], ylim=[-4., 4.])
ani = vis.animate(x_trajectory, repeat=True)

u_trajectory = dircol.ReconstructInputTrajectory()
times = np.linspace(u_trajectory.start_time(), u_trajectory.end_time(), timesteps)
u_lookup = np.vectorize(u_trajectory.value)
u_values = u_lookup(times)

# Print final input cost.
print np.sum(R*np.sum(u_values**2))*dt

plt.figure()
plt.plot(times, u_values)
plt.xlabel('time (seconds)')
plt.ylabel('force (Newtons)')

plt.show()
