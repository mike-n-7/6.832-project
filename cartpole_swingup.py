import math
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import (DirectCollocation, FloatingBaseType,
                         PiecewisePolynomial, RigidBodyTree, RigidBodyPlant,
                         SolutionResult)
from underactuated import (FindResource, PlanarRigidBodyVisualizer)


def run_dircol():
    tree = RigidBodyTree(FindResource("cartpole/cartpole.urdf"),
                         FloatingBaseType.kFixed)
    plant = RigidBodyPlant(tree)
    context = plant.CreateDefaultContext()

    timesteps = 301
    dt = 0.01
    dircol = DirectCollocation(plant, context, num_time_samples=timesteps,
                               minimum_timestep=dt, maximum_timestep=dt)
    dircol.AddEqualTimeIntervalsConstraints()

    initial_state = (0., 0., 0., 0.)
    dircol.AddBoundingBoxConstraint(initial_state, initial_state,
                                    dircol.initial_state())

    final_state = (0., math.pi, 0., 0.)
    # dircol.AddBoundingBoxConstraint(final_state, final_state,
    #                                dircol.final_state())

    xf = dircol.final_state()
    err = xf-final_state
    dircol.AddFinalCost(10000*np.dot(err.T, err))

    # Add constraint on final state.
    # dircol.AddLinearConstraint(xf[0], -5, -1)


    u = dircol.input()

    # Add control constraints.
    # dircol.AddConstraintToAllKnotPoints(u[0] <= 55)
    # dircol.AddConstraintToAllKnotPoints(u[0] >= -55)

    # Add constraints on change in input.
    # for t in xrange(0, timesteps-1):
    #     u, up1 = dircol.input(t), dircol.input(t+1)
    #     dircol.AddConstraint((u[0]-up1[0])**2 <= 300.)
    #     #dircol.AddCost((u-up1)**2)

    # Cost on input "effort".
    R = 1
    dircol.AddRunningCost(R*u[0]**2)


    initial_x_trajectory = \
       PiecewisePolynomial.FirstOrderHold([0., 5.],
                                         np.column_stack((initial_state,
                                                          final_state)))
    initial_u_trajectory = PiecewisePolynomial()
    dircol.SetInitialTrajectory(initial_u_trajectory, initial_x_trajectory)

    print 'Solving...'
    result = dircol.Solve()
    assert(result == SolutionResult.kSolutionFound)

    x_trajectory = dircol.ReconstructStateTrajectory()
    vis = PlanarRigidBodyVisualizer(tree, xlim=[-3, 3], ylim=[-1, 2.5])
    ani = vis.animate(x_trajectory, repeat=True)

    u_trajectory = dircol.ReconstructInputTrajectory()
    times = np.linspace(u_trajectory.start_time(), u_trajectory.end_time(), timesteps)
    u_lookup = np.vectorize(u_trajectory.value)
    u_values = u_lookup(times)

    print 'Input Cost: %f' % (np.sum(R*u_values**2)*dt)


    plt.figure()
    plt.plot(times, u_values)
    plt.xlabel('time (seconds)')
    plt.ylabel('force (Newtons)')

    plt.show()


if __name__ == '__main__':
    run_dircol()
