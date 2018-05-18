import matplotlib.pyplot as plt

from pydrake.all import (BasicVector, SignalLogger, Simulator, DiagramBuilder, RigidBodyPlant, RigidBodyTree, VectorSystem, FloatingBaseType, LinearQuadraticRegulator)
from underactuated import (PlanarRigidBodyVisualizer)
from pydrake.all import Linearize, FirstOrderTaylorApproximation, MathematicalProgram

import numpy as np
import math
import pickle

class Controller(VectorSystem):
    def __init__(self, state_size, control_size, control_law, dt):
        VectorSystem.__init__(self, state_size, control_size)
        self.control_size = control_size
        self.controller = control_law
        self.dt = dt
    
    def _DoCalcVectorOutput(self, context, u, x, y):
        """ The input u is actually the state of the plant.
            y is the output that needs to be set.
        """
        tx = int(context.get_time()/self.dt)
        u = self.controller.control_law(u, tx)
        y[:] = u


class WrapTheta(VectorSystem):
    def __init__(self, wrap_state, n_states):
        VectorSystem.__init__(self, n_states, n_states)
        self.wrap_state = wrap_state
        
    def _DoCalcVectorOutput(self, context, input, state, output):
        wrap_state = self.wrap_state
        output[:] = input
        for w in wrap_state:
            output[w] = output[w] - np.pi*2 * math.floor(output[w]/(np.pi*2))


def run_simulation(control_law, initial_state, urdf, dt, duration, wrap_state):
    builder = DiagramBuilder()

    # Build the subsystems involved in the entire simulation.
    tree = RigidBodyTree(
                urdf,
                FloatingBaseType.kFixed
            )
    plant = builder.AddSystem(
                RigidBodyPlant(
                    tree,
                    dt
                )
            )

    controller = builder.AddSystem(
                    Controller(
                        plant.get_output_size(),
                        plant.get_input_size(),
                        control_law,
                        dt
                    )
                )

    # wrap = builder.AddSystem(WrapTheta(wrap_state, initial_state.shape[0]))

    # Log the state and input trajectories from the simulation.
    input_log = builder.AddSystem(
                    SignalLogger(plant.get_input_size())
                )
    input_log._DeclarePeriodicPublish(dt, 0.0)
    state_log = builder.AddSystem(
                    SignalLogger(plant.get_output_size())
                )
    state_log._DeclarePeriodicPublish(dt, 0.0)

    # Link the controller to the plant.
    # builder.Connect(plant.get_output_port(0), wrap.get_input_port(0))
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))
    # builder.Connect(wrap.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))

    # Connect the state of the plant to the visualizer.
    builder.Connect(plant.get_output_port(0), state_log.get_input_port(0))
    builder.Connect(controller.get_output_port(0), input_log.get_input_port(0))
    diagram = builder.Build()
    
    simulator = Simulator(diagram)
    simulator.set_publish_every_time_step(False)

    context = simulator.get_mutable_context()
    context.get_discrete_state_vector().SetFromVector(initial_state)
    
    # simulator.Initialize()

    # simulator.get_integrator().set_fixed_step_mode(True)
    # simulator.get_integrator().set_maximum_step_size(0.005)
    simulator.StepTo(duration)
    return state_log, input_log


def autodiff(states, inputs, dt, urdf):
    n_timesteps, n_states, n_inputs = states.shape[1], states.shape[0], inputs.shape[0]
    fx = np.zeros((n_states, n_states, n_timesteps))
    fu = np.zeros((n_states, n_inputs, n_timesteps))
    f0 = np.zeros((n_states, 1, n_timesteps))
    # Create a continuous version of the plant.
    tree = RigidBodyTree(urdf, FloatingBaseType.kFixed)
    plant = RigidBodyPlant(tree)
    
    context = plant.CreateDefaultContext()
    for tx in xrange(0, n_timesteps):
        context.FixInputPort(0, BasicVector(inputs[:, tx]))
        context.get_mutable_continuous_state_vector().SetFromVector(states[:, tx])

        lin = FirstOrderTaylorApproximation(plant, context)
        fx[:, :, tx] = np.eye(n_states) + lin.A()*dt
        fu[:, :, tx] = lin.B()*dt
        f0[:, 0, tx] = lin.f0()*dt
    return fx, fu, f0


class QRLoss(object):
    def __init__(self, p_q, p_r, p_xg):
        self.Qf = np.eye(len(p_xg))*p_q
        self.Q = 0.
        self.R = p_r
        self.xg = p_xg
    
    def get_final_cost(self, p_states):
        
        n_states = p_states.shape[0]
        x = p_states[:, -1].copy()

        x_err = x - self.xg
        loss = np.dot(x_err.T, np.dot(self.Qf, x_err))
        loss_x = 2*np.dot(x_err.T, self.Qf)
        loss_xx = 2*self.Qf
        return loss, loss_x.reshape((n_states, 1)), loss_xx
    
    def get_intermediate_costs(self, p_states, p_inputs):
        n_inputs = p_inputs.shape[0]
        n_timesteps = p_inputs.shape[1]
        n_states = p_states.shape[0]
        
        loss = np.zeros((n_timesteps,))
        loss_u = np.zeros((n_inputs, 1, n_timesteps))
        loss_uu = np.ones((n_inputs, n_inputs, n_timesteps))*2*self.R
        
        loss_x = np.zeros((n_states, 1, n_timesteps))
        loss_xx = np.zeros((n_states, n_states, n_timesteps))
        
        for tx in xrange(0, n_timesteps):
            u = p_inputs[:, tx].copy()
            loss[tx] = self.R*np.dot(u.T, u)
            loss_u[:, 0, tx] = 2*u.T*self.R
            
            x = p_states[:, tx].copy()
            x_err = x - self.xg
            loss[tx] += self.Q*np.dot(x_err.T, x_err)
            loss_x[:, 0, tx] = 2*x_err*self.Q
            loss_xx[:, :, tx] = 2*self.Q*np.eye(n_states)
            
        return loss, loss_u, loss_uu, loss_x, loss_xx


class iLQG(object):
    
    def __init__(self, p_model, p_timesteps, p_dt, p_constrain_u=None):
        n_states = p_model.initial_state.shape[0]
        
        self.nominal_states = np.repeat(p_model.initial_state, p_timesteps, axis=1)
        self.nominal_inputs = np.zeros((1, p_timesteps))
        self.k = np.zeros((1, p_timesteps))
        self.K = np.zeros((1, n_states, p_timesteps))
        
        self.loss = p_model.loss
        self.dt = p_dt
        self.updated = False
        self.alpha = 1.
        self.urdf = p_model.urdf

        self.lam = 1.
        self.lam_max = 1000.

        self.constrain_u = p_constrain_u

    def set_nominal(self, p_states, p_inputs):
        self.nominal_states = p_states.copy()
        self.nominal_inputs = p_inputs.copy()

    def update(self, p_states, p_inputs):
        self.updated = True
        n_timesteps = p_states.shape[1]
        
        fx, fu, _ = autodiff(p_states, p_inputs, self.dt, self.urdf)
        _, lxf, lxxf = self.loss.get_final_cost(self.nominal_states)
        _, lu, luu, lx, lxx = self.loss.get_intermediate_costs(self.nominal_states, self.nominal_inputs)
        
        lx *= self.dt
        lxx *= self.dt
        lu *= self.dt
        luu *= self.dt
        
        # Do the backward pass to calculate the controls.
        vx = lxf
        vxx = lxxf

        for tx in xrange(n_timesteps-2, -1, -1):
            qx = lx[:, :, tx] + np.dot(fx[:, :, tx].T, vx)
            qu = lu[:, :, tx] + np.dot(fu[:, :, tx].T, vx)
            qxx = lxx[:, :, tx] + np.dot(fx[:, :, tx].T, np.dot(vxx, fx[:, :, tx]))
            quu = luu[:, :, tx] + np.dot(fu[:, :, tx].T, np.dot(vxx, fu[:, :, tx]))
            qux = np.dot(fu[:, :, tx].T, np.dot(vxx, fx[:, :, tx]))

            if self.constrain_u is None:
                quu_evals, quu_evecs = np.linalg.eig(quu)
                quu_evals += self.lam
                quu_inv = np.dot(quu_evecs, np.dot(np.diag(1. / quu_evals), quu_evecs.T))
                du = -np.dot(quu_inv, qu)
                self.K[:, :, tx] = -np.dot(quu_inv, qux)
            else:
                du = self._solve_qp(qu, quu, self.nominal_inputs[:, tx], self.constrain_u)

            # Set the open loop control.
            self.k[:, tx] = du

            vx = qx - np.dot(self.K[:, :, tx].T, np.dot(quu, self.k[:, tx:tx+1]))
            vxx = qxx - np.dot(self.K[:, :, tx].T, np.dot(quu, self.K[:, :, tx]))

    def _solve_qp(self, qu, quu, u_nom, limit):
        program = MathematicalProgram()
        du = program.NewContinuousVariables(1, 'du')
        program.AddLinearCost(du[0] * qu)
        program.AddQuadraticCost(0.5*du[0]**2*quu)
        program.AddLinearConstraint(du[0], -limit-u_nom, limit-u_nom)
        program.Solve()
        return program.GetSolution(du)

    def control_law(self, state, tx):
        if not self.updated:
            return np.random.uniform(-1, 1)

        du = self.alpha*self.k[:, tx] + np.dot(self.K[:, :, tx], state - self.nominal_states[:, tx])

        return self.nominal_inputs[:, tx] + du


class RandomController(object):

        def control_law(self, state, tx):
            return np.random.uniform(-100, 100)


class Acrobot(object):
    def __init__(self, p_urdf, p_q, p_r):
        self.urdf = p_urdf
        self.goal = np.array([np.pi, 0., 0., 0.])
        self.loss = QRLoss(p_q, p_r, self.goal)
        self.wrap_state = [0, 1]
        self.initial_state = np.array([0., 0., 0., 0.])
        self.initial_state = self.initial_state.reshape(self.initial_state.size, 1)


class CartPole(object):
    def __init__(self, p_urdf, p_q, p_r):
        self.urdf = p_urdf
        self.goal = np.array([0., np.pi, 0., 0.])
        self.loss = QRLoss(p_q, p_r, self.goal)
        self.wrap_state = [1]
        self.initial_state = np.array([0., 0., 0., 0.])
        self.initial_state = self.initial_state.reshape(self.initial_state.size, 1)


class Pendulum(object):
    def __init__(self, p_urdf, p_q, p_r):
        self.urdf = p_urdf
        self.goal = np.array([np.pi, 0.0])
        self.loss = QRLoss(p_q, p_r, self.goal)
        self.initial_state = np.array([0, 0.0])
        self.initial_state = self.initial_state.reshape((self.initial_state.size, 1))
        self.wrap_state = [0]


def get_cost(p_model, p_states, p_inputs, dt):
    f_cost, _, _ = p_model.loss.get_final_cost(p_states)
    i_costs, _, _, _, _ = p_model.loss.get_intermediate_costs(p_states, p_inputs)
    i_costs *= dt
    total_cost = f_cost + np.sum(i_costs)
    return total_cost, i_costs, f_cost


def save_initial_guesses(n_guesses=10):
    controller = RandomController()
    dt = 0.01
    duration = 3.0
    Q = 10. ** 4
    R = 0.01

    model = CartPole('cartpole.urdf', Q, R)

    guesses = []
    for ix in xrange(0, n_guesses):
        state_log, input_log = run_simulation(controller, model.initial_state, model.urdf, dt, duration, wrap_state=model.wrap_state)

        states, inputs = state_log.data(), input_log.data()
        guesses.append({'states':states, 'inputs':inputs})

    import pickle
    with open('initial_guesses.pkl', 'wb') as handle:
        pickle.dump(guesses, handle)



def simulate_single_run():
    dt = 0.01
    duration = 3.0
    timesteps = int(duration / dt) + 1
    n_linesearch = 50
    ix_linesearch = 0
    Q = 10. ** 4
    R = 1.

    # model = Pendulum('pendulum.urdf', Q, R)
    # model = CartPole('cartpole.urdf', Q, R)
    model = Acrobot('acrobot.urdf', Q, R)
    controller = iLQG(model, timesteps, dt, p_constrain_u=None)

    n_updates = 0

    max_lam = 200

    state_log, input_log = run_simulation(controller, model.initial_state, model.urdf, dt, duration, wrap_state=model.wrap_state)
    states, inputs = state_log.data(), input_log.data()
    cost, int_costs, final_cost = get_cost(model, states, inputs, dt)

    best_cost = cost

    state_records = []
    while n_updates < 50:
        # Get the costs.

        # Do an update and evaluate the results.
        controller.set_nominal(states, inputs)
        controller.update(states, inputs)
        tmp_state_log, tmp_input_log = run_simulation(controller, model.initial_state, model.urdf, dt, duration,
                                                     wrap_state=model.wrap_state)
        tmp_states, tmp_inputs = tmp_state_log.data(), tmp_input_log.data()
        cost, int_costs, final_cost = get_cost(model, tmp_states, tmp_inputs, dt)

        while cost > best_cost and ix_linesearch < n_linesearch and cost > 0.5:
            print 'Retrying', final_cost, np.sum(int_costs)
            # If if fails, increase regularization.
            if controller.lam > max_lam: break
            controller.lam *= 10
            controller.update(states, inputs)

            controller.alpha = 1.
            for ix in xrange(0, n_linesearch):
                tmp_state_log, tmp_input_log = run_simulation(controller, model.initial_state, model.urdf, dt, duration, wrap_state=model.wrap_state)
                tmp_states, tmp_inputs = tmp_state_log.data(), tmp_input_log.data()
                cost, int_costs, final_cost = get_cost(model, tmp_states, tmp_inputs, dt)
                if cost < best_cost:
                    break
                else:
                    print controller.alpha
                    controller.alpha *= 0.1

        controller.lam = 1
        state_log, input_log = tmp_state_log, tmp_input_log
        states, inputs = tmp_states, tmp_inputs
        state_records.append(states)

        print 'Succeeded (%d): %f, %f' % (n_updates, final_cost, np.sum(int_costs))
        n_updates += 1
        ix_linesearch = 0
        controller.alpha = 1.
        best_cost = cost

    visualizer = PlanarRigidBodyVisualizer(
                    RigidBodyTree(
                        model.urdf,
                        FloatingBaseType.kFixed
                    ),
                    xlim=[-5, 5],
                    ylim=[-2.5, 2.5]
                )
    visualizer.fig.set_size_inches(10, 5)
    animation = visualizer.animate(state_log, repeat=True)
    plt.show()


if __name__ == '__main__':

    #save_initial_guesses()

    simulate_single_run()
