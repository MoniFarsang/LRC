import time
import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from neuralODE.ode_model import ODEFunc, NODEFunc

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--model', type=str, choices=['lrc','neural_ode'], default='lrc')
parser.add_argument('--lrc_type', type=str, choices=['symmetric','asymmetric','no'], default='symmetric')
parser.add_argument('--data', type=str, choices=['periodic_sinusodial', 'spiral', 'duffing', 'periodic_predator_prey', 'limited_predator_prey', 'nonlinear_predator_prey'], default='spiral')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--viz', type=bool, default=False)
parser.add_argument('--units', type=int, default=16)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--id', type=int, default=1)

args = parser.parse_args()

from tfdiffeq import odeint
tf.keras.backend.set_floatx('float32')

device = 'gpu:' + str(args.gpu) if tf.test.is_gpu_available() else 'cpu:0'

class PeriodicSinusodial(tf.keras.Model):
# Periodic sinusoidal equations
  def call(self, t, v):
    x, y = tf.unstack(v[0])
    xy = tf.sqrt(x * x + y * y)
    dx_dt = x * (1 - xy) - y
    dy_dt = x + y * (1 - xy)
    
    return tf.stack([dx_dt, dy_dt])
  
class Spiral(tf.keras.Model):
# Spiral equation
    def __init__(self, A):
        super().__init__()
        self.A = A

    def call(self, t, y):
        return tf.matmul(y, self.A)
    
class DuffingOscilator(tf.keras.Model):
# Duffing oscillator equations
  def __init__(self):
    super().__init__()

  def call(self, t, x):
    y, v = tf.unstack(x[0])
    dy_dt = v
    dv_dt = y - tf.pow(y, 3)

    return tf.stack([dy_dt, dv_dt])

class PredatorPrey(tf.keras.Model):
# Periodic Lotka-Volterra equations
  def __init__(self, a, b, c, d):
    super().__init__()
    self.a, self.b, self.c, self.d = a, b, c, d
    
  def call(self, t, y):
    r, f = tf.unstack(y[0])
    dR_dT = self.a * r - self.b * r * f
    dF_dT = -self.c * f + self.d * r * f
    
    return tf.stack([dR_dT, dF_dT])
  
class LimitedPredatorPrey(tf.keras.Model):
# Asymptotic Lotka-Volterra equations
  def __init__(self, d, **kwargs):
    super().__init__(**kwargs)
    self.d = d
    
  def call(self, t, y):
    r, f = tf.unstack(y[0])
    dR_dT = r * (1. - r) - r * f
    dF_dT = -f + self.d * r * f
    
    return tf.stack([dR_dT, dF_dT])
  
class NonLinearSystemPredatorPrey(tf.keras.Model):
# Nonlinear Lotka-Volterra equations
  def __init__(self, A):
    super().__init__()
    self.A = A
  
  def call(self, t, v):
    x, y = tf.unstack(v[0])
    dx_dt = x * (1 - x) + self.A * x * y
    dy_dt = y * (1 - y) + x * y
    
    return tf.stack([dx_dt, dy_dt])
  
# Initial value for the ODE problem
ode_params = {
    'periodic_sinusodial': {
        'true_y0': tf.convert_to_tensor([[1, 1]], dtype=tf.float64),
        't': tf.linspace(0., 10., args.data_size)
    },
    'spiral': {
        'true_y0': tf.convert_to_tensor([[0.5, 0.01]], dtype=tf.float64),
        't': tf.linspace(0., 25, args.data_size)
    },
    'duffing': {
        'true_y0': tf.convert_to_tensor([[-1, 1]], dtype=tf.float64),
        't': tf.linspace(0., 25, args.data_size),
    },
    'periodic_predator_prey': {
        'true_y0': tf.convert_to_tensor([[1, 1]], dtype=tf.float64),
        't': tf.linspace(0., 10., args.data_size)
    },
    'limited_predator_prey': { #limited version
        'true_y0': tf.convert_to_tensor([[1, 1]], dtype=tf.float64),
        't': tf.linspace(0., 20., args.data_size)
    },
    'nonlinear_predator_prey': {
        'true_y0': tf.convert_to_tensor([[2, 1]], dtype=tf.float64),
        't': tf.linspace(0., 20., args.data_size)
    },
}

if args.data == 'periodic_sinusodial':
  true_y = odeint(PeriodicSinusodial(), ode_params['periodic_sinusodial']['true_y0'], ode_params['periodic_sinusodial']['t'], method='dopri5')
elif args.data == 'spiral':
  true_y = odeint(Spiral(tf.convert_to_tensor([[-0.1, 3.0], [-3.0, -0.1]], dtype=tf.float64)), ode_params['spiral']['true_y0'], ode_params['spiral']['t'], method='dopri5')
elif args.data == 'duffing':
  true_y = odeint(DuffingOscilator(), ode_params['duffing']['true_y0'], ode_params['duffing']['t'], method='dopri5')
elif args.data == 'periodic_predator_prey':
  true_y = odeint(PredatorPrey(1.5, 1, 3, 1), ode_params['periodic_predator_prey']['true_y0'], ode_params['periodic_predator_prey']['t'], method='dopri5')
elif args.data == 'limited_predator_prey':
    true_y = odeint(LimitedPredatorPrey(2), ode_params['limited_predator_prey']['true_y0'], ode_params['limited_predator_prey']['t'], method='dopri5')
elif args.data == 'nonlinear_predator_prey':
  true_y = odeint(NonLinearSystemPredatorPrey(0.33), ode_params['nonlinear_predator_prey']['true_y0'], ode_params['nonlinear_predator_prey']['t'], method='dopri5')

def get_batch(true_y0, t):
  s = np.random.choice(
      np.arange(args.data_size - args.batch_time,
                dtype=np.int64), args.batch_size,
      replace=False)

  temp_y = true_y.numpy()
  batch_y0 = tf.convert_to_tensor(temp_y[s])  # (M, D)
  batch_t = t[:args.batch_time]  # (T)
  batch_y = tf.stack([temp_y[s + i] for i in range(args.batch_time)], axis=0)  # (T, M, D)
  return batch_y0, batch_t, batch_y

folder_name = f'traj_phase_{args.data}_{args.model}_{args.lrc_type}_{args.id}'
os.makedirs(folder_name, exist_ok=True)
max_y, min_y = true_y.numpy().max(), true_y.numpy().min()
fig = plt.figure(figsize=(8, 4), facecolor='white')

# Visualize the trajectories and phase portraits
def visualize(true_y, pred_y, itr):
    plt.clf()
    t = ode_params[args.data]['t']
    ax_traj = fig.add_subplot(121, frameon=False)
    ax_phase = fig.add_subplot(122, frameon=False)

    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x,y')
    ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-', label='True trajectories')
    ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 0, 1], 'b--', label='Predicted Trajectories')
    ax_traj.set_xlim(min(t.numpy()), max(t.numpy()))
    ax_traj.set_ylim(min_y, max_y+0.1)
    ax_traj.legend()

    ax_phase.set_title('Phase Portrait')
    ax_phase.set_xlabel('x')
    ax_phase.set_ylabel('y')
    ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
    ax_phase.plot(pred_y.numpy()[:, 0, 0, 0], pred_y.numpy()[:, 0, 0, 1], 'b--')
    ax_phase.set_xlim(min_y-0.075, max_y+0.075)
    ax_phase.set_ylim(min_y-0.075, max_y+0.075)

    fig.tight_layout()
    plt.savefig(folder_name + '/png_{:03d}'.format(itr), dpi=200)

ii = 0 # iteration number for visualization

folder_logs = "results"
os.makedirs(folder_logs, exist_ok=True)

with tf.device(device):
  if args.model == 'neural_ode':
    func = NODEFunc(input_shape = ode_params[args.data]['true_y0'].shape)
    f = open(f"{folder_logs}/{args.model}_{args.data}_train_id{args.id}.txt", "w+")
    g = open(f"{folder_logs}/{args.model}_{args.data}_test_id{args.id}.txt", "w+")
  elif args.model == 'lrc':
    func = ODEFunc(input_shape = ode_params[args.data]['true_y0'].shape, elastance = args.lrc_type, units=args.units)
    f = open(f"{folder_logs}/{args.model}_{args.lrc_type}_{args.units}_{args.data}_train_id{args.id}.txt", "w+")
    g = open(f"{folder_logs}/{args.model}_{args.lrc_type}_{args.units}_{args.data}_test_id{args.id}.txt", "w+")

  lr = 1e-3
  optimizer = tf.keras.optimizers.Adam(lr)

  print('Training ODE with data:', args.data)

  end = time.time()

  for itr in range(1, args.niters + 1):

      with tf.GradientTape() as tape:
          batch_y0, batch_t, batch_y = get_batch(ode_params[args.data]['true_y0'], ode_params[args.data]['t'])
          if args.model == 'neural_ode':
            pred_y = odeint(func, tf.cast(batch_y0, tf.float32), batch_t, method='dopri5')
          else:
            pred_y = odeint(func, tf.cast(batch_y0, tf.float32), batch_t, method='euler')
          loss = tf.reduce_mean(tf.abs(pred_y - tf.cast(batch_y, tf.float32)))

      grads = tape.gradient(loss, func.variables)
      grad_vars = zip(grads, func.variables)

      optimizer.apply_gradients(grad_vars)

      f.write(f"Iteration {itr}\n")
      f.write(f"Loss: {loss.numpy()}\n")
      f.write(f"Time: {time.time() - end}\n")
      f.write("\n")

      if itr % 10 == 0:
          pred_y = odeint(func, tf.cast(tf.expand_dims(ode_params[args.data]['true_y0'], axis=0), tf.float32), ode_params[args.data]['t'], method='euler')
          loss = tf.reduce_mean(tf.abs(pred_y[:,0,0,:] - tf.cast(true_y[:,0,:], tf.float32)))
          print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.numpy()))
          g.write(f"Iteration {itr}\n")
          g.write(f"Loss: {loss.numpy()}\n")
          g.write("\n")
          if args.viz:
            # if itr > 4000:
            visualize(true_y, pred_y, ii)
            ii += 1

      end = time.time()

  f.close()
  g.close()