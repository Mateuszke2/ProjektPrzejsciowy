import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
from functools import partial # reduces arguments to function by making some subset implicit
import flax
import os
import warnings
import imageio
import cv2
import pickle 
from jax.example_libraries import stax
from jax.example_libraries import optimizers
# from jax.experimental import stax
# from jax.experimental import optimizers

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from moviepy.editor import ImageSequenceClip
from functools import partial
import proglog
from PIL import Image

CONST_L1 = 1
CONST_L2 = 1
CONST_M1 = 1
CONST_M2 = 1
CONST_G = 9.8

from jax.lib import xla_bridge

# Definiwowanie układu podwójnego wahadła
print(xla_bridge.get_backend().platform)

def kinetic_energy(x_list, m1=CONST_M1, m2=CONST_M2, l1=CONST_L1, l2=CONST_L2, g=CONST_G):
    T1 = 0.5 * m1 * (l1 * x_list[:,2])**2  # Kinetyczna energia pierwszego wahadła
    T2 = 0.5 * m2 * ((l1 * x_list[:,2])**2 + (l2 * x_list[:,3])**2 + 2 * l1 * l2 * x_list[:,2] * x_list[:,3] * np.cos(x_list[:,0] - x_list[:,1]))  # Kinetyczna energia drugiego wahadła
    T = T1 + T2  # Suma kinetycznych energii obu wahadeł
    return T

def potential_energy(x_list, m1=CONST_M1, m2=CONST_M2, l1=CONST_L1, l2=CONST_L2, g=CONST_G):
    U1 = m1 * g * l1 * (1 - np.cos(x_list[:,0]))  # Potencjalna energia pierwszego wahadła
    U2 = m2 * g * (l1 * (1 - np.cos(x_list[:,0])) + l2 * (1 - np.cos(x_list[:,1])))  # Potencjalna energia drugiego wahadła
    U = U1 + U2  # Suma potencjalnych energii obu wahadeł
    return U


def lagrangian(q, q_dot, m1, m2, l1, l2, g = CONST_G):
  th1, th2 = q     # theta 1 and theta 2
  w1, w2 = q_dot # omega 1 and omega 2

  # kinetic energy (T)
  T1 = 0.5 * m1 * (l1 * w1)**2
  T2 = 0.5 * m2 * ((l1 * w1)**2 + (l2 * w2)**2 +
                    2 * l1 * l2 * w1 * w2 * jnp.cos(th1 - th2))
  T = T1 + T2
  
  # potential energy (V)
  y1 = -l1 * jnp.cos(th1)
  y2 = y1 - l2 * jnp.cos(th2)
  V = m1 * g * y1 + m2 * g * y2

  return (T - V)

# Analityczne rozwiązanie dla q_tt
# https://dassencio.org/33
def f_analytical(state, t=0, m1=CONST_M1, m2=CONST_M2, l1=CONST_L1, l2=CONST_L2, g=CONST_G):
  th1, th2, w1, w2 = state
  a1 = (l2 / l1) * (m2 / (m1 + m2)) * jnp.cos(th1 - th2)
  a2 = (l1 / l2) * jnp.cos(th1 - th2)
  f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * jnp.sin(th1 - th2) - \
      (g / l1) * jnp.sin(th1)
  f2 = (l1 / l2) * (w1**2) * jnp.sin(th1 - th2) - (g / l2) * jnp.sin(th2)
  g1 = (f1 - a1 * f2) / (1 - a1 * a2)
  g2 = (f2 - a2 * f1) / (1 - a1 * a2)

  return jnp.stack([w1, w2, g1, g2])

# Równanie ruchu
def equation_of_motion(lagrangian, state, t=None):
  print(state)
  q, q_t = jnp.split(state, 2)
  q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
          @ (jax.grad(lagrangian, 0)(q, q_t)
             - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
  print(q)
  print(q_t)
  print(q_tt)
  return jnp.concatenate([q_t, q_tt])

def solve_lagrangian(lagrangian, initial_state, **kwargs):
  @partial(jax.jit, backend='cpu')
  def f(initial_state):
    return odeint(partial(equation_of_motion, lagrangian),initial_state, **kwargs)
  return f(initial_state)

# Double pendulum dynamics via the rewritten Euler-Lagrange
@partial(jax.jit, backend='cpu')
def solve_autograd(initial_state, times, m1=CONST_M1, m2=CONST_M2, l1=CONST_L1, l2=CONST_L2, g=CONST_G):
  L = partial(lagrangian, m1=m1, m2=m2, l1=l1, l2=l2, g=g)
  return solve_lagrangian(L, initial_state, t=times, rtol=1e-10, atol=1e-10)

# Double pendulum dynamics via analytical forces taken from Diego's blog
@partial(jax.jit, backend='cpu')
def solve_analytical(initial_state, times, m1=CONST_M1, m2=CONST_M2, l1=CONST_L1, l2=CONST_L2, g=CONST_G):
  return odeint(partial(f_analytical, m1=m1, m2=m2, l1=l1, l2=l2, g=g), initial_state, t=times, rtol=1e-10, atol=1e-10)


def normalize_dp(state):
  return jnp.concatenate([(state[:2] + np.pi) % (2 * np.pi) - np.pi, state[2:]])

def rk4_step(f, x, t, h):
  # one step of runge-kutta integration
  k1 = h * f(x, t)
  k2 = h * f(x + k1/2, t + h/2)
  k3 = h * f(x + k2/2, t + h/2)
  k4 = h * f(x + k3, t + h)
  return x + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)

# choose an initial state
x0 = np.array([3*np.pi/7, 3*np.pi/4, 0, 0], dtype=np.float32)
noise = np.random.RandomState(0).randn(x0.size)
t = np.linspace(0, 40, num=401, dtype=np.float32)

# print(x0)
# print(noise)
# print(t)

# compute dynamics analytically
x_analytical = jax.device_get(solve_analytical(x0, t))
# print(x_analytical)
# print(x_analytical.shape)

x_autograd = jax.device_get(solve_autograd(x0, t))
# print(x_autograd)
# print(x_autograd.shape)

# PLOT PORÓWNANIE
# plt.title("Analytic vs Autograd")
# plt.xlabel("Time") ; plt.ylabel("State")
# plt.plot(t, x_analytical[:, 0], 'g-', label='$q$')
# plt.plot(t, x_analytical[:, 1], 'c-', label='$\dot q$')
# plt.plot(t, x_autograd[:, 0], 'g--', label='autograd $q$')
# plt.plot(t, x_autograd[:, 1], 'c--', label='autograd $\dot q$')
# plt.show()

# ANIMICJA

fig, ax = plt.subplots()

def make_plot(i, cart_coords, l1= CONST_L1, l2 = CONST_L2, max_trail=30, trail_segments=20, r = 0.05):
    # Plot and save an image of the double pendulum configuration for time step i.
    plt.cla()
    # plt.gca()
    # plt.figure(figsize=[6,6], dpi=120)
    # fig, ax = plt.subplots()

    x1, y1, x2, y2 = cart_coords
    ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k') # rods
    c0 = Circle((0, 0), r/2, fc='k', zorder=10) # anchor point
    c1 = Circle((x1[i], y1[i]), r, fc='b', ec='b', zorder=10) # mass 1
    c2 = Circle((x2[i], y2[i]), r, fc='r', ec='r', zorder=10) # mass 2
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)

    # plot the pendulum trail (ns = number of segments)
    # s = max_trail // trail_segments
    # for j in range(trail_segments):
    #     imin = i - (trail_segments-j)*s
    #     if imin < 0: continue
    #     imax = imin + s + 1
    #     alpha = (j/trail_segments)**2 # fade the trail into alpha
    #     ax.plot(x2[imin:imax], y2[imin:imax], c='r', solid_capstyle='butt',
    #             lw=2, alpha=alpha)

    # Center the image on the fixed anchor point. Make axes equal.
    ax.set_xlim(-l1-l2-r, l1+l2+r)
    ax.set_ylim(-l1-l2-r, l1+l2+r)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    # plt.savefig('./frames/_img{:04d}.png'.format(i//di), dpi=72)

def radial2cartesian(t1, t2, l1= CONST_L1, l2 = CONST_L2):
  # Convert from radial to Cartesian coordinates.
  x1 = l1 * np.sin(t1)
  y1 = -l1 * np.cos(t1)
  x2 = x1 + l2 * np.sin(t2)
  y2 = y1 - l2 * np.cos(t2)
  return x1, y1, x2, y2

def fig2image(fig):
  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return image

# Zapis sekwencji obrazów jako plik GIF
def SaveToGif(tablica_image, nazwa = "animacja.gif"):
  with imageio.get_writer(nazwa, mode='I') as writer:
      for image in tablica_image:
          writer.append_data(image)

# Funkcja do wyświetlania sekwencji obrazów
def display_images(images):
    for image in images:
        cv2.imshow('Frame', image)
        if cv2.waitKey(75) & 0xFF == ord('q'):  # Czekaj 25ms na klatkę, 'q' zamyka okno
            break

    cv2.destroyAllWindows()  # Zamyka okno po zakończeniu sekwencji

# theta1, theta2 = x_analytical[:, 0], x_analytical[:, 1]
def CreateImageSeq(motion_array, di = 1, N = 0, L1 = CONST_L1, L2 = CONST_L2):
  if N == 0:
     print(len(motion_array))
     N = len(motion_array)
  theta1, theta2 = motion_array[:, 0], motion_array[:, 1]
  cart_coords = radial2cartesian(theta1, theta2, L1, L2)
  images = []
  for i in range(0, N, di):
    print("{}/{}".format(i // di, N // di), end='\n' if i//di%20==0 else ' ')
    make_plot(i, cart_coords, L1, L2)
    images.append( fig2image(fig) )
  return images

# PLOT LOSS
def PlotLoss(train_losses, test_losses):
  plt.figure(figsize=(8, 3.5), dpi=120)
  plt.plot(train_losses, label='Train loss')
  plt.plot(test_losses, label='Test loss')
  plt.yscale('log')
  plt.ylim(None, 200)
  plt.title('Losses over training')
  plt.xlabel("Train step") ; plt.ylabel("Mean squared error")
  plt.legend() ; plt.show()

# DANE DO TRENINGU WYGENEROWANE PRZEZ METODE ANALITYCZNĄ

time_step = 0.01
N = 3000
prc = 0.5

analytical_step = jax.jit(jax.vmap(partial(rk4_step, f_analytical, t=0.0, h=time_step)))

t = np.arange(N, dtype=np.float32) # time steps 0 to N
x = jax.device_get(solve_analytical(x0, t)) # dynamics for first N time steps
xt = jax.device_get(jax.vmap(f_analytical)(x)) # time derivatives of each state
y = jax.device_get(analytical_step(x)) # analytical next step
print(t)
print(x)
print(xt)
# podzial
t_train = t[:int(prc*len(t))]# time steps 0 to prc * len(t)
x_train = x[:int(prc*len(x)),:]
xt_train = xt[:int(prc*len(x)),:]
y_train = y[:int(prc*len(x)),:]

t_test = t[int(prc*len(t)):]# time steps 0 to prc * len(t)
x_test = x[int(prc*len(x)):,:]
xt_test = xt[int(prc*len(x)):,:]
y_test = y[int(prc*len(x)):,:]

print(t_test.shape)

print(x_test.shape)

# print(t_train[-1])
# print(t_test[0])

# MODEL SIECI LAGRANGE'A

# replace the lagrangian with a parameteric model
def learned_lagrangian(params):
  def lagrangian(q, q_t):
    assert q.shape == (2,)
    state = normalize_dp(jnp.concatenate([q, q_t]))
    return jnp.squeeze(nn_forward_fn(params, state), axis=-1)
  return lagrangian

# Funkcja strat oparta na porównaniu wartosci na wyuczonym L(preds) z rozw. analitycznm (targets)
@jax.jit
def loss(params, batch, time_step=None):
  state, targets = batch
  if time_step is not None:
    f = partial(equation_of_motion, learned_lagrangian(params))
    preds = jax.vmap(partial(rk4_step, f, t=0.0, h=time_step))(state)
  else:
    preds = jax.vmap(partial(equation_of_motion, learned_lagrangian(params)))(state)
  return jnp.mean((preds - targets) ** 2)

# Model sieci
init_random_params, nn_forward_fn = stax.serial(
    stax.Dense(128),
    stax.Softplus,
    stax.Dense(128),
    stax.Softplus,
    stax.Dense(1),
)

@jax.jit
def update_timestep(i, opt_state, batch):
  params = get_params(opt_state)
  return opt_update(i, jax.grad(loss)(params, batch, time_step), opt_state)

@jax.jit
def update_derivative(i, opt_state, batch):
  params = get_params(opt_state)
  return opt_update(i, jax.grad(loss)(params, batch, None), opt_state)

x_train = jax.device_put(jax.vmap(normalize_dp)(x_train))
y_train = jax.device_put(y_train)

x_test = jax.device_put(jax.vmap(normalize_dp)(x_test))
y_test = jax.device_put(y_test)


rng = jax.random.PRNGKey(0)
_, init_params = init_random_params(rng, (-1, 4))


batch_size = 100
test_every = 10
num_batches = 1500

train_losses = []
test_losses = []

# Adaptacyjny LR
opt_init, opt_update, get_params = optimizers.adam(
    lambda t: jnp.select([t < batch_size*(num_batches//3),
                          t < batch_size*(2*num_batches//3),
                          t > batch_size*(2*num_batches//3)],
                         [1e-3, 3e-4, 1e-4]))
opt_state = opt_init(init_params)

# # UCZENIE

import time
start_time = 0
for iteration in range(batch_size*num_batches + 1):
  if iteration % batch_size == 0:
    params = get_params(opt_state)
    train_loss = loss(params, (x_train, xt_train))
    train_losses.append(train_loss)
    test_loss = loss(params, (x_test, xt_test))
    test_losses.append(test_loss)
    if iteration % (batch_size*test_every) == 0:
      end_time = time.time()
      print(f"iteration={iteration}, train_loss={train_loss:.66f}, test_loss={test_loss:.6f}")
      print(end_time - start_time)
      start_time = time.time()
  opt_state = update_derivative(iteration, opt_state, (x_train, xt_train))

# params = get_params(opt_state)

# ZAPIS PARAMETROW
# def create_directory(path):
#     if not os.path.exists(path):
#         try:
#             os.makedirs(path)
#         except OSError as e:
#             print(f"Błąd podczas tworzenia katalogu: {e}")


# directory_path = "testy/model_{}_{}_{}/".format(batch_size,num_batches,test_every)
# create_directory(directory_path)

# params_path = "testy/model_{}_{}_{}/learned_model_params.pkl".format(batch_size,num_batches,test_every)
# train_losses_path = "testy/model_{}_{}_{}/train_losses.pkl".format(batch_size,num_batches,test_every)
# test_losses_path = "testy/model_{}_{}_{}/test_losses.pkl".format(batch_size,num_batches,test_every)

# try:
#   with open(params_path, 'wb') as file:
#       pickle.dump(params, file)
#   with open(train_losses_path, 'wb') as file:
#       pickle.dump(train_losses, file)
#   with open(test_losses_path, 'wb') as file:
#       pickle.dump(test_losses, file)
# except Exception as e:
#     print(f"Błąd podczas zapisywania pliku: {e}")
# finally:
#    print("Koniec zapisu")


# PLOT LOSS

# plt.figure(figsize=(8, 3.5), dpi=120)
# plt.plot(train_losses, label='Train loss')
# plt.plot(test_losses, label='Test loss')
# plt.yscale('log')
# plt.ylim(None, 200)
# plt.title('Losses over training')
# plt.xlabel("Train step") ; plt.ylabel("Mean squared error")
# plt.legend() ; plt.show()

# PLOT POROWNANIE
# xt_pred = jax.vmap(partial(equation_of_motion, learned_lagrangian(params)))(x_test)
# print(xt_pred)
# fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=120)
# axes[0].scatter(xt_test[:, 2], xt_pred[:, 2], s=6, alpha=0.2)
# axes[0].set_title('Predicting $\dot q$')
# axes[0].set_xlabel('$\dot q$ actual')
# axes[0].set_ylabel('$\dot q$ predicted')
# axes[1].scatter(xt_test[:, 3], xt_pred[:, 3], s=6, alpha=0.2)
# axes[1].set_title('Predicting $\ddot q$')
# axes[1].set_xlabel('$\ddot q$ actual')
# axes[1].set_ylabel('$\ddot q$ predicted')
# plt.tight_layout()
# plt.show()
## KONIEC UCZENIA



# # WCZYTANIE

try:
  with open(r'D:\Users\User\Desktop\Studia\MGR\SEM2\PPrzejsciowy\testy\model_100_1500_10\learned_model_params.pkl', 'rb') as f:
      params = pickle.load(f)
  with open(r'D:\Users\User\Desktop\Studia\MGR\SEM2\PPrzejsciowy\testy\model_100_1500_10\test_losses.pkl', 'rb') as f:
      test_losses = pickle.load(f)    
  with open(r'D:\Users\User\Desktop\Studia\MGR\SEM2\PPrzejsciowy\testy\model_100_1500_10\train_losses.pkl', 'rb') as f:
      train_losses = pickle.load(f)
except Exception as e:
    print(f"Błąd podczas wczytania pliku: {e}")

# War pocz
x1 = np.array([3*np.pi/7, 3*np.pi/4, 0, 0], dtype=np.float32)
t2 = np.linspace(0, 20, num=200)
x1_model = jax.device_get(solve_lagrangian(learned_lagrangian(params), x1, t=t2))
print("\nModel:")
print(x1_model)
x1_analytical = jax.device_get(solve_analytical(x1, t2))
print("\nAnalitycznie:")
print(x1_analytical)

mae = jnp.mean(jnp.abs(x1_model - x1_analytical))
print(f"Średni błąd Bezwzględny (MAE): {mae}")

mse = np.mean((x1_model - x1_analytical)**2)
print(f"Błąd Średniokwadratowy (MSE): {mse}")

# ENERGIE

Ek_model = kinetic_energy(x1_model)
Ep_model = potential_energy(x1_model)
Ek_an = kinetic_energy(x1_analytical)
Ep_an = potential_energy(x1_analytical)
print(Ek_model)
print(Ep_model)
print(Ek_an)
print(Ep_an)

E_model_sum = Ek_model + Ep_model
E_an_sum = Ek_an + Ep_an

plt.figure(figsize=(8, 3.5), dpi=120)
plt.plot(t2,E_model_sum, label='Energia predykowanego modelu')
plt.plot(t2, E_an_sum, label='Energia rozwiązania analitycznego')
plt.title('Wykres sum energii')
plt.xlabel("Czas") ; plt.ylabel("Wartość")
plt.legend() ; plt.show()

plt.figure(figsize=(8, 3.5), dpi=120)
plt.plot(t2,Ek_an, label='Energia kinetyczna analitycznie',color = 'green')
plt.plot(t2,Ep_an, label='Energia potencjalna analitycznie',color = 'blue')
plt.title('Wykres energii wyznaczonych analitycznie')
plt.xlabel("Czas") ; plt.ylabel("Wartość")
plt.legend() ; plt.show()

# PORÓWNANIE q

plt.figure(figsize=(12, 6), dpi=120)

# Subplot dla q_1
plt.subplot(2, 1, 1)
plt.plot(t2, x1_model[:, 1], label='Model $q_2$')
plt.plot(t2, x1_analytical[:, 1], label='Analityczne $q_2$')
plt.title('Porównanie wyników dla $q_2$')
plt.xlabel("Czas")
plt.ylabel("Wartość")
plt.legend()

# Subplot dla \dot{q_1}
plt.subplot(2, 1, 2)
plt.plot(t2, x1_model[:, 3], label='Model $\dot{q_2}$')
plt.plot(t2, x1_analytical[:, 3], label='Analityczne $\dot{q_2}$')
plt.title('Porównanie wyników dla $\dot{q_2}$')
plt.xlabel("Czas")
plt.ylabel("Wartość")
plt.legend()

# Wyświetlanie wykresu
plt.tight_layout()  # Poprawia układ subpłotów
plt.show()


# images_model = CreateImageSeq(x1_model,di = 1, N = 0, L1 = CONST_L1, L2 = CONST_L2)
# display_images(images_model)
# SaveToGif(images_model,"animacja_model.gif")
# images_analytical = CreateImageSeq(x1_analytical,di = 1, N = 0, L1 = CONST_L1, L2 = CONST_L2)
# display_images(images_analytical)
# SaveToGif(images_model,"animacja_analytical.gif")

# PlotLoss(train_losses,test_losses)



