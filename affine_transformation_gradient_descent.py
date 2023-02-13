import numpy as np
import tensorflow as tf
from stn import spatial_transformer_network as spatial_transformer

def affine_transformation_gradient_descent(
    x,
    y, 
    id_ = None, 
    opt_name = 'Adam', 
    lr = 1e-1, 
    max_steps = 1000, 
    patience = 20, 
    epsilon = 1e-3, 
    save_theta_array = False
):
    '''
    Let S_θ be the affine transformation specified by the transformation matrix: 
    θ = [ [ʎ, 0, tx],
          [0, ʎ, ty] ]
          
    This function uses gradient descent to solve the following optimization problem:
    min_{ʎ,tx,ty}[MSE(S_θ(x),y)]
    
    Args:
    x (tf.Tensor): input tensor to be transformed with shape=(n, height, width, channels)
    y (tf.Tensor): target tensor with shape=(n, height, width, channels)
    id_ (str): name of experiment
    opt_name (str): optimizer to be used
    lr (float): learning rate for optimizer
    max_steps (int): maximum number of steps for gradient descent.
    patience(int): number of steps used to determine if the loss is not improving.
    epsilon(float): tolerance
    save_theta_array(bool): If True save θ at every iteration of GD
    '''
    loss_array = []
    theta = tf.Variable([1., 0., 0., 0., 1., 0.]) # start with identity transformation
    if save_theta_array == True:
        theta_array = [list(theta.numpy())]
    opt = getattr(tf.keras.optimizers, opt_name)(learning_rate=lr)
    for step in range(max_steps):
        # record operations inside tape to allow auto-diff
        with tf.GradientTape() as tape:
            pred = spatial_transformer(x, theta) # apply affine transformation to x
            loss = tf.reduce_sum(tf.metrics.mse(y, pred)) # compute loss
        loss_array.append(loss.numpy())
        grads = tape.gradient(loss, theta) # get gradient of theta with respect to the loss
        processed_grads = []
        for i, g in enumerate(grads):
            if i == 1 or i == 3: # Don't update off-diagonal entries of rotation matrix
                processed_grads.append(g * 0)
            elif i == 4: 
                processed_grads.append(grads[0]) # Isotropic Scaling.
            else:
                processed_grads.append(g) 
        grads_and_vars = zip([processed_grads], [theta])
        opt.apply_gradients(grads_and_vars) # update theta using the gradient
        if save_theta_array == True:
            theta_array.append(list(theta.numpy()))
        # Stop gradient descent if the loss is not improving
        if step > patience + 1:
            if np.abs(np.mean(loss_array[-patience:]) - loss_array[-1]) < epsilon:
                break

    output_dict = {}
    output_dict['id'] = id_
    output_dict['optimizer'] = opt_name
    output_dict['lr'] = lr
    output_dict['loss_array'] = loss_array
    output_dict['final_loss'] = loss_array[-1]
    output_dict['theta'] = list(theta.numpy())
    if save_theta_array == True:
         output_dict['theta_array'] = theta_array
    return output_dict