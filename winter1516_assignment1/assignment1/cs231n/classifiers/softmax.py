import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]
  for i in range(num_train):
    scores = np.dot(X[i], W)
    scores_exp = np.exp(scores)
    prob = scores_exp[y[i]] / np.sum(scores_exp)
    loss += -np.log(prob)
    dW[:, y[i]] -= X[i]
    for j in range(W.shape[1]):
      dW[:, j] += scores_exp[j]/ np.sum(scores_exp) * X[i]
  loss = loss / num_train + 0.5 * reg * np.sum(W * W)
  dW = dW / num_train + reg * W
  """
  N, C = X.shape[0], W.shape[1]
  for i in range(N):
    f = np.dot(X[i], W)
    f -= np.max(f)  # f.shape = C
    loss = loss + np.log(np.sum(np.exp(f))) - f[y[i]]
    dW[:, y[i]] -= X[i]
    s = np.exp(f).sum()
    for j in range(C):
      dW[:, j] += np.exp(f[j]) / s * X[i]
  loss = loss / N + 0.5 * reg * np.sum(W * W)
  dW = dW / N + reg * W
  """
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = np.dot(X, W) # 500, 10
  scores -= np.reshape(np.max(scores, axis=1), (num_train, -1)) # scale to numercial stable
  scores_exp = np.exp(scores) # 500, 10
  prob = np.reshape(scores_exp[range(num_train), y], (num_train, -1)) / np.reshape((np.sum(scores_exp, axis=1)), (X.shape[0], -1)) # 500, 1
  loss = -np.sum(np.log(prob))
  loss = loss / num_train + 0.5 * reg * np.sum(W * W)

  dW = np.dot(X.T, scores_exp / np.reshape((np.sum(scores_exp, axis=1)), (X.shape[0], -1)))
  const = np.zeros(scores_exp.shape)
  const[range(num_train), y] = 1
  dW -= np.dot(X.T, const)
  dW = dW / num_train + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

