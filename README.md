# neuralNet

Api-like code for creating neural nets. Run using python 2.7

INFORMATION ABOUT MODEL:

Uses sigmoid acitivation function (sigmoid for RELU as well).
Can compute neg log loss.
Computes least squared for loss function.
Computes modified batch gradient descent (data points not randomly selected)

FRAMEWORK:

1. The csv file that holds your data should have data points as rows and features as columns, solution data should be at the end. Be sure to modify the slicing within dataCleaning.py so that all features are read in and that all solutions are read in.
2. Use the sample given to construct and train a nNet
      1. declare a net:
      2. first param: depth of the net
      3. second param: batch of training data points to pass through net
      4. third param: row dimensions of weights per layer (at min 2 inputs)
      5. fourth param: col dimensions of weights per layer (at min 2 inputs)
3. Train the net over all your specified "batches" of data (second param passed into net)
      1. for one section of data points, or batch:
      2. run <net>.readInputs(instance) (forward prop)
      3. run <net>.Optimize() (back prop and grad descent)
      4. extrapolate from NLLScore usage in sample to print neg log likelihood score
4. Adjust the step as you see fit
