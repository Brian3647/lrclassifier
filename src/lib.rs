#![forbid(unsafe_code)]
#![forbid(missing_docs)]
#![forbid(clippy::all)]
#![doc = include_str!("../README.md")]

use rand::Rng;

/// The sigmoid function.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Logistic Regression Classifier
///
/// Logistic regression is a simple but effective classification algorithm
/// that is used for binary classification problems (i.e. problems with two
/// classes). It is a linear model that uses the logistic function to
/// predict the probability of a binary response.
///
/// It shines specially when the data is linearly separable (when
/// there is a linear boundary that separates the two classes). An example
/// of this is separation of positive and negative numbers.
pub struct LRClassifier {
    /// The weights of the classifier.
    pub weights: Vec<f64>,
    /// The learning rate of the classifier.
    pub learning_rate: f64,
    /// Lambda parameter for L2 regularization.
    pub lambda: f64,
}

impl LRClassifier {
    /// Creates a new `LRClassifier` with default parameters.
    pub fn new(input_size: usize, learning_rate: f64, lambda: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect();

        Self {
            weights,
            learning_rate,
            lambda,
        }
    }

    /// Predicts the output for a given input.
    pub fn predict(&self, input: &[f64]) -> f64 {
        let linear_model = input.iter().zip(&self.weights).map(|(x, w)| x * w).sum();
        sigmoid(linear_model)
    }

    /// Trains the classifier once with the given inputs and outputs.
    pub fn train_once(&mut self, inputs: &[Vec<f64>], outputs: &[f64]) {
        for (input, &output) in inputs.iter().zip(outputs) {
            let prediction = self.predict(input);
            let error = prediction - output;

            for (j, weight) in self.weights.iter_mut().enumerate() {
                let l2r = self.lambda * *weight;
                *weight -= self.learning_rate * (error * input[j] + l2r);
            }
        }
    }

    /// Trains the classifier with the given inputs and outputs for the given number of epochs.
    pub fn train(&mut self, inputs: &[Vec<f64>], outputs: &[f64], epochs: usize) {
        for _ in 0..epochs {
            self.train_once(inputs, outputs);
        }
    }
}
