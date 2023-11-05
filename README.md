# lrclassifier: Logistic Regression Classifiers in rust

`lrclassifier` is a Rust library that provides an implementation of the Logistic Regression Classifier, a popular machine learning algorithm for binary classification problems.

## Features

- Efficient implementation of Logistic Regression algorithm.
- Supports L2 regularization.
- Customizable learning rate and regularization parameter.
- Easy-to-use API for training models and making predictions.

## Quick start

This is the code for a sample positive-negative number classification.

```toml
# Cargo.toml

[dependencies]
lrclassifier = "*"
```

```rust
// main.rs
use lrclassifier::LRClassifier;

fn main() {
    // Data: positive and negative numbers
    let inputs: Vec<Vec<f64>> = vec![
        vec![-3.0],
        vec![-2.0],
        vec![-1.0],
        vec![1.0],
        vec![2.0],
        vec![3.0],
    ];

    let expected: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 0 for negative, 1 for positive

    let learning_rate = 0.1;
    let lambda = 0.1;
    let input_size = inputs[0].len();

    // Create a classifier
    let mut classifier = LRClassifier::new(input_size, learning_rate, lambda);

    // Train the classifier
    classifier.train(&inputs, &expected, 10000);

    // Test the classifier
    for (input, output) in inputs.iter().zip(expected) {
        let prediction = dbg!(classifier.predict(input));
        let binary_prediction = if prediction >= 0.5 { 1.0 } else { 0.0 };
        assert_eq!(binary_prediction, output);
    }
}
```

## License

This project is licensed under the MIT license that can be found [here](./LICENSE)
