/*
    These auxillary functions are used throughout mod.rs for the bits of calculation for different uses. 

    CITE: Maass 2002 paper pg. 18, connectivity of neurons.
    CITE: Equation 21 second order dynamics in Texas ANM
*/

#![allow(dead_code)]

pub fn dist(
    (x1, y1, z1): &(f32, f32, f32), // point 1
    (x2, y2, z2): &(f32, f32, f32), // point 2
) -> f32 {
    // Finds the Euclidean Distance between 2 3D points \\
    ((x2 - x1).powf(2.) + (y2 - y1).powf(2.) + (z2 - z1).powf(2.)).sqrt()
}

pub fn connect_chance(
    d_ab: f32,   // The distance between neurons a and b
    c: f32,      // A hyper parameter. At very close distances, the probability output is c.
    lambda: f32, // A hyper parameter. At larger values of lambda, the decay slows.
) -> f32 {
    // Computes the probability that two neurons are connected based on the     \\
    // distance between neurons a and b, and two hyper parameters C and lambda. \\

    let exponent: f32 = -1. * ((d_ab / lambda).powf(2.));
    c * exponent.exp()
}

pub fn delta(n: i32) -> f32 {
    // Kronecker / Dirac delta function
    // A spike helper function
    // It outputs 1 only at 0 (Dirac outputs infinity at 0)
    if n == 0 {
        return 1.;
    }
    0.
}

pub fn heaviside(n: i32) -> f32 {
    // Heaviside step function
    // When n < 0, H(n) = 0
    //      n > 0, H(n) = 1
    //      n = 0, H is undefined, but we put 1
    if n < 0 {
        return 0.;
    }
    1.
}

pub fn liq_response(
    model: &str,    // One of ["static", "first order", "second order"]
    curr_t: i32,    // Time step at which we are calculating the change in voltage
    t_spike: i32,   // Time at which a presynaptic neuron spiked
    delay: i32,     // Delay from voltage dynamics equation
    first_tau: u32, // If "first order" model, then this is its hyper parameter
    tau_s1: u32,    // If "second order" model, then the next two are hyper parameters
    tau_s2: u32,
) -> f32 {
    // Implementation of the S function in voltage dynamics for each model. 
    if model == "static" {
        return delta(curr_t - t_spike - delay);
    } else if model == "first order" {
        let exponent = ((curr_t - t_spike - delay) as f32 * -1.) / (first_tau as f32);
        return (1. / first_tau as f32) * f32::exp(exponent) * heaviside(curr_t - t_spike - delay);
    } else if model == "second order" {
        let exponent1: f32 = ((curr_t - t_spike - delay) as f32 * -1.) / (tau_s1 as f32);
        let exponent2: f32 = ((curr_t - t_spike - delay) as f32 * -1.) / (tau_s2 as f32);
        let denominator: f32 = tau_s1 as f32 - tau_s2 as f32;
        let h = heaviside(curr_t - t_spike - delay);
        let part1 = f32::exp(exponent1);
        let part2 = f32::exp(exponent2);
        return (part1 - part2) * h / denominator;
    }
    panic!("Model was chosen incorrectly"); // instead of asserting the model is one of three strings
}

pub fn string_to_seed(word: &str) -> [u8; 32] {
    // Takes a string (usually the LSM's guess for an epoch), and converts it into
    // a seed for a random number generator. 
    
    // Notice that a seed in Rust, in StdRng package, is a 32-length array of u8s. 
    // At maximum, u8's can be 512. This function converts the string
    // into ASCII, concatentates them, and then splits them by two digit strings into the seed. 
    // It isn't perfect because of zeros, and we could try include numbers less than 512, but it works
    // up to a certain bound of word length.   

    let mut seed = [0_u8; 32]; // we will fill this and return it

    // Converts string into a list of its ASCII values per character
    let words = word.as_bytes();
    // Converts those bytes into strings and collects them into a vector, not an array
    let words: Vec<String> = words.iter().map(|bit| bit.to_string()).collect();
    let seed_str = words.concat(); // Concatenates all strings in a vector
    let seed_chars: Vec<char> = seed_str.chars().collect(); // Converts each to a character

    assert_eq!(seed_str.len() < 65, true); // arbitrary bound from observation
    let mut mini_seed: String; 

    // Loop through characters by 2s and fill the seed parsed to u8s. 
    for char_idx in (0..seed_str.len() - 1).step_by(2) {
        mini_seed = [
            seed_chars[char_idx].to_string(),
            seed_chars[char_idx + 1].to_string(),
        ]
        .concat();
        seed[char_idx / 2] = mini_seed.parse().unwrap();
    }
    seed
}
