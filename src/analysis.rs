use std::fs::File;
use std::io::Write;

use nalgebra::base::DMatrix;

pub fn analysis(
    connections: &DMatrix<u32>, // The connections matrix made of 0's and 1's. 1 - connection between the indexed neurons, 0 - no connection
    dists: &Vec<f32>,           // All edge distances
    n: usize,                   // The number of neurons in a cluster
    c: [f32; 5],                // C and lambda are our hyper-parameters for connect_chance in LSM
    lambda: f32,
    rm_dis_n: bool,
    rm_n_count: usize, // The number of disconnected neurons
) {
    // Calculates the average number of connections per neuron and outputs some  \\
    // information about hyper parameters to a txt file. \\
    let mut data: Vec<String> = Vec::new(); // Our return value
    let mut sum_connects: u32 = 0; // The number of connections in total
    let mut sum_dist: f32 = 0.; // The total length of all edges in a cluster
    for connect in connections.iter() {
        sum_connects += connect;
    }
    for d in dists.iter() {
        sum_dist += d;
    }

    // For connects, we subtract by n because there are n "dead" connections since the entries on the main
    // diagonal represents the connections of a neuron to itself.
    let avg_num: f32 = (sum_connects - n as u32) as f32 / n as f32; // if n = 0, then it returns NaN
    let avg_dist: f32 = sum_dist / dists.len() as f32; // another div by 0 if dists is empty

    // Pretty printing
    data.push(format!("Lambda: {:.2}\n", lambda));
    data.push(format!(
        "C     : [EE: {}, EI: {}, IE: {}, II: {}, Loop: {}]\n",
        c[0], c[1], c[2], c[3], c[4]
    ));
    data.push(format!("\nNumber of Neurons: {}", n));
    data.push(format!(
        "\nNumber of connections: {}\n",
        sum_connects - n as u32
    ));
    data.push(format!("\nAverage number of connections per neuron: {:.2}\nAverage distance between connection     : {:.2}\n",
        avg_num, avg_dist
    ));
    let m = min_max(dists);
    data.push(format!(
        "\nFarthest connection: {:.2}\nClosest connection : {:.2}\n",
        m.1, m.0
    ));

    if rm_dis_n {
        data.push(format!("\nNumber of disconnected Neurons: {}", rm_n_count));
        data.push(format!("\nNumber of remaining Neurons: {}", n - rm_n_count));
    }

    let mut f = File::create("analysis.txt").expect("Unable to create file");
    for datum in data.iter() {
        f.write_all(datum.as_bytes()).expect("Unable to write data");
    }
}

// Helper functions \\

fn min_max(dists: &Vec<f32>) -> (f32, f32) {
    // Helper function to get the minimum and maximum in a vector of floats. \\
    if dists.len() == 0 {
        return (f32::NAN, f32::NAN);
    }

    let mut min = dists[0];
    let mut max = dists[0];
    for d in dists.iter() {
        if d < &min {
            min = d.clone();
        }
        if d > &max {
            max = d.clone();
        }
    }

    (min, max)
}

pub fn epoch_accuracy(scores: Vec<String>, last_n: usize) -> f32 {
    let scores_len = scores.len();
    if scores_len < last_n {
        return 0.;
    } else {
        let mut n_correct: u32 = 0;
        for n in 0..last_n {
            if scores[scores_len - n - 1] == "correct".to_string() {
                n_correct += 1;
            }
        }
        return n_correct as f32 / last_n as f32;
    }
}
