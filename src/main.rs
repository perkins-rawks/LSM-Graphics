/*
/// Project: Simple Brain-like LSM Build
/// Branch: EPOCH-training
///
/// Authors: Awildo Gutierrez, Sampreeth Aravilli, Sosina Abuhay, Siqi Fang, Dave Perkins
///
/// Date: July 13, 2020
///
/// Description: We implement a Liquid State Machine with a visual representation and clusters, imitating the brain.
///              We take input from an environment with either ambient noise or some meaningful information, all
///              represented by special or non-special patterns of 0's and 1's. These abstract uses are towards the
///              goal of getting two or more "LSM"s to talk to each other to complete a task.
///
/// To do:
///        o Training Readouts to EAT, RUN, and IDLE
///        o Teaching / Learning signals ^ ^
///        o TALK function
*/

use std::fs::File;
use std::io::Write;

use kiss3d::light::Light;
use kiss3d::window::Window;

use nalgebra::Point3;

mod analysis;
mod graphics;
mod input;
mod machine;
use machine::LSM;

// These global variables are for us to start the program in whatever graphics
// mode we want to.
const AXIS_ON: bool = false; // toggles x, y, and z axis
const LINES_ON: bool = true; // toggles edges between neurons
const READOUT_LINES_ON: bool = true; // toggles lines between the liquid and the readout
const FANCY: bool = false; // toggles directional dotted lines (white to black)
const YAW: bool = false; // toggles a rotation along the y axis
const SP_ON: bool = true; // toggles each liquid neuron
const READOUT_ON: bool = true; // toggles the readout neurons
const RM_DIS_N: bool = true; // determines whether we want to remove neurons with no connections

fn main() {
    // let now1 = Instant::now(); // Time
    // Important Variables \\
    let mut window = Window::new("Liquid State Machine"); // For graphics display
    window.set_light(Light::StickToCamera); // Graphics settings
    const N_CLUSTERS: usize = 1; // The number of clusters
                                 // Input neurons, number of clusters, ratio of
                                 // excitatory to inhibitory
    let inputs_per_cluster = 81;
    let n_readout_clusters = 3;
    let n_input_copies = 1;
    let mut l1 = LSM::new(
        inputs_per_cluster,
        n_input_copies,
        N_CLUSTERS,
        n_readout_clusters,
        0.8,
    );
    // let mut l2 = LSM::new(inputs_per_cluster, n_input_copies, N_CLUSTERS, 1, 0.8);
    // PINK: Input: 27x4 = 108  => 27 for 3x3 pic. spiketrain
    // YELLOW: Output: 216/4 = 54/2 = 27 => 3x3 talk spike train

    // Each cluster: Idle; Run; Eat; Talk

    // Creating Test Clusters \\

    // Colors are tetradic numbers from
    // https://www.colorhexa.com/78866b

    let c1 = Point3::new(3.5, 3.5, 4.0);
    let c2 = Point3::new(-2.5, 2.5, -1.0);
    let c3 = Point3::new(2.0, 1.5, -2.5);
    let c4 = Point3::new(-1.2, -0.5, 2.5);

    let color1 = (172. / 255., 222. / 255., 248. / 255.);
    let color2 = (255. / 255., 148. / 255., 113. / 255.);
    let color3 = (224. / 255., 187. / 255., 228. / 255.);
    let color4 = (235. / 255., 212. / 255., 148. / 255.);

    // Clusters \\
    let n: usize = 200; // The total number of neurons in all clusters
    let var: f32 = 1.15; //1.75 // The variance in std. dev.
    let r: f32 = 0.1; // The radius of a single sphere

    // IF UPDATED MUST UPDATE NEURON GLOBAL 'CLUSTERS'
    // let cluster_types: Vec<&str> = vec!["talk", "hide", "run", "eat"];
    let cluster_types: Vec<&str> = vec!["hide", "run", "eat"];
    // let bildo = "hide";

    let cluster_locs: Vec<Point3<f32>> = vec![c1, c2, c3, c4];
    let cluster_colors: Vec<(f32, f32, f32)> = vec![color1, color2, color3, color4];
    // Try to keep it odd
    let n_readouts: Vec<usize> = vec![3, 1, 3, 3]; // number of readout neurons per cluster
    assert_eq!(0, n_readouts[0] % n_readout_clusters);

    assert_eq!(0, n % N_CLUSTERS);

    for idx in 0..N_CLUSTERS {
        l1.make_cluster(
            &mut window,
            n / N_CLUSTERS,
            r,
            var,
            &cluster_types[idx],
            &cluster_locs[idx],
            cluster_colors[idx],
            n_readouts[idx],
        );
        /*
        l2.make_cluster(
            &mut window,
            n / N_CLUSTERS,
            r,
            var,
            &bildo,
            &cluster_locs[idx + 1],
            cluster_colors[idx + 1],
            n_readouts[idx + 1],
        );
        */
    }

    let n_len = l1.get_neurons().len();

    // hyper parameters
    let lambda = 2.; //5.//10.
    let c: [f32; 5] = [0.45, 0.3, 0.6, 0.15, 0.1]; // [EE, EI, IE, II, Loop]
    let connects_data = l1.make_connects(&mut window, c, lambda);
    // let _connects_data2 = l2.make_connects(&mut window, c, lambda);
    let lines = connects_data.0;
    let dists = connects_data.1;
    let readout_lines = connects_data.2;
    let train = true;
    let epochs = 16;
    let time_steps = 30;
    let input = input::read_training_input();
    let labels = input::read_training_labels();
    assert_eq!(labels.len(), input.len()); // epochs
                                           // assert_eq!(labels.len(), epochs);
                                           // assert_eq!(labels.len(), input[0].len());
    let models = ["static", "first order", "second order"];
    let delay = 1;
    let first_tau = 4;
    let mut f1 = File::create("output.txt").expect("Unable to create a file");
    let mut f2 = File::create("readout.txt").expect("Unable to create a file");
    let mut f3 = File::create("guesses.txt").expect("Unable to create a file");
    let mut f4 = File::create("epochs.txt").expect("Unable to create a file");
    let mut f5 = File::create("weights.txt").expect("Unable to create a file");
    // let mut f3_2 = File::create("l2-guesses.txt").expect("Unable to create a file");
    println!("\nVARIABLE TIME STEPS\n");
    // let mut run_accuracy: f32 = 0.0;
    let mut scores: Vec<String> = Vec::new(); // has either corrects or incorrects
    let mut prev_epoch_accuracy: f32;
    let last_n = 20;
    for epoch in 0..epochs {
        println!("Running epoch {} ...", epoch);
        l1.reset();
        prev_epoch_accuracy = analysis::epoch_accuracy(scores.clone(), last_n);
        scores.push(l1.run(
            train,
            epoch,
            &mut f1,
            &mut f2,
            &mut f3,
            &mut f4,
            &mut f5,
            &input[epoch],
            &labels[epoch],
            models[1],
            delay,
            first_tau,
            prev_epoch_accuracy,
        ));
        println!("Epoch {} finished", epoch);
        f1.write_all(format!("\nEpoch {} finished\n\n\n", epoch).as_bytes())
            .expect("Unable to write");
        f2.write_all(format!("\nEpoch {} finished\n\n\n", epoch).as_bytes())
            .expect("Unable to write");
        f3.write_all(format!("Epoch {} finished\n\n", epoch).as_bytes())
            .expect("Unable to write");
    }

    // let trained_weights = true;
    let tests = 4;

    if epochs == 0 && tests > 0 {
        l1.load_readout_weights();
    }

    prev_epoch_accuracy = 0.;
    let mut test_scores: Vec<String> = Vec::new();
    for test in epochs..epochs + tests {
        println!("Running test {} ...", test);
        l1.reset();
        test_scores.push(l1.run(
            !train,
            test,
            &mut f1,
            &mut f2,
            &mut f3,
            &mut f4,
            &mut f5,
            &input[test],
            &labels[test],
            models[1],
            delay,
            first_tau,
            prev_epoch_accuracy,
        ));
        println!("TEST {} finished", test);
        f1.write_all(format!("\nTEST {} finished\n\n\n", test).as_bytes())
            .expect("Unable to write");
        f2.write_all(format!("\nTEST {} finished\n\n\n", test).as_bytes())
            .expect("Unable to write");
        f3.write_all(format!("TEST {} finished\n\n", test).as_bytes())
            .expect("Unable to write");
    }

    let mut correct = 0;
    // assert_eq!(scores.len() >= tests, true);
    for score in test_scores.iter() {
        if score == "correct" {
            correct += 1;
        }
    }
    if test_scores.len() > 0 {
        println!(
            "Total correct is {} / {}, {:.3}%",
            correct,
            tests,
            (correct as f32) / (tests as f32) * 100.
        );
    }

    // for train in speech.iter() {
    //     for thing in train.iter() {
    //         print!("{} ", thing);
    //     }
    //     println!();
    // }
    /*
    let epochs2 = 40;
    let tests2 = 200;
    for epoch in 0..epochs2 {
        println!("Running epoch {} ...", epoch);
        let speech = l1.talk(bildo, time_steps);
        l2.reset();
        prev_epoch_accuracy = analysis::epoch_accuracy(scores.clone(), last_n);
        scores.push(l2.run(
            train,
            epoch,
            &mut f1,
            &mut f2,
            &mut f3_2,
            &mut f4,
            &mut f5,
            &speech,
            &bildo.to_string(),
            models[2],
            delay,
            first_tau,
            prev_epoch_accuracy,
        ));
        println!("Epoch {} finished", epoch);
        f1.write_all(format!("\nEpoch {} finished\n\n\n", epoch).as_bytes())
            .expect("Unable to write");
        f2.write_all(format!("\nEpoch {} finished\n\n\n", epoch).as_bytes())
            .expect("Unable to write");
        f3_2.write_all(format!("Epoch {} finished\n\n", epoch).as_bytes())
            .expect("Unable to write");
    }

    let mut test_scores: Vec<String> = Vec::new();
    for test in epochs2..epochs2 + tests2 {
        println!("Running test {} ...", test);
        l2.reset();
        let speech = l1.talk(bildo, time_steps);
        test_scores.push(l2.run(
            !train,
            test,
            &mut f1,
            &mut f2,
            &mut f3_2,
            &mut f4,
            &mut f5,
            &speech,
            &bildo.to_string(),
            models[2],
            delay,
            first_tau,
            prev_epoch_accuracy,
        ));
        println!("TEST {} finished", test);
        f1.write_all(format!("\nTEST {} finished\n\n\n", test).as_bytes())
            .expect("Unable to write");
        f2.write_all(format!("\nTEST {} finished\n\n\n", test).as_bytes())
            .expect("Unable to write");
        f3_2.write_all(format!("TEST {} finished\n\n", test).as_bytes())
            .expect("Unable to write");
    }

    let mut correct = 0;
    // assert_eq!(scores.len() >= tests, true);
    for score in test_scores.iter() {
        if score == "correct" {
            correct += 1;
        }
    }
    if test_scores.len() > 0 {
        println!(
            "Total correct is {} / {}, {:.3}%",
            correct,
            tests2,
            (correct as f32) / (tests2 as f32) * 100.
        );
    }
    */

    // Rendering \\
    let axis_len = 5.0;
    let conditions: [bool; 8] = [
        AXIS_ON,
        LINES_ON,
        READOUT_LINES_ON,
        FANCY,
        YAW,
        SP_ON,
        READOUT_ON,
        RM_DIS_N,
    ];
    graphics::render_lines(
        &mut window,
        axis_len,
        &lines,
        &dists,
        &readout_lines,
        n,
        time_steps,
        conditions,
        &mut l1,
    );

    // Refers back to how many neurons it used to have before they were removed
    // to figure out how many were removed
    let rm_n_count = n_len - l1.get_neurons().len();

    // Analysis \\
    analysis::analysis(
        l1.get_connections(),
        &dists,
        n_len,
        c,
        lambda,
        RM_DIS_N,
        rm_n_count,
    );
}

// end of file
