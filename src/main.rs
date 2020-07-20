/*
/// Project: Simple Brain-like LSM Build
/// Branch: Train
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

use std::io::Write;
// use std::time::Instant;
use std::{fs, fs::File};

use kiss3d::camera::ArcBall;
use kiss3d::event::{Action, Key, WindowEvent};
use kiss3d::light::Light;
use kiss3d::scene::SceneNode;
use kiss3d::window::Window;

use nalgebra::base::DMatrix;
use nalgebra::geometry::Translation3;
use nalgebra::Point3;

mod machine;
use machine::LSM;

// These global variables are for us to start the program in whatever graphics
// mode we want to.
const AXIS_ON: bool = true; // toggles x, y, and z axis
const LINES_ON: bool = true; // toggles edges between neurons
const READOUT_LINES_ON: bool = true; // toggles lines between the liquid and the readout
const FANCY: bool = false; // toggles directional dotted lines (white to black)
const YAW: bool = false; // toggles a rotation along the y axis
const SP_ON: bool = true; // toggles each liquid neuron
const READOUT_ON: bool = true; // toggles the readout neurons
const RM_DIS_N: bool = true; // determines whether we want to remove neurons with no connections

fn _read_input() -> Vec<Vec<u32>> {
    // Read Perkins' input file (a list of spike trains)
    // This input is of the form where # rows > # columns
    // 0 0 0 1 1 0 0 1 0
    // 0 1 0 1 1 0 1 0 1
    // ...
    // This function stores that in a list by column (assuming the file is transposed).
    const RADIX: u32 = 10; // base 10
    let contents = fs::read_to_string("input.txt").expect("Unable to read input file");
    let contents: Vec<&str> = contents.split("\n").collect();
    let mut input: Vec<Vec<u32>> = Vec::new();
    for line in contents.iter() {
        let mut mini_input: Vec<u32> = Vec::new(); // One line
        let new_line = line.trim(); // Trims whitespace
        for character in new_line.chars() {
            if character != ' ' {
                // to_digit takes a number base
                mini_input.push(character.to_digit(RADIX).unwrap());
            }
        }
        input.push(mini_input);
    }
    input
}

fn _read_labels() -> Vec<String> {
    // A label is a description of the action desired for each column of input.
    // Our file, labels.txt, is what we will use to supervise learning.
    let contents = fs::read_to_string("labels.txt").expect("Unable to read input file");
    let contents: Vec<&str> = contents.split("\n").collect();
    let mut labels: Vec<String> = Vec::new();
    for line in contents.iter() {
        labels.push(line.trim().to_string());
    }
    labels
}

fn read_training_input() -> Vec<Vec<Vec<u32>>> {
    // Read Perkins' input file (a list of spike trains)
    // This input is of the form where # rows > # columns
    // 0 0 0 1 1 0 0 1 0
    // 0 1 0 1 1 0 1 0 1
    // ...
    // This function stores that in a list by column (assuming the file is transposed).
    let mut input: Vec<Vec<Vec<u32>>> = Vec::new();
    let mut train_batch: Vec<Vec<u32>> = Vec::new();

    const RADIX: u32 = 10; // base 10
    let contents = fs::read_to_string("train_input.txt").expect("Unable to read input file");
    let contents: Vec<&str> = contents.split("\n").collect();

    for line in contents.iter() {
        let new_line = line.trim(); // Trims whitespace
        if new_line == "#" {
            input.push(train_batch);
            train_batch = Vec::new();
            continue;
        }
        let mut mini_input: Vec<u32> = Vec::new(); // One line
        for character in new_line.chars() {
            if character != ' ' {
                // to_digit takes a number base
                mini_input.push(character.to_digit(RADIX).unwrap());
            }
        }
        train_batch.push(mini_input);
    }
    input
}

fn read_training_labels() -> Vec<Vec<String>> {
    // A label is a description of the action desired for each column of input.
    // Our file, labels.txt, is what we will use to supervise learning.
    let contents = fs::read_to_string("train_labels.txt").expect("Unable to read input file");
    let contents: Vec<&str> = contents.split("\n").collect();
    let mut labels: Vec<Vec<String>> = Vec::new();
    let mut train_batch: Vec<String> = Vec::new();
    for line in contents.iter() {
        let new_line = line.trim(); // Trims whitespace
        if new_line == "#" {
            labels.push(train_batch);
            train_batch = Vec::new();
            continue;
        }
        train_batch.push(new_line.to_string());
    }
    labels
}

fn render_lines(
    window: &mut Window,                                          // Our window
    axis_len: f32,                                                // The length of the axis lines
    lines: &Vec<(Point3<f32>, Point3<f32>, Point3<f32>)>,         // The edges between neurons
    dists: &Vec<f32>, // Distances between each neuron to each other neuron
    readout_lines: &Vec<(Point3<f32>, Point3<f32>, Point3<f32>)>, // Very similar to lines vector above
    l: &mut LSM,                                                  // List of neurons
) {
    // Renders the edges between neurons as well as the lines of axis. \\

    // Cloning global variables so we can make them mutable.
    let mut axis_on: bool = AXIS_ON;
    let mut lines_on: bool = LINES_ON;
    let mut readout_lines_on: bool = READOUT_LINES_ON;
    let mut fancy: bool = FANCY;
    let mut yaw: bool = YAW;
    let mut sp_on: bool = SP_ON;
    let mut readout_on: bool = READOUT_ON;
    let rm_dis_n = RM_DIS_N;

    // We want to start off at a point other than the origin so we don't have to
    // zoom out immediately.
    let eye = Point3::new(10.0, 10.0, 10.0);
    let at = Point3::origin();
    let mut arc_ball = ArcBall::new(eye, at);

    // Removes neurons that are not connected to any other neurons
    if rm_dis_n {
        l.remove_disconnects(window);
    }

    // Connect spheres will be filled of a line of spheres in a gradient
    // of white to black
    let mut connect_sp: Vec<SceneNode> = Vec::new();
    if fancy {
        connect_sp = fancy_lines(window, lines, dists);
    }

    // Arc ball allows for some useful user controls.
    while window.render_with_camera(&mut arc_ball) {
        // update the current camera.
        for event in window.events().iter() {
            // When a key is pressed, put it into key
            match event.value {
                WindowEvent::Key(key, Action::Release, _) => {
                    // Toggle the axis with A
                    if key == Key::A {
                        axis_on = !axis_on;
                    // Toggle the edges between liquid neurons with L
                    } else if key == Key::L {
                        lines_on = !lines_on;
                    // Toggle the readout lines with C
                    } else if key == Key::C {
                        readout_lines_on = !readout_lines_on;
                    }
                    // Toggle the yaw with Y
                    else if key == Key::Y {
                        yaw = !yaw;
                    // Toggle the liquid neurons with S
                    } else if key == Key::S {
                        sp_on = !sp_on;
                    // Toggle the readout set with R
                    } else if key == Key::R {
                        readout_on = !readout_on;
                    }
                    // Toggle fancy lines with F (causes lag with big setups)
                    else if key == Key::F {
                        fancy = !fancy;
                    }
                }
                _ => {}
            }
        }

        if !sp_on {
            sp_show(l);
            sp_on = !sp_on;
        }

        if !readout_on {
            readout_show(l);
            readout_on = !readout_on;
        }

        if !fancy {
            fancy_show(&mut connect_sp);
            fancy = !fancy;
        }

        if yaw {
            // A yaw rotates the whole window slowly
            let curr_yaw = arc_ball.yaw();
            arc_ball.set_yaw(curr_yaw + 0.025);
        }

        // Our axis lines
        // x axis - Red
        // y axis - Green
        // z axis - Blue
        if axis_on {
            window.draw_line(
                &Point3::new(-axis_len, 0.0, 0.0),
                &Point3::new(axis_len, 0.0, 0.0),
                &Point3::new(1.0, 0.0, 0.0),
            );
            window.draw_line(
                &Point3::new(0.0, -axis_len, 0.0),
                &Point3::new(0.0, axis_len, 0.0),
                &Point3::new(0.0, 1.0, 0.0),
            );
            window.draw_line(
                &Point3::new(0.0, 0.0, -axis_len),
                &Point3::new(0.0, 0.0, axis_len),
                &Point3::new(0.0, 0.0, 1.0),
            );
        }

        // Draw edges between neurons
        if lines_on {
            for coords in lines.iter() {
                window.draw_line(&coords.0, &coords.1, &coords.2);
            }
        }

        // Lines between liquid neurons and the readout set of neurons
        if readout_lines_on {
            for coords in readout_lines.iter() {
                window.draw_line(&coords.0, &coords.1, &coords.2);
            }
        }
    }
}

fn fancy_lines(
    window: &mut Window, // We draw the spheres directly using the window
    lines: &Vec<(
        Point3<f32>, // Start point of a line
        Point3<f32>, // Endpoint of a line
        Point3<f32>, // R, G, B color values
    )>,
    dists: &Vec<f32>, // Distance of one sphere to each other in order of LSM.neurons
) -> Vec<SceneNode> {
    // Alternative way to draw all the lines in the graphics. We represent lines by small spheres. \\
    let sp_per_dist = 0.03_f32; // 1 sphere every 0.05 units
    let mut spheres: Vec<SceneNode> = Vec::new(); // Our output
    for idx in 0..lines.len() {
        let n_spheres: usize = (dists[idx] / sp_per_dist) as usize; // number of spheres in 1 line
                                                                    // Coordinates of each point
        let x1 = (lines[idx].0)[0];
        let y1 = (lines[idx].0)[1];
        let z1 = (lines[idx].0)[2];
        let x2 = (lines[idx].1)[0];
        let y2 = (lines[idx].1)[1];
        let z2 = (lines[idx].1)[2];

        // Drawing 1 line made of 'n_spheres' spheres
        for k in 0..n_spheres {
            let k = (k as f32) / (n_spheres as f32); // shadowing it to do a range of floats
            let mut sp = window.add_sphere(0.01);

            // (x1, y1, z1) + k(x2 - x1, y2 - y1, z2 - z1) is each new sphere
            // center since k increments a bit each time.
            let t = Translation3::new(x1 + k * (x2 - x1), y1 + k * (y2 - y1), z1 + k * (z2 - z1));
            sp.append_translation(&t);
            // Color changes from white to black
            sp.set_color(1.0 - k, 1.0 - k, 1.0 - k);
            // We don't care about moving the actual sphere at this point
            spheres.push(sp);
        }
    }
    spheres
}

fn sp_show(l: &mut LSM) {
    // Toggles the visibility of neurons in an LSM \\
    let neurons = l.get_neurons();
    // We don't want to assert, we want it to print regardless.
    if neurons.len() == 0 {
        return;
    }
    // If one is visible, then they are all visible
    let visible: bool = neurons[0].get_obj().is_visible();
    for n_idx in 0..neurons.len() {
        // get_obj() gets a reference to the sphere in a neuron
        neurons[n_idx].get_obj().set_visible(!visible);
    }
}

fn readout_show(l: &mut LSM) {
    // Toggles the visibility of readout neurons. \\
    let readouts = l.get_readouts();
    let visible: bool = readouts[0][0].get_obj().is_visible();
    // Readouts is a list of clusters (clusters are lists of neurons).
    for cluster_idx in 0..readouts.len() {
        for readout in readouts[cluster_idx].iter_mut() {
            // get_obj returns a mutable reference to a Sphere
            readout.get_obj().set_visible(!visible);
        }
    }
}

fn fancy_show(sp_list: &mut Vec<SceneNode>) {
    // Toggles the visibility of any set of spheres. \\
    if sp_list.len() == 0 {
        return;
    }

    // If one is visible, then they are all visible
    let visible: bool = sp_list[0].is_visible();
    for sp_idx in 0..sp_list.len() {
        sp_list[sp_idx].set_visible(!visible);
    }
}

fn analysis(
    connections: &DMatrix<u32>, // The connections matrix made of 0's and 1's. 1 - connection between the indexed neurons, 0 - no connection
    dists: &Vec<f32>,           // All edge distances
    n: usize,                   // The number of neurons in a cluster
    c: [f32; 5],                // C and lambda are our hyper-parameters for connect_chance in LSM
    lambda: f32,
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

    if RM_DIS_N {
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

fn main() {
    // let now1 = Instant::now(); // Time
    // Important Variables \\
    let mut window = Window::new("Liquid State Machine"); // For graphics display
    window.set_light(Light::StickToCamera); // Graphics settings
    const N_CLUSTERS: usize = 2; // The number of clusters
                                 // Input neurons, number of clusters, ratio of
                                 // excitatory to inhibitory
    let inputs_per_cluster = 48;
    let mut l1 = LSM::new(inputs_per_cluster, N_CLUSTERS, 0.8);
    // PINK: Input: 27x4 = 108  => 27 for 3x3 pic. spiketrain
    // YELLOW: Output: 216/4 = 54/2 = 27 => 3x3 talk spike train

    // Each cluster: Idle; Run; Eat; Talk

    // Creating Test Clusters \\

    // Colors are tetradic numbers from
    // https://www.colorhexa.com/78866b

    let c1 = Point3::new(2.5, 2.5, 3.0);
    let c2 = Point3::new(-2.5, 2.5, -1.0);
    let c3 = Point3::new(2.0, 1.5, -2.5);
    let c4 = Point3::new(-1.2, -0.5, 2.5);

    let color1 = (172. / 255., 222. / 255., 248. / 255.);
    let color2 = (255. / 255., 148. / 255., 113. / 255.);
    let color3 = (224. / 255., 187. / 255., 228. / 255.);
    let color4 = (235. / 255., 212. / 255., 148. / 255.);

    // Clusters \\
    let n: usize = 300; // The total number of neurons in all clusters
    let var: f32 = 1.15; //1.75 // The variance in std. dev.
    let r: f32 = 0.1; // The radius of a single sphere

    // IF UPDATED MUST UPDATE NEURON GLOBAL 'CLUSTERS'
    let cluster_types: Vec<&str> = vec!["talk", "hide", "run", "eat"];

    let cluster_locs: Vec<Point3<f32>> = vec![c1, c2, c3, c4];
    let cluster_colors: Vec<(f32, f32, f32)> = vec![color1, color2, color3, color4];
    // Try to keep it odd
    let n_readouts: Vec<usize> = vec![5, 5, 5, 5]; // number of readout neurons per cluster

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
    }

    let n_len = l1.get_neurons().len();

    // hyper parameters
    let lambda = 2.; //5.//10.
    let c: [f32; 5] = [0.45, 0.3, 0.6, 0.15, 0.1]; // [EE, EI, IE, II, Loop]

    let connects_data = l1.make_connects(&mut window, c, lambda);
    let lines = connects_data.0;
    let dists = connects_data.1;
    let readout_lines = connects_data.2;

    let train = true;
    // let epochs = 250;
    let epochs = 500;
    // let input = read_input();
    // let labels = read_labels();
    let input = read_training_input();
    let labels = read_training_labels();
    assert_eq!(labels.len(), input.len()); // epochs
    assert_eq!(labels[0].len(), input[0][0].len()); // # of labels == # of columns
    let models = ["static", "first order", "second order"];
    let delay = 1;
    let first_tau = 0;
    let mut f1 = File::create("output.txt").expect("Unable to create a file");
    let mut f2 = File::create("readout.txt").expect("Unable to create a file");
    let mut f3 = File::create("guesses.txt").expect("Unable to create a file");
    let mut f4 = File::create("epochs.txt").expect("Unable to create a file");

    // println!("\nFIRST ONE AFTER RESET IS IMPLEMENTED\n");
    // println!("\nREADOUTS TAKE TIME OUTS AFTER SPIKE; DELTA CALCIUM IS 1 / 10\n");
    println!("\nPROB: 0.9 (TAXI-CAB)\n");
    for epoch in 0..epochs {
        println!("Running epoch {} ...", epoch);
        l1.run(
            train,
            epoch,
            &mut f1,
            &mut f2,
            &mut f3,
            &mut f4,
            &input[epoch],
            &labels[epoch],
            models[0],
            delay,
            first_tau,
        );
        l1.reset();
        println!("Epoch {} finished", epoch);
        f1.write_all(format!("\nEpoch {} finished\n\n\n", epoch).as_bytes())
            .expect("Unable to write");
        f2.write_all(format!("\nEpoch {} finished\n\n\n", epoch).as_bytes())
            .expect("Unable to write");
        f3.write_all(format!("\nEpoch {} finished\n\n\n", epoch).as_bytes())
            .expect("Unable to write");
    }

    /*
    for test in 995..1000 {
        println!("Running test {} ...", test);
        l1.run(
            train,
            test,
            &mut f1,
            &mut f2,
            &mut f3,
            &mut f4,
            &input[test],
            &labels[test],
            models[0],
            delay,
            first_tau,
        );
        l1.reset();
        println!("Test {} finished", test);
        f1.write_all(format!("\nTest {} finished\n\n\n", test).as_bytes())
            .expect("Unable to write");
        f2.write_all(format!("\nTest {} finished\n\n\n", test).as_bytes())
            .expect("Unable to write");
        f3.write_all(format!("\nTest {} finished\n\n\n", test).as_bytes())
            .expect("Unable to write");
    }
    */

    // let run_time1 = now1.elapsed().as_millis() as f64 / 1000.;
    // println!("Time elapsed: {:.2} secs", run_time1);
    // println!("Time elapsed: {:.2} mins", run_time1 / 60.);

    // l1.run(!train, &input, &labels, models[0], delay, first_tau);

    // let run_time2 = now2.elapsed().as_millis() as f64 / 1000.;
    // println!("Time elapsed: {:.2} secs", run_time2);
    // println!("Time elapsed: {:.2} mins", run_time2 / 60.);

    // Rendering \\
    let axis_len = 10.0;
    render_lines(
        &mut window,
        axis_len,
        &lines,
        &dists,
        &readout_lines,
        &mut l1,
    );

    // Refers back to how many neurons it used to have before they were removed
    // to figure out how many were removed
    let rm_n_count = n_len - l1.get_neurons().len();

    // Analysis \\
    analysis(l1.get_connections(), &dists, n_len, c, lambda, rm_n_count);
}

// end of file
