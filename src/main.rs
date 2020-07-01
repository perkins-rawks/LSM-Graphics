/*
/// Project: Simple Brain-like LSM Build
///
/// Authors: Awildo Gutierrez, Sampreeth Aravilli, Sosina Abuhay, Siqi Fang, Dave Perkins
///
/// Date: June 30, 2020
///
/// Description: We implement a neuron class with a visual representation
///
/// To do: o Finish Neuron class
///        o Input and output layers
///        o LSM Struct
///        o
///        o
*/
// extern crate lsm_graphics;
use std::fs::File;
use std::io::Write;

use kiss3d::camera::ArcBall;
use kiss3d::event::{Action, Key, WindowEvent};
use kiss3d::light::Light;
use kiss3d::window::Window;

use nalgebra::base::DMatrix;
use nalgebra::Point3;

mod machine;
use machine::LSM;

fn render_lines(
    window: &mut Window,                                 // Our window
    axis_len: f32,                                       // The length of the axis lines
    lines: Vec<(Point3<f32>, Point3<f32>, Point3<f32>)>, // The edges between neurons
    l: &mut LSM,                                         // List of neurons
    rm_dis_n: bool, // False - All Neurons, True - Remove Neuron with no connections
) {
    // Renders the edges between neurons as well as the lines of axis. \\
    let mut axis_on: bool = true;
    let mut lines_on: bool = true;
    let mut yaw: bool = false;
    let mut sp_on: bool = true;

    // We want to start off at a point other than the origin so we don't have to
    // zoom out immediately.
    let eye = Point3::new(5.0, 5.0, 5.0);
    let at = Point3::origin();
    let mut arc_ball = ArcBall::new(eye, at);

    // Removes neurons that are not connected to any other neurons
    if rm_dis_n {
        l.remove_disconnects(window);
    }
    // window.se();

    // Arc ball allows for some useful user controls.
    while window.render_with_camera(&mut arc_ball) {
        // update the current camera.
        for event in window.events().iter() {
            // When a key is pressed, put it into key
            match event.value {
                WindowEvent::Key(key, Action::Release, _) => {
                    // Turn on or off the axis with A
                    if key == Key::A {
                        axis_on = !axis_on;
                    // Turn on or off the edges with L
                    } else if key == Key::L {
                        lines_on = !lines_on;
                    // Turn on or off the yaw with Y
                    } else if key == Key::Y {
                        yaw = !yaw;
                    } else if key == Key::S {
                        sp_on = !sp_on;
                        // sp_show(l);
                    }
                }
                _ => {}
            }
        }

        if !sp_on {
            sp_show(l);
            sp_on = !sp_on;
        }

        if yaw {
            let curr_yaw = arc_ball.yaw();
            arc_ball.set_yaw(curr_yaw + 0.025);
        }
        // A yaw rotates the whole window slowly, but allows us
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
    }
}

fn sp_show(l: &mut LSM) {
    let neurons = l.get_neurons();
    if neurons.len() > 0 {
        // assert neurons is non empty
        let invisible: bool = neurons[0].get_obj().is_visible();
        for n_idx in 0..neurons.len() {
            neurons[n_idx].get_obj().set_visible(!invisible);
        }
    }
}

fn analysis(
    connections: &DMatrix<u32>, // The connections matrix made of 0's and 1's. 1 - connection between the indexed neurons, 0 - no connection
    dists: &Vec<f32>,           // All edge distances
    n: usize,                   // The number of neurons in a cluster
    c: [f32; 4],                // C and lambda are our hyper-parameters.
    lambda: f32,
    rm_n_count: usize,
    rm_n: bool,
) {
    // Calculates the average number of connections per neuron and outputs some  \\
    // information about hyper parameters to a txt file. \\
    let mut data: Vec<String> = Vec::new();
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
    data.push(format!("Lambda: {:.2}\n", lambda));
    data.push(format!(
        "C     : [EE: {}, EI: {}, IE: {}, II: {}]\n",
        c[0], c[1], c[2], c[3]
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

    if rm_n {
        data.push(format!("\nNumber of disconnected Neurons: {}", rm_n_count));
        data.push(format!("\nNumber of remaining Neurons: {}", n - rm_n_count));
    }

    let mut f = File::create("analysis.txt").expect("Unable to create file");
    for datum in data.iter() {
        f.write_all(datum.as_bytes()).expect("Unable to write data");
    }
}

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
    // Important Variables \\
    let mut window = Window::new("Liquid State Machine"); // For graphics display
    window.set_light(Light::StickToCamera); // Graphics settings
    let mut l1 = LSM::new(4, 5, 0.8);

    // Creating Test Clusters \\
    // Cluster 1 \\
    let n = 100; // The number of neurons in a single cluster
    let var: f32 = 0.35; //1.75 // The variance in std. dev.
    let r = 0.1; // The radius of a single sphere
    let c1 = Point3::new(0.0, 0.0, 0.0);
    let color1 = (0.2578, 0.5273, 0.957); //blueish
                                          // A cluster takes a window, a size, radius, center, variance, and color
    l1.make_cluster(&mut window, n, r, &c1, var, color1);

    // Line Drawing \\
    // Some of these may have been changed from spheres to neurons in name.

    let n_len = l1.get_neurons().len();

    // The paper from Yong Zhang et al initialized the grid with neurons equidistant from each other

    // We found c = 0.2755 and lambda = 2. generate good results after playing around with it
    // c: .25  lambda: 1.8
    // let c = 0.75; //0.25//1.
    let lambda = 2.; //5.//10.
    let c: [f32; 4] = [0.45, 0.3, 0.6, 0.15]; // [EE, EI, IE, II]
    let connects_data = l1.make_connects(c, lambda);
    let lines = connects_data.0;
    let dists = connects_data.1;

    // Rendering \\
    let axis_len = 3.0;
    let rm_dis_n = false;
    render_lines(&mut window, axis_len, lines, &mut l1, rm_dis_n);

    // Refers back to how many neurons it used to have before they were removed
    // to figure out how many were removed
    let rm_n_count = n_len - l1.get_neurons().len();

    // Analysis \\
    analysis(
        l1.get_connections(),
        &dists,
        n_len,
        c,
        lambda,
        rm_n_count,
        rm_dis_n,
    );
}

// end of file
