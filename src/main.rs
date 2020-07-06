/*
/// Project: Simple Brain-like LSM Build
///
/// Authors: Awildo Gutierrez, Sampreeth Aravilli, Sosina Abuhay, Siqi Fang, Dave Perkins
///
/// Date: July 6, 2020
///
/// Description: We implement a neuron class with a visual representation.
///
/// To do: 
///        o Make readout clusters inside of bigger clusters. 
///        o Take Input
///        o Convert input to voltage
///        o Getting liquid states
///        o Voltage diff eq's and change in voltage function
*/

use std::fs::File;
use std::io::Write;

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

const AXIS_ON: bool = true; // toggles x, y, and z axis
const LINES_ON: bool = true; // toggles edges between neurons
const FANCY: bool = false; // toggles directional dotted lines (white to black)
const YAW: bool = false; // toggles a rotation along the y axis
const SP_ON: bool = true; // toggles each neuron
const RM_DIS_N: bool = true; // determines whether we want to remove neurons with no connections

fn render_lines(
    window: &mut Window,                                  // Our window
    axis_len: f32,                                        // The length of the axis lines
    lines: &Vec<(Point3<f32>, Point3<f32>, Point3<f32>)>, // The edges between neurons
    dists: &Vec<f32>,
    l: &mut LSM,    // List of neurons
) {
    // Renders the edges between neurons as well as the lines of axis. \\
    let mut axis_on: bool = AXIS_ON;
    let mut lines_on: bool = LINES_ON;
    let mut fancy: bool = FANCY;
    let mut yaw: bool = YAW;
    let mut sp_on: bool = SP_ON;
    let rm_dis_n = RM_DIS_N;

    // We want to start off at a point other than the origin so we don't have to
    // zoom out immediately.
    // let eye = Point3::new(5.0, 5.0, 5.0);
    let eye = Point3::new(10.0, 10.0, 10.0);
    let at = Point3::origin();
    let mut arc_ball = ArcBall::new(eye, at);

    // Removes neurons that are not connected to any other neurons
    if rm_dis_n {
        l.remove_disconnects(window);
    }

    // let mut connect_sp: Vec<SceneNode> = Vec::new();
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
                    } else if key == Key::F {
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

        if !fancy {
            fancy_show(&mut connect_sp);
            fancy = !fancy;
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

fn fancy_lines(
    window: &mut Window, // We draw the spheres directly using the window
    lines: &Vec<(
        Point3<f32>, // Start point of a line
        Point3<f32>, // Endpoint of a line
        Point3<f32>,
    )>, // R, G, B color values
    dists: &Vec<f32>,    // Distance of one sphere to each other in order of LSM.neurons
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
    c: [f32; 5],                // C and lambda are our hyper-parameters.
    lambda: f32,
    rm_n_count: usize,
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
    let mut l1 = LSM::new(108, 216, 0.8);
    // PINK: Input: 27x4 = 108  => 27 for 3x3 pic. spiketrain
    // YELLOW: Output: 216/4 = 54/2 = 27 => 3x3 talk spike train

    // Each cluster: Nothing; Talk; Run; Eat

    // Creating Test Clusters \\

    // Colors are tetratic numbers from
    // https://www.colorhexa.com/78866b

    // Cluster 1 \\
    let mut n = 200; // The number of neurons in a single cluster
    let mut var: f32 = 1.25; //1.75 // The variance in std. dev.
    let r = 0.1; // The radius of a single sphere

    let c1 = Point3::new(2.5, 2.5, 3.0);
    let color1 = (134./255., 121./255., 107./255.); 
    // A cluster takes a window, a size, radius, center, variance, and color
    l1.make_cluster(&mut window, n, r, &c1, var, color1);

    n = 150;
    var = 1.15;
    let c2 = Point3::new(-2.5, 2.5, -1.0);
    // let color2 = (0.9453, 0.0938, 0.6641);
    let color2 = (120./255., 134./255., 107./255.);
    l1.make_cluster(&mut window, n, r, &c2, var, color2);

    let c3 = Point3::new(2.0, 1.5, -2.5);
    // let color3 = (0.9453, 0.8203, 0.0938);
    let color3 = (107./255., 120./255., 134./255.);
    l1.make_cluster(&mut window, n, r, &c3, var, color3);

    n = 100;
    var = 1.075;
    let c4 = Point3::new(-1.2, -0.5, 2.5);
    // let color4 = (0.2266, 0.875, 0.4023);
    let color4 = (121./255., 107./255., 134./255.);
    l1.make_cluster(&mut window, n, r, &c4, var, color4);

    // Line Drawing \\
    // Some of these may have been changed from spheres to neurons in name.

    let n_len = l1.get_neurons().len();

    // The paper from Yong Zhang et al initialized the grid with neurons equidistant from each other

    // We found c = 0.2755 and lambda = 2. generate good results after playing around with it
    // c: .25  lambda: 1.8
    // let c = 0.75; //0.25//1.
    let lambda = 2.; //5.//10.
    let c: [f32; 5] = [0.45, 0.3, 0.6, 0.15, 0.1]; // [EE, EI, IE, II, Loop]
    // let c: [f32; 5] = [0.25, 0.25, 0.25, 0.25, 0.1];
    let connects_data = l1.make_connects(c, lambda);
    let lines = connects_data.0;
    let dists = connects_data.1;

    // Rendering \\
    let axis_len = 10.0;
    render_lines(&mut window, axis_len, &lines, &dists, &mut l1);

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
    );
}

// end of file
