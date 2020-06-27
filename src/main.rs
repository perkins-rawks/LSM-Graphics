/*
/// Project: Simple Brain-like LSM Build
///
/// Authors: Awildo Gutierrez, Sampreeth Aravilli, Sosina Abuhay, Siqi Fang, Dave Perkins
///
/// Date: June 27, 2020
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
    let mut axis_on: bool = false;
    let mut lines_on: bool = true;
    let mut yaw: bool = false;

    // We want to start off at a point other than the origin so we don't have to
    // zoom out immediately.
    let eye = Point3::new(10.0f32, 10.0, 10.0);
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
                    }
                }
                _ => {}
            }
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

fn analysis(
    connections: &DMatrix<u32>, // The connections matrix made of 0's and 1's. 1 - connection between the indexed neurons, 0 - no connection
    dists: &Vec<f32>,           // All edge distances
    n: usize,                   // The number of neurons in a cluster
    c: f32,                     // C and lambda are our hyper-parameters.
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
    let avg_num: f32 = (sum_connects - n as u32) as f32 / n as f32;
    let avg_dist: f32 = sum_dist / dists.len() as f32;
    data.push(format!("C     : {}\nLambda: {:.2}\n", c, lambda));
    data.push(format!("\nTotal connections: {}", sum_connects - n as u32));
    data.push(format!("\nAverage number of connections per neuron: {:.2}\nAverage distance between connection     : {:.2}\n",
        avg_num, avg_dist
    ));
    let m = min_max(dists);
    data.push(format!(
        "\nFarthest connection: {:.2}\nClosest connection : {:.2}\n",
        m.1, m.0
    ));

    if rm_n {
        data.push(format!(
            "\nNumber of disconnected Neurons: {}\n",
            rm_n_count
        ));
    }

    let mut f = File::create("analysis.txt").expect("Unable to create file");
    for datum in data.iter() {
        f.write_all(datum.as_bytes()).expect("Unable to write data");
    }

    // for datum in data.iter() {
    //     println!("{}", datum);
    // }
}

fn min_max(dists: &Vec<f32>) -> (f32, f32) {
    // Helper function to get the minimum and maximum in a vector of floats. \\
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
                                            // let mut spheres: Vec<SceneNode> = Vec::new();
                                            // let mut neurons: Vec<Neuron> = Vec::new();
                                            // let mut clusters: Vec<Vec<Neuron>> = Vec::new();
    let mut l1 = LSM::new(Vec::new(), Vec::new());

    // Creating Test Clusters \\
    let mut n = 200; // The number of neurons in a single cluster
    let mut var: f32 = 1.25; //1.75 // The variance in stdev
    let r = 0.1; // The radius of a single sphere

    // Generate a cluster by giving it a cluster center (-1., 2., -3.)
    // e.g
    // let c1000 = Point3::new(-1., 2., -3.);

    // Cluster 1 \\
    let c1 = Point3::new(2.5, 2.5, 3.0);
    let color1 = (0.2578, 0.5273, 0.957);
    // let clust1: Vec<SceneNode> = make_cluster(&mut window, n, r, &c1, var, (0.2578, 0.5273, 0.957));
    // let clust1: Vec<Neuron> = make_cluster(&mut window, n, r, &c1, var, color1);
    // // clusters.push(clust1);
    // l1.add_cluster(clust1);
    l1.make_cluster(&mut window, n, r, &c1, var, color1);
    // dist(clust1[0].data().local_translation(),
    // clust1[1].data().local_translation());

    // Cluster 2 \\
    n = 150;
    var = 1.15; // 1.55
                // let c400 = Point3::new(5., -1.3, -3.4);
    let c2 = Point3::new(-2.5, 2.5, -1.0);
    let color2 = (0.9453, 0.0938, 0.6641);
    // let clust2: Vec<Neuron> = make_cluster(&mut window, n, r, &c2, var, color2);
    // // clusters.push(clust2);
    // l1.add_cluster(clust2);
    l1.make_cluster(&mut window, n, r, &c2, var, color2);
    // dist(clust1[0].data().local_translation(), clust2[1].data().local_translation());

    // Cluster 3 \\
    // let c600 = Point3::new(3., 6., 1.);
    let c3 = Point3::new(2.0, 1.5, -2.5);
    let color3 = (0.9453, 0.8203, 0.0938);
    // let clust3: Vec<Neuron> = make_cluster(&mut window, n, r, &c3, var, color3);
    // // clusters.push(clust3);
    // l1.add_cluster(clust3);
    l1.make_cluster(&mut window, n, r, &c3, var, color3);

    // Cluster 4 \\
    n = 100;
    var = 1.075;
    let c4 = Point3::new(-1.2, -0.5, 2.5);
    let color4 = (0.2266, 0.875, 0.4023);
    // let clust4: Vec<Neuron> = make_cluster(&mut window, n, r, &c4, var, color4);
    // // clusters.push(clust4);
    // l1.add_cluster(clust4);
    l1.make_cluster(&mut window, n, r, &c4, var, color4);

    // Line Drawing \\
    // Some of these may have been changed from spheres to neurons in name.

    // unpack(&clusters, &mut spheres);
    // unpack(&clusters, &mut neurons);
    // l1.unpack();

    // let sp_len = spheres.len();\
    let n_len = l1.get_neurons().len();
    // let mut connections = DMatrix::<u32>::zeros(sp_len, sp_len);
    // let mut connections = DMatrix::<u32>::zeros(n_len, n_len);
    // l1.add_connections(DMatrix::<u32>::zeros(n_len, n_len));
    // connections.fill_diagonal(1); // Because all the neurons are connected to
    // themselves
    //l1.get_connections().fill_diagonal(1);

    // We found c = 0.2755 and lambda = 2. generate good results after playing around with it
    // c: .25  lambda: 1.8
    let c = 0.25;
    let lambda = 1.8;
    // let connects_data = make_connects(/*&mut window,*/ &spheres, &mut connections, c, lambda);
    // let connects_data = make_connects(
    //     /*&mut window,*/ l1.get_neurons(),
    //     l1.get_connections(),
    //     c,
    //     lambda,
    // );

    let connects_data = l1.make_connects(c, lambda);
    let lines = connects_data.0;
    let dists = connects_data.1;

    // let n_count = neurons.len();

    // Rendering \\
    let axis_len = 7.5;
    let rm_dis_n = true;
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
