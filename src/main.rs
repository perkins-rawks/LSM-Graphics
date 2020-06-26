/*
/// Project: Simple Brain-like LSM Build
///
/// Authors: Awildo G., Sam. A., Sosina A., Siqi F., Dave P.
///
/// Date: June 25, 2020
///
/// We have an outer bound for a big sphere. We plot a few neurons randomly inside spaced far apart.
/// (we have to prevent forming clusters) Then, for each neuron we choose a (random) smaller radius
/// and we make more neurons and connect all of them with a high probability.
///  To do: o Neuron classes
///         o Input and output layers
///         o Plotting
///
/// First let's do a small example with just two fixed locations.
///
/// In Spherical coordinates r theta phi
///
/// Sphere r = 10
/// two clusters at with center (2.5, 90 (pi/4), 0) and (2.5, 270 (3pi/2), 0)
/// each cluster has a radius of 1
*/

// extern crate kiss3d;
// extern crate nalgebra as na;
// extern crate rand;

// use na::{Vector3, UnitQuaternion};
// use kiss3d::window::Window;
// use kiss3d::light::Light;

// struct Neuron {
//     // Some stuff
//     shape: ,

// }

// struct LSM {
//     // Big structure of clustered up neurons and connections
// }

// impl LSM {
//     fn init() -> LSM {
//         Self {
//             // maybe number of clusters, radius of brain sphere, etc
//         }
//     }

//     fn plot() {
//         // plots each cluster in a 3d simulation
//     }

//     fn create (self) {
//         // Make the outer brain and the inner clusters based on attributes
//     }
// }

use kiss3d::camera::ArcBall;
use kiss3d::light::Light;
use kiss3d::scene::SceneNode;
use kiss3d::window::Window;
// use kiss3d::resource::{GPUVec, BufferType, AllocationType};

use nalgebra::base::DMatrix;
use nalgebra::geometry::Translation3;
use nalgebra::{Point3, Vector3};

use rand::prelude::*;
use rand_distr::{Distribution, Normal};

// window, number of neurons, location to be centered around, variance (for bell
// curve)
fn cluster(
    window: &mut Window,
    n: usize,
    radius: f32,
    loc: &Point3<f32>,
    var: f32,
    (r, g, b): (f32, f32, f32),
) -> Vec<SceneNode> {
    // StreamDraw means we will constantly change the vector
    // Dynamic
    // let mut spheres = GPUVec::new(Vec::new(), BufferType::Array, AllocationType::StreamDraw);
    let mut spheres: Vec<SceneNode> = Vec::new();
    let mut rng = rand::thread_rng(); // a single random number
                                      // let color = (rng.gen(), rng.gen(), rng.gen());

    for sphere in 0..n {
        // Normal takes mean and then variance
        let normal_x = Normal::new(loc[0], var).unwrap();
        let normal_y = Normal::new(loc[1], var).unwrap();
        let normal_z = Normal::new(loc[2], var).unwrap();
        let t = Translation3::new(
            normal_x.sample(&mut rng),
            normal_y.sample(&mut rng),
            normal_z.sample(&mut rng),
        );
        // if let Some(spheres) = spheres.data_mut() {
        //     spheres.push(window.add_sphere(0.1));
        //     spheres[sphere].set_color(color.0, color.1, color.2);
        //     spheres[sphere].append_translation(&t);
        // }

        spheres.push(window.add_sphere(radius));
        // spheres[sphere].set_color(color.0, color.1, color.2);
        spheres[sphere].set_color(r, g, b);
        spheres[sphere].append_translation(&t);
    }
    // println!("{:?}", spheres[0].data().local_translation());
    spheres
}

fn render_lines(
    window: &mut Window,
    axis_on: bool,
    axis_len: f32,
    lines: Vec<(Point3<f32>, Point3<f32>)>,
    lines_on: bool,
) {
    let eye = Point3::new(10.0f32, 10.0, 10.0);
    let at = Point3::origin();
    let mut first_person = ArcBall::new(eye, at);

    while window.render_with_camera(&mut first_person) {
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
        if lines_on {
            for coords in lines.iter() {
                window.draw_line(&coords.0, &coords.1, &Point3::new(1., 1., 1.));
            }
        }
    }
}

// Takes the list of clusters of spheres and individually add the spheres to the
// sp_list so all sphere can be accessed regardless of cluster
fn unpack(clusters: &Vec<Vec<SceneNode>>, sp_list: &mut Vec<SceneNode>) {
    for cluster in clusters.iter() {
        for sphere in cluster.iter() {
            sp_list.push(sphere.clone());
        }
    }
}

// Finds the Euclidean Distance between 2 3D points
fn dist((x1, y1, z1): &(f32, f32, f32), (x2, y2, z2): &(f32, f32, f32)) -> f32 {
    // let d = ((x2 - x1).powf(2.) + (y2 - y1).powf(2.) + (z2 - z1).powf(2.)).sqrt();
    // println!("\nCoordinate 1: ({}, {}, {})", x1, y1, z1);
    // println!("Coordinate 2: ({}, {}, {}) \n", x2, y2, z2);
    // println!("Distance: {}", d);
    // d
    ((x2 - x1).powf(2.) + (y2 - y1).powf(2.) + (z2 - z1).powf(2.)).sqrt()
}

// Uses the equation from the paper to determine the probability of 2 neurons connecting
fn connect_chance(c: f32, d_ab: f32, lambda: f32) -> f32 {
    let exponent: f32 = -1. * ((d_ab / lambda).powf(2.));
    c * exponent.exp()
}

// Adds the actual lines that connect the spheres
// fn draw_connects(
//     window: &mut Window,
//     (x1, y1, z1): (f32, f32, f32),
//     (x2, y2, z2): (f32, f32, f32),
// ) -> (Point3<f32>, Point3<f32>){
//     // let mut rend = LineRenderer::new();
//     // rend.draw_line(
//     //     Point3::new(x1, y1, z1),
//     //     Point3::new(x2, y2, z2),
//     //     Point3::new(1., 1., 1.),
//     // );
//     // window.draw_line(
//     //     &Point3::new(x1, y1, z1),
//     //     &Point3::new(x2, y2, z2),
//     //     &Point3::new(1., 1., 1.),
//     // )
//     (Point3::new(x1, y1, z1), Point3::new(x2, y2, z2))
// }

// Uses the equation determine if any two spheres (neurons will have a connection)
fn make_connects(
    // window: &mut Window,
    sp_list: &Vec<SceneNode>,
    connects: &mut DMatrix<u32>,
    c: f32,
    lambda: f32,
) -> Vec<(Point3<f32>, Point3<f32>)> {
    let mut rng = rand::thread_rng();
    let mut connect_lines: Vec<(Point3<f32>, Point3<f32>)> = Vec::new();
    for idx1 in 0..sp_list.len() {
        let coord1: Vector3<f32> = sp_list[idx1].data().local_translation().vector;
        let (x1, y1, z1) = (coord1.x, coord1.y, coord1.z);

        for idx2 in idx1 + 1..sp_list.len() {
            let coord2: Vector3<f32> = sp_list[idx2].data().local_translation().vector;
            let (x2, y2, z2) = (coord2.x, coord2.y, coord2.z);

            let prob_connect = connect_chance(c, dist(&(x1, y1, z1), &(x2, y2, z2)), lambda);
            let rand_num: f32 = rng.gen();

            if rand_num <= prob_connect {
                connects[(idx1, idx2)] = 1;
                connects[(idx2, idx1)] = 1;
                connect_lines.push((Point3::new(x1, y1, z1), Point3::new(x2, y2, z2)));
            }
        }
    }
    connect_lines
}

// Calculates the average number of connections per neuron
fn avg_connects(connections: &DMatrix<u32>, n: usize, c: f32, lambda: f32) {
    let mut sum_connects: u32 = 0;
    for connect in connections.iter() {
        sum_connects += connect;
    }
    let avg: f32 = (sum_connects - n as u32) as f32 / n as f32;
    println!(
        "\nC     : {}\nLambda: {}\nAverage number of connections per neuron: {}\n",
        c, lambda, avg
    );
}

fn main() {
    // Important Varaibles -------------------------------------------------------------------
    let mut window = Window::new("Neuron Clusters in Brain");
    window.set_light(Light::StickToCamera);

    let mut spheres: Vec<SceneNode> = Vec::new();
    let mut clusters: Vec<Vec<SceneNode>> = Vec::new();

    // number of neurons in a single cluster
    let mut n = 200;
    let mut var: f32 = 1.75;
    let r = 0.1;
    //----------------------------------------------------------------------------------------

    // Clusters ------------------------------------------------------------------------------

    // Generate a cluster by giving it a cluster center (-1., 2., -3.)
    // let c1000 = Point3::new(-1., 2., -3.);
    let c1 = Point3::new(2.5, 2.5, 3.0);
    let clust1: Vec<SceneNode> = cluster(&mut window, n, r, &c1, var, (0.2578, 0.5273, 0.957));
    clusters.push(clust1);
    // dist(clust1[0].data().local_translation(),
    // clust1[1].data().local_translation());

    /*
        n = 100;
        var = 1.55;
        // let c400 = Point3::new(5., -1.3, -3.4);
        let c2 = Point3::new(-2.5, 2.5, -1.0);
        let clust2: Vec<SceneNode> = cluster(&mut window, n, r, &c2, var, (0.2266, 0.875, 0.4023));
        clusters.push(clust2);
        // dist(clust1[0].data().local_translation(), clust2[1].data().local_translation());

        // let c600 = Point3::new(3., 6., 1.);
        let c3 = Point3::new(2.0, 1.5, -2.5);
        let clust3: Vec<SceneNode> = cluster(&mut window, n, r, &c3, var, (0.9453, 0.8203, 0.0938));
        clusters.push(clust3);

        n = 50;
        var = 1.;
        let c4 = Point3::new(-1.2, -0.5, 2.5);
        let clust4: Vec<SceneNode> = cluster(&mut window, n, r, &c4, var, (0.9453, 0.0938, 0.6641));
        clusters.push(clust4);
    */
    //----------------------------------------------------------------------------------------

    // Line Drawing --------------------------------------------------------------------------
    unpack(&clusters, &mut spheres);

    let sp_len = spheres.len();
    let mut connections = DMatrix::<u32>::zeros(sp_len, sp_len);
    connections.fill_diagonal(1); // Because all the neurons are connected to themselves

    // c = 0.15   lamda = 1.75    generate good results
    let c = 0.275;
    let lambda = 2.;
    let lines = make_connects(/*&mut window,*/ &spheres, &mut connections, c, lambda);
    //----------------------------------------------------------------------------------------

    // Analysis -----------------------------------------------------------------------------
    avg_connects(&connections, sp_len, c, lambda);
    //----------------------------------------------------------------------------------------

    // Rendering -----------------------------------------------------------------------------
    let axis_len = 10.;
    let axis_on = false;
    let lines_on = false;
    render_lines(&mut window, axis_on, axis_len, lines, lines_on);
    //----------------------------------------------------------------------------------------
}
