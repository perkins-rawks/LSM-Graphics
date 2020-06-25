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

use nalgebra::geometry::Translation3;
use nalgebra::Point3;

use rand::prelude::*;
use rand_distr::{Distribution, Normal};
// let val: f64 = thread_rng().sample(StandardNormal);

// window, number of neurons, location to be centered around, variance (for bell
// curve)
fn cluster(window: &mut Window, n: usize, loc: &Point3<f32>, var: f32) {
    let mut spheres: Vec<SceneNode> = Vec::new();
    let mut rng = rand::thread_rng(); // a single random number
    let color = (rng.gen(), rng.gen(), rng.gen());

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

        spheres.push(window.add_sphere(0.1));
        spheres[sphere].set_color(color.0, color.1, color.2);
        spheres[sphere].append_translation(&t);
    }

    // spheres
}

fn main() {
    let eye = Point3::new(10.0f32, 10.0, 10.0);
    let at = Point3::origin();
    let mut first_person = ArcBall::new(eye, at);

    let mut window = Window::new("Neuron Cluters in Brain");
    window.set_light(Light::StickToCamera);

    // number of neurons in a single cluster
    let mut n = 1000;
    let mut var: f32 = 1.0;

    // Generate a cluster by giving it a clucter center (-1., 2., -3.)
    let c1 = Point3::new(-1., 2., -3.);
    cluster(&mut window, n, &c1, var);

    n = 500;
    var = 0.75;
    let c2 = Point3::new(5., -1.3, -3.4);
    cluster(&mut window, n, &c2, var);

    let c3 = Point3::new(3., 6., 1.);
    cluster(&mut window, n, &c3, var);

    // This renders whatever
    let scale = 10.;
    while window.render_with_camera(&mut first_person) {
        window.draw_line(
            &Point3::new(-scale, 0.0, 0.0),
            &Point3::new(scale, 0.0, 0.0),
            &Point3::new(1.0, 0.0, 0.0),
        );
        window.draw_line(
            &Point3::new(0.0, -scale, 0.0),
            &Point3::new(0.0, scale, 0.0),
            &Point3::new(0.0, 1.0, 0.0),
        );
        window.draw_line(
            &Point3::new(0.0, 0.0, -scale),
            &Point3::new(0.0, 0.0, scale),
            &Point3::new(0.0, 0.0, 1.0),
        );
    }
}
