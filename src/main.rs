/*
/// Project: Simple Brain-like LSM Build
///
/// Authors: Awildo Gutierrez, Sampreeth Aravilli, Sosina Abuhay, Siqi Fang, Dave Perkins
///
/// Date: June 26, 2020
///
/// Description: We implement a neuron class with a visual representation
///
/// To do: o Finish Neuron class
///        o Input and output layers
///        o LSM Struct
///        o
///        o
*/

// Our imports
use std::fs::File;
use std::io::Write;

use kiss3d::camera::ArcBall;
use kiss3d::light::Light;
use kiss3d::scene::SceneNode;
use kiss3d::window::Window;

use nalgebra::base::DMatrix;
use nalgebra::geometry::Translation3;
use nalgebra::{Point3, Vector3};

use rand::prelude::*;
use rand_distr::{Distribution, Normal};

#[derive(Clone)]
pub struct Neuron {
    // maybe it should know which cluster it belongs to? and cluster could be a
    pub obj: SceneNode, // a sphere design associated with the neuron
    pub connects: Vec<u32>,
    // v: f32, // voltage input
    // theta: f32, // threshold to activate
    // v_rest: f32, // resting voltage
    // n_t : String,
    // input: bool,
    // read_out: bool,
}

impl Neuron {
    pub fn new(
        obj: SceneNode,
        connects: Vec<u32>, /*, v: f32, theta: f32, v_rest: f32, n_t: String, input: bool, read_out: bool*/
    ) -> Neuron {
        Self {
            obj: obj,
            connects: connects,
            // v: v,
            // theta: theta,
            // v_rest: v_rest
            // n_t: n_t
            // input: input,
            // read_out: read_out
        }
    }

    pub fn get_obj(&mut self) -> &mut SceneNode {
        &mut self.obj
    }

    pub fn set_connects(&mut self, connects: Vec<u32>) {
        self.connects = connects;
    }

    pub fn get_connects(&self) -> &Vec<u32> {
        &self.connects
    }
}

pub struct LSM {
    // Big structure of clustered up neurons and connections
    pub clusters: Vec<Vec<Neuron>>,
    pub neurons: Vec<Neuron>,
    pub connections: DMatrix<u32>,
    // readout: Vec<Neuron>,
    // output: Vec<
}

impl LSM {
    pub fn new(clusters: Vec<Vec<Neuron>>, neurons: Vec<Neuron>) -> LSM {
        Self {
            // maybe number of clusters, radius of brain sphere, etc
            clusters: clusters,
            neurons: neurons,
            connections: DMatrix::<u32>::zeros(0, 0), // Starts empty because you can't fill till later
        }
    }

    // Cluster Methods --------------------------------------------------------------------
    pub fn make_cluster(
        &mut self,
        window: &mut Window,        // Our screen
        n: usize,                   // The number of neurons we want in a cluster
        radius: f32,                // Radius of each sphere neuron
        loc: &Point3<f32>,          // Center of the cluster
        var: f32, // The variance in the distribution (Higher number means bell curve is wider)
        (r, g, b): (f32, f32, f32), // Color of the cluster
    ) {
        // Creates a cluster of n (r,g,b)-colored, radius sized neurons around a   \\
        // center at loc, distributed around the center with variance var.         \\
        // Returns a list of spheres in a cluster.                                 \\
        // let mut spheres: Vec<SceneNode> = Vec::new(); // our output, a vector of spheres
        let mut neurons: Vec<Neuron> = Vec::new();
        let mut rng = rand::thread_rng(); // a single random number

        for sphere in 0..n {
            // Normal takes mean and then variance
            let normal_x = Normal::new(loc[0], var).unwrap();
            let normal_y = Normal::new(loc[1], var).unwrap();
            let normal_z = Normal::new(loc[2], var).unwrap();

            // Generate a random point for a new neuron
            let t = Translation3::new(
                normal_x.sample(&mut rng),
                normal_y.sample(&mut rng),
                normal_z.sample(&mut rng),
            );

            let temp_connect: Vec<u32> = Vec::new();
            let neuron: Neuron = Neuron::new(window.add_sphere(radius), temp_connect);
            // Push the sphere into the spheres list
            neurons.push(neuron);
            neurons[sphere].get_obj().set_color(r, g, b);
            neurons[sphere].get_obj().append_translation(&t);
            // spheres.push(window.add_sphere(radius));

            // Set the color
            // spheres[sphere].set_color(r, g, b);
            // Move it by our translation t
            // spheres[sphere].append_translation(&t);
        }
        // spheres
        self.add_cluster(neurons);
        self.unpack();
    }

    fn unpack(&mut self) {
        // Puts all the neurons in the cluster into a vector of neurons. \\
        for neuron in self.clusters[self.clusters.len() - 1].iter() {
            self.neurons.push(neuron.clone()); // sphere.clone() to avoid moving problems
        }
    }

    fn add_cluster(&mut self, add_cluster: Vec<Neuron>) {
        self.clusters.push(add_cluster);
    }

    // Neuron Methods ---------------------------------------------------------------------
    fn get_neurons(&mut self) -> &mut Vec<Neuron> {
        &mut self.neurons
    }

    // Connection Methods -----------------------------------------------------------------
    pub fn make_connects(
        &mut self,
        c: f32, // This and lambda are hyper parameters for connect_chance function.
        lambda: f32,
    ) -> (Vec<(Point3<f32>, Point3<f32>, Point3<f32>)>, Vec<f32>)
// Returns a tuple of two vectors. 
// The first vector has two points that are the centers of two "connected"
// neurons, and one point containing the r, g, and b values for the color of the
// edge. 
// The second vector is a list of how long the edges are.
    {
        let n_len = self.neurons.len();
        let mut connects = DMatrix::<u32>::zeros(n_len, n_len);
        connects.fill_diagonal(1);

        // This function makes the edges between neurons based on a probability \\
        // function previously defined. We don't render the lines until later.  \\
        let mut connect_lines: Vec<(Point3<f32>, Point3<f32>, Point3<f32>)> = Vec::new();
        let mut dist_list: Vec<f32> = Vec::new();
        let mut rng = rand::thread_rng(); // some random number between 0 and 1 for computing probability
        for idx1 in 0..n_len {
            // To get the point location of a sphere, we call .data() to convert it to
            // a SceneNodeData object, then call the local_translation() function to
            // get the Translation3 object and we take the vector attribute of that which
            // is a Vector3.
            let coord1: Vector3<f32> = self.neurons[idx1]
                .get_obj()
                .data()
                .local_translation()
                .vector;
            // x, y, and z are components of a Vector3
            let (x1, y1, z1) = (coord1.x, coord1.y, coord1.z);

            // Let S = [s_1, s_2, s_3, ..., s_n] be a set of spheres. Then,
            //  1. Every neuron is connected to itself.
            //  2. A neural connection ab is the same as the connection ba.
            // We can represent connections in an n x n matrix. I'll show an
            // example for n = 4. An entry of 1 means there is a connection between
            // the row-th neuron and the col-th neuron.
            //
            // e.g.
            // idx   0 1 2 3
            //      _________
            //   0 | 1 0 0 0 |
            //   1 | 0 1 0 0 |
            //   2 | 0 0 1 0 |
            //   3 | 0 0 0 1 |
            // The connections on the main diagonal represent the connections of a neuron to itself (1).
            // Say s_1 was connected to s_2 and s_3 is connected to s_1. Then the matrix is
            //       0 1 2 3
            //      _________
            //   0 | 1 0 0 0 |
            //   1 | 0 1 0 1 |
            //   2 | 0 1 1 0 |
            //   3 | 0 0 0 1 |
            // However, using (2), we can say s_2 is also connected to s_1 and s_1 is also connected to s_3. So,
            // the matrix looks like
            // idx   0 1 2 3
            //      _________
            //   0 | 1 0 0 0 |
            //   1 | 0 1 1 1 |
            //   2 | 0 1 1 0 |
            //   3 | 0 1 0 1 |
            // This means that our connections matrix is symmetric, so we can cut our operations by over half to not
            // repeat checking entries. We only check the upper triangular half of the matrix.
            for idx2 in idx1 + 1..n_len {
                let coord2: Vector3<f32> = self.neurons[idx2]
                    .get_obj()
                    .data()
                    .local_translation()
                    .vector;
                let (x2, y2, z2) = (coord2.x, coord2.y, coord2.z);

                let d = self.dist(&(x1, y1, z1), &(x2, y2, z2)); // distance between the two points

                let prob_connect = self.connect_chance(d, c, lambda);
                // println!("Distance: {}\nConnect chance: {}", d, prob_connect);
                let rand_num: f32 = rng.gen();

                if rand_num <= prob_connect {
                    connects[(idx1, idx2)] = 1;
                    connects[(idx2, idx1)] = 1;
                    connect_lines.push((
                        Point3::new(x1, y1, z1), // point 1
                        Point3::new(x2, y2, z2), // point 2
                        Point3::new(1., 1., 1.), // color of edge
                    ));
                    dist_list.push(d); // edge length
                }
            }
        }
        self.add_connections(connects);
        self.update_n_connects();
        (connect_lines, dist_list)
    }

    pub fn remove_disconnects(&mut self) {
        // Removes neurons that are not connected to any other neurons \\
        let mut rm_n: Vec<usize> = Vec::new(); // You can collect and re-add these in
        for idx in 0..self.neurons.len() {
            let sum_connects: u32 = self.neurons[idx].get_connects().iter().sum();
            if sum_connects == 1 {
                self.neurons[idx].get_obj().set_visible(false);
                rm_n.push(idx); // You can collect and re-add these in
            }
        }
        for idx in 0..rm_n.len() {
            self.neurons.remove(rm_n[idx] - idx);
        }
    }

    fn connect_chance(
        &self,
        d_ab: f32,   // The distance between neurons a and b
        c: f32,      // A hyper parameter. At very close distances, the probability output is c.
        lambda: f32, // A hyper parameter. At larger values of lambda, the decay slows.
    ) -> f32 {
        // Computes the probability that two neurons are connected based on the     \\
        // distance between neurons a and b, and two hyper parameters C and lambda. \\
        // CITE: Maass 2002 paper pg. 18, connectivity of neurons.                  \\
        let exponent: f32 = -1. * ((d_ab / lambda).powf(2.));
        c * exponent.exp()
    }

    fn dist(
        &self,
        (x1, y1, z1): &(f32, f32, f32), // point 1
        (x2, y2, z2): &(f32, f32, f32), // point 2
    ) -> f32 {
        // Finds the Euclidean Distance between 2 3D points \\
        ((x2 - x1).powf(2.) + (y2 - y1).powf(2.) + (z2 - z1).powf(2.)).sqrt()
    }

    // The list of connections. 1 if there is a connection, 0 if not
    fn update_n_connects(&mut self) {
        // This function updates the neuron's connections list using the
        // connections matrix we made in make_connects. \\
        let n_len = self.neurons.len();
        for col in 0..n_len {
            let mut n_connect: Vec<u32> = Vec::new();
            for row in 0..n_len {
                n_connect.push(self.connections[(col, row)]);
            }

            // // Debug step
            // if col < 2 {
            //     println!(
            //         "Number of connects: {}\nNumber of neurons: {}",
            //         n_connect.len(),
            //         n_list.len()
            //     );
            // }

            self.neurons[col].set_connects(n_connect);
        }
    }

    fn add_connections(&mut self, connects: DMatrix<u32>) {
        self.connections = connects;
    }

    fn get_connections(&mut self) -> &mut DMatrix<u32> {
        &mut self.connections
    }
}

fn render_lines(
    window: &mut Window,                                 // Our window
    axis_on: bool, // True - axis lines visible, False - invisible
    axis_len: f32, // The length of the axis lines
    lines: Vec<(Point3<f32>, Point3<f32>, Point3<f32>)>, // The edges between neurons
    lines_on: bool, // True - edges between neurons, False - no edges
    l: &mut LSM, // List of neurons
    rm_dis_n: bool, // False - All Neurons, True - Remove Neuron with no connections
) {
    // Renders the edges between neurons as well as the lines of axis. \\

    // We want to start off at a point other than the origin so we don't have to
    // zoom out immediately.
    let eye = Point3::new(10.0f32, 10.0, 10.0);
    let at = Point3::origin();
    let mut first_person = ArcBall::new(eye, at);

    // Removes neurons that are not connected to any other neurons
    if rm_dis_n {
        l.remove_disconnects();
    }

    // First person allows for some useful user controls.
    while window.render_with_camera(&mut first_person) {
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
    data.push(format!("\nC     : {}\nLambda: {:.2}\n", c, lambda));
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
    let mut window = Window::new("Neuron Clusters in Brain"); // For graphics display
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
    let axis_len = 10.;
    let axis_on = false;
    let lines_on = true;
    let rm_dis_n = true;
    render_lines(
        &mut window,
        axis_on,
        axis_len,
        lines,
        lines_on,
        &mut l1,
        rm_dis_n,
    );

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
