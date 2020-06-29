use kiss3d::window::Window;

use nalgebra::base::DMatrix;
use nalgebra::geometry::Translation3;
use nalgebra::{Point3, Vector3};

use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};

mod neuron;
use neuron::Neuron;

pub struct LSM {
    // Big structure of clustered up neurons and connections
    clusters: Vec<Vec<Neuron>>,
    neurons: Vec<Neuron>,
    connections: DMatrix<u32>,
    n_total: usize,  // the number of total neurons
    n_inputs: usize, // the number of input neurons
    n_outputs: usize, // the number of output neurons
                     // readout: Vec<Neuron>,
                     // output: Vec<
}

impl LSM {
    pub fn new(n_inputs: usize, n_outputs: usize) -> LSM {
        Self {
            // maybe number of clusters, radius of brain sphere, etc
            clusters: Vec::new(),
            neurons: Vec::new(),
            n_total: 0, // at least n_inputs + n_outputs
            n_inputs: n_inputs,
            n_outputs: n_outputs,
            connections: DMatrix::<u32>::zeros(0, 0), // Starts empty because you can't fill until later
        }
    }

    // Cluster Methods \\
    pub fn make_cluster(
        &mut self,
        window: &mut Window,        // Our screen
        n: usize,                   // The number of neurons we want in a cluster
        radius: f32,                // Radius of each sphere neuron
        loc: &Point3<f32>,          // Center of the cluster
        var: f32, // The variance in the distribution (Higher number means bell curve is wider)
        (r, g, b): (f32, f32, f32), // Color of the cluster
    ) {
        if n == 0 {
            println!("No neurons.");
            return;
        }
        // Creates a cluster of n (r,g,b)-colored, radius sized neurons around a   \\
        // center at loc, distributed around the center with variance var.         \\
        // Returns a list of spheres in a cluster.                                 \\
        // let mut spheres: Vec<SceneNode> = Vec::new(); // our output, a vector of spheres
        let mut neurons: Vec<Neuron> = Vec::new();
        let seed = [1_u8; 32]; // our seed, the 1s can be changed later
                               // let mut rng = thread_rng(); // something like that
        let mut rng = StdRng::from_seed(seed); // a seeded (predictable) random number
                                               // println!("n: {}", n);
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
            // debug
            // if sphere < 2 {
            //     println!("{:?}", t);
            // }

            let temp_connect: Vec<u32> = Vec::new();
            let neuron: Neuron = Neuron::new(window.add_sphere(radius), temp_connect, "liq");
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
        // The last neuron list in clusters
        for neuron in self.clusters[self.clusters.len() - 1].iter() {
            self.neurons.push(neuron.clone()); // sphere.clone() to avoid moving problems
        }
        self.n_total = self.neurons.len(); // update total neuron count
    }

    fn assign_specs(&mut self) {
        // Assign input and readout neurons. \\
        assert_eq!(true, self.n_total >= self.n_inputs + self.n_outputs);
        if self.n_inputs == 0 || self.n_outputs == 0 {
            // println!("u stupid");
            return;
        }
        // tools: n_inputs, n_outputs
        // We want to choose n_inputs neurons from all the neurons created to be
        // inputs
        // So, don't stop until you reach n_inputs.
        // likewise with n_outputs
        let seed = [2; 32];
        let mut rng = StdRng::from_seed(seed);

        // indices assoc list, note that liq_idx[n] = n
        let mut liq_idx: Vec<usize> = (0..self.n_total).collect();
        // for (idx, val) in liq_idx.iter().enumerate() {
            
        // }

        // previous choices 
        // choose a random number between 0 and n_total, put it into previous
        // choices
        // next time, check that the number u choose is not a previous choice (!contains)
        let mut curr_ins: usize = 0;
        let mut curr_outs: usize = 0;

        while curr_ins < self.n_inputs {
            let in_idx: usize = *liq_idx.choose(&mut rng).unwrap();
            self.neurons[in_idx].set_spec("in");
            let idx = liq_idx.iter().position(|&r| r==in_idx).unwrap();
            liq_idx.remove(idx);
            curr_ins += 1;
        }

        while curr_outs < self.n_outputs {
            let out_idx: usize = *liq_idx.choose(&mut rng).unwrap();
            self.neurons[out_idx].set_spec("out");
            let idx = liq_idx.iter().position(|&r| r==out_idx).unwrap();
            liq_idx.remove(idx);
            curr_outs += 1;
        }
    }

    fn add_cluster(&mut self, add_cluster: Vec<Neuron>) {
        self.clusters.push(add_cluster);
    }

    // Neuron Methods ---------------------------------------------------------------------
    pub fn get_neurons(&mut self) -> &mut Vec<Neuron> {
        &mut self.neurons
    }

    // Connection Methods -----------------------------------------------------------------
    // Returns a tuple of two vectors.
    // The first vector has two points that are the centers of two "connected"
    // neurons, and one point containing the r, g, and b values for the color of the
    // edge.
    // The second vector is a list of how long the edges are.
    pub fn make_connects(
        &mut self,
        c: f32, // This and lambda are hyper parameters for connect_chance function.
        lambda: f32,
    ) -> (Vec<(Point3<f32>, Point3<f32>, Point3<f32>)>, Vec<f32>) {
        self.assign_specs();
        let n_len = self.neurons.len();
        let mut connects = DMatrix::<u32>::zeros(n_len, n_len);
        connects.fill_diagonal(1);

        // This function makes the edges between neurons based on a probability \\
        // function previously defined. We don't render the lines until later.  \\
        let mut connect_lines: Vec<(Point3<f32>, Point3<f32>, Point3<f32>)> = Vec::new();
        let mut dist_list: Vec<f32> = Vec::new();
        // let mut rng = rand::thread_rng(); // some random number between 0 and 1 for computing probability
        let seed = [0; 32];
        let mut rng = StdRng::from_seed(seed);
        // rng.gen::<f32>() for generating a (fixed) random number
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
            //   repeat checking entries. We only check the upper triangular
            //   half of the matrix.

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
                let rand_num: f32 = rng.gen::<f32>();
                // if idx1 < 3 && idx2 < 3 {
                //     println!("{}", rand_num);
                // }

                if rand_num <= prob_connect {
                    connects[(idx1, idx2)] = 1;
                    connects[(idx2, idx1)] = 1;
                    connect_lines.push((
                        Point3::new(x1, y1, z1),    // point 1
                        Point3::new(x2, y2, z2),    // point 2
                        Point3::new(0.8, 0.8, 0.8), // color of edge
                    ));
                    dist_list.push(d); // edge length
                }
            }
        }
        self.add_connections(connects);
        self.update_n_connects();
        (connect_lines, dist_list)
    }

    pub fn remove_disconnects(&mut self, window: &mut Window) {
        // Removes neurons that are not connected to any other neurons \\
        let mut rm_n: Vec<usize> = Vec::new(); // You can collect and re-add these in
        for idx in 0..self.neurons.len() {
            let sum_connects: u32 = self.neurons[idx].get_connects().iter().sum();
            if sum_connects == 1 {
                // self.neurons[idx].get_obj().set_visible(false);
                window.remove_node(self.neurons[idx].get_obj());
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

    pub fn get_connections(&mut self) -> &mut DMatrix<u32> {
        &mut self.connections
    }
}
