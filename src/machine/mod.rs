use kiss3d::scene::SceneNode;
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
use neuron::CLUSTERS;

use std::io::Write;
use std::time::Instant;
use std::{fs, fs::File};

pub struct LSM {
    // Big structure of clustered up neurons and connections
    clusters: Vec<Vec<Neuron>>,         // A cluster is a set of neurons
    neurons: Vec<Neuron>,               // List of all neurons // has ownership of neurons
    connections: DMatrix<u32>,          // List of all connections (1 if connected, 0 if not)
    n_total: usize,                     // The number of total neurons
    n_inputs: usize,                    // The number of input neurons
    n_input_copies: usize,              // The number of copies of the input is sent into the LSM
    n_clusters: usize,                  // The number of cluster neurons
    n_readout_clusters: usize,          // The number of readout clusters in a single cluster
    e_ratio: f32,                       // 0-1 the ratio of excitatory to inhibitory neurons
    readouts: Vec<Vec<Neuron>>,         // List of all clusters of read out neurons
    readout_weights: Vec<DMatrix<f32>>, // List of all the plastic weights between readouts and reservoir
    input_layer: Vec<Neuron>,           // Set of neurons that read input outside the reservoir
    tau_m: u32,                         // time constant for membrane potential (ms)
    tau_c: u32,                         // Time constant for calcium decay
    liq_weights: [i32; 4],              // Fixed synaptic weights in the reservoir
    r_weight_max: f32,                  // The maximum weight between readout and reservoir
    r_weight_min: f32,                  // The minimum weight between readout and reservoir
    // delta_weight: f32,
    readouts_c_r: Vec<Vec<Vec<f32>>>, // Each readout neuron's calcium values from each cluster for each epoch
    readouts_c_d: Vec<Vec<Vec<f32>>>, // Each readout neuron's desired calcium values from each cluster for each epoch
    outputs: Vec<String>,             // The output of the LSM at any epoch
    c_th: f32,                        // Calcium threshold for a readout neuron
    c_margin: f32,                    // Margin around the Calcium threshold
    responsible_liq: Vec<Vec<f32>>, // The neurons indices that cause spikes for each readout in one time step
}

impl LSM {
    pub fn new(
        n_inputs: usize,
        n_input_copies: usize,
        n_clusters: usize,
        n_readout_clusters: usize,
        e_ratio: f32,
    ) -> LSM {
        Self {
            clusters: Vec::new(),
            neurons: Vec::new(),
            connections: DMatrix::<u32>::zeros(0, 0), // Starts empty because you can't fill until later
            n_total: n_inputs,                        // at least n_inputs + n_outputs
            n_inputs: n_inputs,
            n_input_copies: n_input_copies,
            n_clusters: n_clusters,
            n_readout_clusters: n_readout_clusters,
            e_ratio: e_ratio,
            readouts: Vec::new(),
            readout_weights: Vec::new(),
            input_layer: Vec::new(),
            tau_m: 32,
            tau_c: 64,
            liq_weights: [3, 6, -2, -2], // [EE, EI, IE, II]
            r_weight_max: 8.,            // from weight-max calc
            r_weight_min: -8.,           // from weight_min calc
            // delta_weight: 0.0002 * (2 as f32).powf(13. - 4.), // 0.05 0.008 // 0.0002 * 2 ^ (n_bits - 4) but scaled down
            readouts_c_r: Vec::new(),
            readouts_c_d: Vec::new(),
            outputs: Vec::new(),
            c_th: 5.,
            c_margin: 3.,
            responsible_liq: Vec::new(),
        }
    }

    // Cluster Methods \\
    pub fn make_cluster(
        &mut self,
        window: &mut Window,        // Our screen
        n: usize,                   // The number of neurons we want in a cluster
        radius: f32,                // Radius of each sphere neuron
        var: f32, // The variance in the distribution (Higher number means bell curve is wider)
        cluster: &str, // The task for which the cluster is created (talk, eat, etc.)
        loc: &Point3<f32>, // Center of the cluster
        (r, g, b): (f32, f32, f32), // Color of the cluster
        n_readouts: usize,
    ) {
        if n == 0 {
            println!("No neurons.");
            return;
        }
        // Creates a cluster of n (r,g,b)-colored, radius sized neurons around a
        // center at loc, distributed around the center with variance var.
        // Also creates the readout set associated with each cluster and assigns
        // the neurons in it to be input randomly. At the end, the cluster is
        // stored in a cluster list, and its neurons are stored in an overall
        // neurons list variable.
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

            // let temp_connect: Vec<u32> = Vec::new();
            let neuron: Neuron = Neuron::new(window.add_sphere(radius), "liq", cluster);
            // Push the new sphere into the spheres list
            neurons.push(neuron);
            // let mut curr_n: &Neuron = &neurons[sphere];
            neurons[sphere].get_obj().set_color(r, g, b);
            neurons[sphere].get_obj().append_translation(&t);
            neurons[sphere].set_loc(&t);
            // get obj returns the sphere associated with a neuron
        }

        self.assign_input(&mut neurons);
        let readouts: Vec<Neuron> =
            self.make_readouts(window, n_readouts, &loc, radius, cluster.clone());
        self.add_readouts(readouts);
        self.add_cluster(neurons);
        self.unpack(); //flattens the cluster list
    }

    fn unpack(&mut self) {
        // Puts all the neurons in the cluster into a vector of neurons. \\
        // Also updates node identifications to be their index in the neurons list.
        // The last neuron list in clusters
        for (id, neuron) in self.clusters[self.clusters.len() - 1].iter().enumerate() {
            self.neurons.push(neuron.clone()); // .clone() to avoid moving problems

            // Setting ID based on previous ID
            if id == 0 && self.clusters.len() == 1 {
                self.neurons[id].set_id(0); // base case
            } else {
                let last_idx = self.neurons.len() - 1;
                let prev_id: usize = self.neurons[last_idx - 1].get_id();
                self.neurons[last_idx].set_id(prev_id + 1);
            }
        }
        self.n_total = self.neurons.len(); // update total neuron count

        // Readout IDS
        // For each readout in the most recent readout set made, set an
        // appropriate ID

        // INDEX WISE
        // assuming each cluster has the same # of neurons
        // We CANNOT use self.n_total because the LSM is not filled yet. This is
        // only called in the process of setting up the LSM.
        let count: usize = self.clusters[0].len(); // number of neurons in a single cluster
        let r_len: usize = self.readouts.len();
        // For each readout in the most recent readout set made
        for r_idx in 0..self.readouts[r_len - 1].len() {
            if r_idx == 0 && r_len == 1 {
                // The very first readout neuron being setup \\
                self.readouts[r_len - 1][r_idx].set_id(count * self.n_clusters);
            } else if r_idx == 0 && r_len > 1 {
                // If NOT the first readout cluster and the index is 0 \\
                let last_readout: &Neuron = &self.readouts[r_len - 2] // prev cluster
                    [self.readouts[r_len - 2].len() - 1]; // last item in that prev cluster
                let prev_id: usize = last_readout.get_id();
                self.readouts[r_len - 1][r_idx].set_id(prev_id + 1);
            } else {
                // Generally \\
                let last_readout: &Neuron = &self.readouts[r_len - 1][r_idx - 1];
                let prev_id: usize = last_readout.get_id();
                self.readouts[r_len - 1][r_idx].set_id(prev_id + 1);
            }
        }
    }

    fn assign_input(&mut self, neurons: &mut Vec<Neuron>) {
        // Assign input neurons. All neurons are also output now. \\
        // assert_eq!(true, self.n_total >= self.n_inputs);
        assert_eq!(true, self.n_inputs > 0);
        // assert_eq!(true, self.n_input_copies * self.n_inputs <= self.n_total);

        // We want to choose n_inputs neurons from all the neurons created to be
        // inputs.
        let seed = [2; 32];
        let mut rng = StdRng::from_seed(seed);

        // This list has the indices of every unique choice for an input neuron
        // in this struct 'neurons' list.
        let mut liq_idx: Vec<usize> = (0..neurons.len()).collect();

        for _ in 0..self.n_input_copies {
            for _ in 0..self.n_inputs {
                // choose chooses a random element of the list.
                let in_idx: usize = *liq_idx.choose(&mut rng).unwrap();
                // Set to input, change the color, and remove that index so that we
                // get unique input neurons
                neurons[in_idx].set_spec("liq_in");
                neurons[in_idx].get_obj().set_color(0.9453, 0.0938, 0.6641);
                // the index of in_idx in liq_idx
                let idx = liq_idx.iter().position(|&r| r == in_idx).unwrap();
                liq_idx.remove(idx);
            }
        }
    }

    fn assign_nt(&mut self) {
        assert_eq!(true, self.e_ratio <= 1.);
        // For a small LSM (2 nodes), make them both excitatory
        if self.n_total < 2 {
            for n in self.neurons.iter_mut() {
                n.set_nt("exc");
            }
            return;
        }

        // Calculates the number of excitatory and inhibitory neurons
        let n_exc: usize = (self.n_total as f32 * self.e_ratio) as usize;
        let n_inh: usize = self.n_total - n_exc;

        let seed = [3; 32];
        let mut rng = StdRng::from_seed(seed);

        // This list has the indices of every unique choice for an input neuron
        // in this struct 'neurons' list.
        let mut liq_idx: Vec<usize> = (0..self.n_total).collect();
        // Setting the excitatory neurons
        for _ in 0..n_exc {
            // Choose picks a random element of liq_idx
            let exc_idx: usize = *liq_idx.choose(&mut rng).unwrap();
            // Since we picked out an index, we can use that value to set the type
            self.neurons[exc_idx].set_nt("exc");
            self.neurons[exc_idx].set_second_tau(4, 8);
            // Find the idx at which it chose that index and remove it
            let idx = liq_idx.iter().position(|&r| r == exc_idx).unwrap();
            liq_idx.remove(idx);
        }
        // The same thing as above for inhibitory
        for _ in 0..n_inh {
            let inh_idx: usize = *liq_idx.choose(&mut rng).unwrap();
            self.neurons[inh_idx].set_nt("inh");
            self.neurons[inh_idx].set_second_tau(4, 2);
            let idx = liq_idx.iter().position(|&r| r == inh_idx).unwrap();
            liq_idx.remove(idx);
        }
    }

    fn add_cluster(&mut self, add_cluster: Vec<Neuron>) {
        self.clusters.push(add_cluster);
    }

    pub fn get_neurons(&mut self) -> &mut Vec<Neuron> {
        &mut self.neurons
    }

    fn add_readouts(&mut self, readouts: Vec<Neuron>) {
        self.readouts.push(readouts);
    }

    fn make_readouts(
        &mut self,
        window: &mut Window, // Window we will draw on
        n_readouts: usize,   // Total number of readout neurons that we want
        loc: &Point3<f32>,   // The center of the Cluster
        radius: f32,         // The radius of each readout neuron
        _cluster: &str,      // The cluster type
    ) -> Vec<Neuron> {
        // Returns a set of neurons.
        assert_eq!(n_readouts % self.n_readout_clusters, 0);

        // Makes a \\
        let mut readouts: Vec<Neuron> = Vec::new();
        for idx in 0..n_readouts {
            let x: f32 = loc.x * 2.5;
            let y: f32 = loc.y;
            let z: f32 = loc.z * 2.5;

            let t = Translation3::new(x, y + n_readouts as f32 / 2. - idx as f32, z);

            // less generally
            // let neuron: Neuron;
            // if idx >= n_readouts / 2 {
            //     neuron = Neuron::new(window.add_sphere(radius), "readout", "hide");
            // } else {
            //     neuron = Neuron::new(window.add_sphere(radius), "readout", "talk");
            // }
            // more generally
            let function: &str = CLUSTERS[idx % self.n_readout_clusters];
            let neuron: Neuron = Neuron::new(window.add_sphere(radius), "readout", function);
            readouts.push(neuron);

            readouts[idx].get_obj().append_translation(&t);
            readouts[idx].set_loc(&t);
        }
        readouts
    }

    pub fn make_connects(
        &mut self,
        window: &mut Window,
        c: [f32; 5], // This and lambda are hyper parameters for connect_chance function.
        lambda: f32,
    ) -> (
        Vec<(Point3<f32>, Point3<f32>, Point3<f32>)>,
        Vec<f32>,
        Vec<(Point3<f32>, Point3<f32>, Point3<f32>)>,
    ) {
        // Connection Methods \\
        // Returns a tuple of two vectors.
        // The first vector has two points that are the centers of two "connected"
        // neurons, and one point containing the r, g, and b values for the color of the
        // edge.
        // The second vector is a list of how long the edges are.
        //self.assign_input();
        self.set_input_weights();
        self.assign_nt();
        let n_len = self.n_total;
        let mut connects = DMatrix::<u32>::zeros(n_len, n_len);

        // This function makes the edges between neurons based on a probability \\
        // function previously defined. We don't render the lines until later.  \\
        let mut connect_lines: Vec<(Point3<f32>, Point3<f32>, Point3<f32>)> = Vec::new();
        let mut dist_list: Vec<f32> = Vec::new();
        // let mut rng = rand::thread_rng(); // some random number between 0 and 1 for computing probability
        let seed = [12; 32];
        let mut rng = StdRng::from_seed(seed);
        // rng.gen::<f32>() for generating a (fixed) random number
        for idx1 in 0..n_len {
            let coord1: &Vector3<f32> = self.neurons[idx1].get_loc();
            // x, y, and z are components of a Vector3
            let (x1, y1, z1) = (coord1.x, coord1.y, coord1.z);
            let idx1_nt: String = self.neurons[idx1].get_nt().clone();

            // We can represent connections in an n x n (column major) matrix where n is the
            // total size of the LSM. Suppose we have 4 neurons in the LSM, and
            // neuron 0 is connected to neuron 3 and neuron 1 is connected to 2.
            // Then connections looks like
            // 0 0 0 1
            // 0 0 1 0
            // 0 0 0 0  (Note the neuron is not connected to itself, so the)
            // 0 0 0 0  (main diagonal is always clear.)
            for idx2 in 0..n_len {
                // If it's part of the diagonal, then leave it as 0
                if idx1 == idx2 {
                    continue;
                }

                let mut c_idx: usize = 0;
                let idx2_nt: String = self.neurons[idx2].get_nt().clone();
                // starts with E
                if idx1_nt == "exc".to_string() {
                    // ends with I
                    if idx2_nt == "inh".to_string() {
                        c_idx = 1;
                    }
                // if it doesn't end with E, then it needs to be 0, but it
                // already is 0, so don't change it
                // starts with I
                } else {
                    // ends with E
                    if idx2_nt == "exc".to_string() {
                        c_idx = 2;
                    // ends with I
                    } else if idx2_nt == "inh".to_string() {
                        c_idx = 3;
                    }
                }

                // the connection going the other way is on
                if connects[(idx2, idx1)] == 1 {
                    c_idx = 4; // a loop
                } else if self.neurons[idx2].get_spec() == &"liq_in".to_string()
                    && self.neurons[idx1].get_spec() != &"liq_in".to_string()
                {
                    // if input neuron
                    c_idx = 4;
                }

                let coord2: &Vector3<f32> = self.neurons[idx2].get_loc();
                let (x2, y2, z2) = (coord2.x, coord2.y, coord2.z);
                // either exc or inh

                let d = self.dist(&(x1, y1, z1), &(x2, y2, z2)); // distance between the two points

                // Choosing the correct weight based on the combination of
                // pre-synaptic  to postsynaptic type (They can be either
                // excitatory inhibitory)
                // c = [EE, EI, IE, II]

                // make connections based on distance and some hyper parameters
                let prob_connect = self.connect_chance(d, c[c_idx], lambda);
                // if self.neurons[idx1] ==
                let rand_num: f32 = rng.gen::<f32>();

                if rand_num <= prob_connect {
                    connects[(idx1, idx2)] = 1;
                    connect_lines.push((
                        Point3::new(x1, y1, z1),    // point 1
                        Point3::new(x2, y2, z2),    // point 2
                        Point3::new(0.9, 0.9, 0.9), // color of edge
                    ));
                    dist_list.push(d); // edge length
                }
            }
        }
        self.update_pre_syn_connects(&connects);
        self.add_connections(connects);
        let readout_lines = self.make_readout_connects();
        self.make_input_layer(window);
        self.make_input_connects();
        // self.update_n_connects();
        (connect_lines, dist_list, readout_lines)
    }

    fn update_pre_syn_connects(&mut self, connects: &DMatrix<u32>) {
        // Updates each neuron's pre-synaptic connections

        // Basic algorithm:
        // Go through all the neurons, (column wise in the connects DMatrix)
        // For each neuron connecting to it, add that number to a list (that is
        // representing a neuron being connected to it)

        // We know that connects is n_total x n_total
        for col in 0..self.n_total {
            let mut pre_connects_idx: Vec<usize> = Vec::new();
            for row in 0..self.n_total {
                // This order of index-ing results in getting pre-synaptic connects
                // This row connects to col (col is our current neuron)
                if connects[(row, col)] == 1 {
                    pre_connects_idx.push(row);
                }
            }
            self.neurons[col].set_pre_syn_connects(pre_connects_idx);
        }
    }

    pub fn load_readout_weights(&mut self) {
        // IN ORDER: "HIDE", "RUN", "EAT"
        // WORKS ONLY FOR UNO CLUSTER
        let count = self.n_total / self.n_clusters;
        let readout_len = self.readouts[0].len();

        let contents =
            fs::read_to_string("trained_weights.txt").expect("Unable to read input file");
        let contents: Vec<&str> = contents.split("\n").collect();
        let mut weights = DMatrix::<f32>::zeros(readout_len, count);
        for (r_idx, line) in contents.iter().enumerate() {
            let new_line: Vec<&str> = line.trim().split(", ").collect();
            for (n_idx, num) in new_line.iter().enumerate() {
                // println!("{}", num);
                weights[(r_idx, n_idx)] = num.parse::<f32>().unwrap();
            }
        }
        self.readout_weights = vec![weights];
    }

    fn make_readout_connects(&mut self) -> Vec<(Point3<f32>, Point3<f32>, Point3<f32>)> {
        // connect_lines: <(point 1, point 2, color), ... >
        assert_eq!(self.readout_weights.is_empty(), true);
        let mut connect_lines: Vec<(Point3<f32>, Point3<f32>, Point3<f32>)> = Vec::new();
        let seed = [99; 32];
        let mut rng1 = StdRng::from_seed(seed);
        // let mut rng1 = rand::thread_rng();

        // Number of neurons in one cluster, assuming all clusters are equally sized
        let count = self.n_total / self.n_clusters;
        for cluster_idx in 0..self.n_clusters {
            // Number of readout neurons in a given cluster
            let readout_len = self.readouts[cluster_idx].len();
            // Each column is all the connections for one readout neuron.
            let mut readout_weights = DMatrix::<f32>::zeros(readout_len, count);
            // For all readout neurons in a readout set
            for readout_idx in 0..readout_len {
                let r_loc = self.readouts[cluster_idx][readout_idx].get_loc();
                let (x1, y1, z1) = (r_loc.x, r_loc.y, r_loc.z);

                // Steps by the number of neurons in a cluster
                // For each neuron, save its location
                for neuron_idx in count * cluster_idx..count * (cluster_idx + 1) {
                    let n_loc = self.neurons[neuron_idx].get_loc();
                    let (x2, y2, z2) = (n_loc.x, n_loc.y, n_loc.z);
                    connect_lines.push((
                        Point3::new(x1, y1, z1),
                        Point3::new(x2, y2, z2),
                        Point3::new(227. / 255., 120. / 255., 105. / 255.), // light pink
                    ));

                    // 'The neuron_idx % count' is to sort by cluster
                    // All readout weights are set to 1 at first
                    readout_weights[(readout_idx, neuron_idx % count)] =
                        (rng1.gen::<f32>() * 2.) - 1.;
                }
            }
            self.readout_weights.push(readout_weights);
        }
        connect_lines
    }

    pub fn get_readouts(&mut self) -> &mut Vec<Vec<Neuron>> {
        &mut self.readouts
    }

    pub fn remove_disconnects(&mut self, window: &mut Window) {
        // Removes neurons that are not connected to any other neurons \\
        // You can collect and re-add these in
        // for idx in 0..self.neurons.len() {
        //     let sum_connects: u32 = self.neurons[idx].get_connects().iter().sum();
        //     if sum_connects == 1 {
        //         // self.neurons[idx].get_obj().set_visible(false);
        //         window.remove_node(self.neurons[idx].get_obj());
        //         rm_n.push(idx); // You can collect and re-add these in
        //     }
        // }
        let mut rm_n: Vec<usize> = Vec::new();
        for col in 0..self.n_total {
            let mut connected: bool = false; // if current neuron has a connection, true
            for row in 0..self.n_total {
                // We need to remove only if the neuron has no connections into
                // or out from it.
                if self.connections[(col, row)] == 1 || self.connections[(row, col)] == 1 {
                    connected = true;
                    break;
                }
            }
            if !connected {
                window.remove_node(self.neurons[col].get_obj());
                rm_n.push(col);
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

    fn add_connections(&mut self, connects: DMatrix<u32>) {
        self.connections = connects;
    }

    pub fn get_connections(&mut self) -> &mut DMatrix<u32> {
        &mut self.connections
    }

    fn make_input_layer(&mut self, window: &mut Window) {
        // Creates neurons that will feed into the reservoir's input
        // How many? There will be n_inputs of them
        // Where are they located? Somewhere away from clusters.
        // What do they do? This function doesn't know.
        let mut sp = window.add_sphere(0.1);
        sp.set_visible(false);
        let mut empty = SceneNode::new_empty();
        empty.set_visible(false);
        for _ in 0..self.n_inputs {
            self.input_layer
                .push(Neuron::new(empty.clone(), "in", "input layer"));
        }
    }

    fn set_input_weights(&mut self) {
        // Sets the weight of input for one neuron to be either -8 or 8
        // FROM Zhang et al 2015 pg 2645
        let seed = [10; 32];
        let mut rng = StdRng::from_seed(seed);
        let percentage = self.e_ratio.clone();

        let weights = [-8, 8];
        for neuron in self.get_neurons().iter_mut() {
            if neuron.get_spec() == &"liq_in".to_string() {
                // neuron.set_input_weight(weights.choose(&mut
                // rng).unwrap().clone());
                if rng.gen::<f32>() < percentage {
                    neuron.set_input_weight(weights[1]);
                } else {
                    neuron.set_input_weight(weights[0]);
                }
            }
        }
    }

    fn make_input_connects(&mut self) {
        // Connects the outside input layer to an input neuron in each cluster
        // in the reservoir

        // Updates a neuron classified as "liq_in"'s input_connect attribute
        // This represents the index of that neuron in LSM input_layer vector

        let count: usize = self.n_total / self.n_clusters;
        for cluster in 0..self.n_clusters {
            let mut input_idx: usize = 0;
            for neuron_idx in count * cluster..count * (cluster + 1) {
                // Connect an input layer neuron to one reservoir input in each cluster
                if input_idx >= self.n_inputs {
                    input_idx = 0;
                }
                if self.neurons[neuron_idx].get_spec() == &"liq_in".to_string() {
                    self.neurons[neuron_idx].set_input_connect(input_idx);
                    input_idx += 1;
                }
            }
        }
    }

    fn set_spike_times(&mut self, input: &Vec<Vec<u32>>) {
        assert_eq!(self.n_inputs, input.len());
        // For each liq_in neuron's pre-synaptic connects, set the
        // spike_times array to be the index of 1's in the input vector

        // We know input is as long as the n_inputs.
        // For all the inputs in the reservoir
        for input_idx in 0..self.n_inputs {
            // Go through input spike, if there is a 1, we put the index that we found
            // it in.
            let mut curr_spike_times: Vec<u32> = Vec::new();
            for (key, item) in input[input_idx].iter().enumerate() {
                if item == &1 {
                    curr_spike_times.push(key as u32);
                }
            }
            self.input_layer[input_idx].set_spike_times(curr_spike_times.clone());
        }
    }

    fn delta(&self, n: i32) -> f32 {
        // Kronecker / Dirac delta function
        // A spike helper function
        // It outputs 1 only at 0 (Dirac outputs infinity at 0)
        if n == 0 {
            return 1.;
        }
        0.
    }

    fn heaviside(&self, n: i32) -> f32 {
        // Heaviside step function
        // When n < 0, H(n) = 0
        //      n > 0, H(n) = 1
        //      n = 0, H is undefined, but we put 1
        if n </*=*/ 0 {
            return 0.;
        }
        1.
    }

    fn delta_calcium(&self, n: i32) -> f32 {
        // Kronecker / Dirac delta function
        // A spike helper function
        // It outputs 1 only at 0 (Dirac outputs infinity at 0)
        // let seed = [13; 32];
        // let mut rng = StdRng::from_seed(seed);
        if n == 0 {
            return 1.;
        }
        0.
    }

    fn liq_response(
        &self,
        model: &str,
        curr_t: i32,
        t_spike: i32,
        delay: i32,
        first_tau: u32,
        tau_s1: u32,
        tau_s2: u32,
    ) -> f32 {
        if model == "static" {
            return self.delta(curr_t - t_spike - delay);
        } else if model == "first order" {
            let exponent = ((curr_t - t_spike - delay) as f32 * -1.) / (first_tau as f32);
            return (1. / first_tau as f32)
                * f32::exp(exponent)
                * self.heaviside(curr_t - t_spike - delay);
        } else if model == "second order" {
            // returns two different things
            // one with exponent1 and the other with exponent2
            let exponent1: f32 = ((curr_t - t_spike - delay) as f32 * -1.) / (tau_s1 as f32);
            let exponent2: f32 = ((curr_t - t_spike - delay) as f32 * -1.) / (tau_s2 as f32);
            let denominator: f32 = tau_s1 as f32 - tau_s2 as f32;
            let h = self.heaviside(curr_t - t_spike - delay);
            let part1 = f32::exp(exponent1);
            let part2 = f32::exp(exponent2);
            // In equation 21, if we factor out w, we can just
            // subtract the two parts together instead of calculate
            // two separately
            return (part1 - part2) * h / denominator;
        }
        panic!("Model was chosen incorrectly");
    }

    fn string_to_seed(&self, word: &str) -> [u8; 32] {
        let words = word.as_bytes();
        let words: Vec<String> = words.iter().map(|bit| bit.to_string()).collect();
        let seed_str = words.concat();
        let seed_chars: Vec<char> = seed_str.chars().collect();

        assert_eq!(seed_str.len() < 65, true);
        let mut seed = [0_u8; 32];
        let mut mini_seed: String;
        for char_idx in (0..seed_str.len() - 1).step_by(2) {
            mini_seed = [
                seed_chars[char_idx].to_string(),
                seed_chars[char_idx + 1].to_string(),
            ]
            .concat();
            seed[char_idx / 2] = mini_seed.parse().unwrap();
        }
        seed
    }

    pub fn talk(&self, word: &str, n_time_steps: u32) -> Vec<Vec<u32>> {
        let seed = self.string_to_seed(word);
        let mut rng = StdRng::from_seed(seed);
        let prob_of_one = 0.667;
        let prob_noise = 0.05;

        let mut spike_trains: Vec<Vec<u32>> = Vec::new();

        for _ in 0..self.n_inputs {
            let mut spike_train: Vec<u32> = Vec::new();
            if rng.gen::<f32>() < prob_of_one {
                for _ in 0..n_time_steps {
                    if rng.gen::<f32>() < prob_noise {
                        spike_train.push(0);
                    } else {
                        spike_train.push(1);
                    }
                }
            } else {
                for _ in 0..n_time_steps {
                    if rng.gen::<f32>() < prob_noise {
                        spike_train.push(1);
                    } else {
                        spike_train.push(0);
                    }
                }
            }
            spike_trains.push(spike_train);
        }
        spike_trains
    }

    pub fn run(
        &mut self,
        train: bool,
        epoch: usize,
        f1: &mut File,
        f2: &mut File,
        f3: &mut File,
        f4: &mut File,
        f5: &mut File,
        input: &Vec<Vec<u32>>,
        label: &String,
        model: &str,
        delay: i32,
        first_tau: u32,
        prev_epoch_accuracy: f32,
    ) -> String {
        // Updates voltage connections for all neurons in the LSM.
        // Implementation of Equation 14, 19, 20, 21 in Zhang et al 2015

        /*
          3 Different kinds of running:
            o Input layer to reservoir input neurons
            o Any liquid to liquid
            o Reservoir to readout neurons
        */
        if epoch == 0 {
            for col in 0..self.n_readout_clusters {
                f5.write_all(
                    format!("{}: [", CLUSTERS[self.readouts[0][col].get_cluster()],).as_bytes(),
                )
                .expect("Unable to write weights");
                for row in 0..self.n_total {
                    f5.write_all(format!("{}, ", self.readout_weights[0][(col, row)]).as_bytes())
                        .expect("Unable to write weights");
                }
                f5.write_all(format!("]\n").as_bytes())
                    .expect("Unable to write weights");
            }
        }

        let now = Instant::now();

        self.set_spike_times(input);

        let input_time_steps: usize = input[0].len();

        self.responsible_liq = vec![vec![0.; self.n_total]; self.readouts[0].len()];

        // let additional_delay: i32 = {
        //     let in_to_liq_delay = delay;
        //     let zero_to_spike_delay = 3; // v_th / max_input = 20 /8
        //     let liq_to_readout_delay = delay;
        //     let margin = 1;
        //     in_to_liq_delay + zero_to_spike_delay + liq_to_readout_delay + margin + delay
        // };
        let additional_delay: i32 = 8;

        //let add_delay = in_to_liq_delay + zero_to_spike_delay + liq_to_readout_delay + margin;
        let n_time_steps = input_time_steps + (additional_delay as usize);
        // let mut
        // For every time step in all the time steps
        for t in 0..n_time_steps {
            // For every post synaptic neuron in the liquid
            for n_idx in 0..self.neurons.len() {
                // If the neuron is in time out, then skip and update timeout
                // Using self.neurons[n_idx] instead of curr_neuron because of
                // differing mutability calls with get and update methods.
                if self.neurons[n_idx].get_time_out() > 0 {
                    self.neurons[n_idx].update_time_out();
                    continue;
                }

                let curr_neuron = &self.neurons[n_idx];
                // The total change in curr_neuron's voltage
                let mut delta_v: f32 = -curr_neuron.get_voltage() / (self.tau_m as f32);

                // If the neuron is a reservoir input
                if curr_neuron.get_spec() == &"liq_in".to_string() {
                    // We have it already so that the reservoir input knows
                    // which input layer neuron it's connected to
                    // We want to change the current neuron's voltage.

                    // For the curr_neuron (liq_in), it gets the list of spikes
                    // coming from the input layer
                    let spike_times: &Vec<u32> =
                        self.input_layer[curr_neuron.get_input_connect()].get_spike_times();
                    // Looks through all the spikes of the input and calculates
                    // how the voltage will change
                    for spike_time in spike_times.iter() {
                        let weight: i32 = curr_neuron.get_input_weight();
                        let model_calc: f32;
                        if curr_neuron.get_input_weight() == 8 {
                            // excitatory synapse
                            model_calc = self.liq_response(
                                model,
                                t as i32,
                                *spike_time as i32,
                                delay,
                                first_tau, // first order tau
                                4, // values from Zhang et al paper for second order tau values
                                8,
                            );
                        } else {
                            // inhibitory synapse
                            model_calc = self.liq_response(
                                model,
                                t as i32,
                                *spike_time as i32,
                                delay,
                                first_tau,
                                4,
                                2,
                            );
                        }
                        delta_v += (weight as f32) * model_calc;
                    }
                }

                // The indices of the pre-syn. connections self.neurons[n_idx]
                // has with the rest of the LSM
                let pre_syn_connects: &Vec<usize> = curr_neuron.get_pre_syn_connects();
                // Loop through all the pre synaptic neuron indices
                for connect_idx_idx in 0..pre_syn_connects.len() {
                    // For each pre synaptic neuron, loop through all its spikes
                    // The pre synaptic neuron that we are currently looking at
                    // pre_syn_connects[connect_idx_idx] is the index of one of
                    // the pre synaptic connections of curr_neuron
                    let pre_syn_neuron: &Neuron = &self.neurons[pre_syn_connects[connect_idx_idx]];

                    // Tau values for second order are based on whether the PRE
                    // Synaptic neuron was Excitatory or Inhibitory
                    // E -> I or E -> E, [4, 8]
                    // I -> I or I -> E, [4, 2]
                    let tau_s1 = pre_syn_neuron.get_second_tau()[0];
                    let tau_s2 = pre_syn_neuron.get_second_tau()[1];
                    let spike_times: &Vec<u32> = pre_syn_neuron.get_spike_times();
                    // Looks through all the spikes of the pre-syn neuron and calculates
                    // how the voltage will change
                    for spike_time in spike_times.iter() {
                        // The weight is depended on the neurotransmitter that each pre and
                        // post synaptic neurons puts out
                        let weight: i32; // Weird warnings. Says it doesn't need to be mut.
                                         /* EE */
                        if pre_syn_neuron.get_nt() == &"exc".to_string()
                            && curr_neuron.get_nt() == &"exc".to_string()
                        {
                            weight = self.liq_weights[0];
                        }
                        /* EI */
                        else if pre_syn_neuron.get_nt() == &"exc".to_string()
                            && curr_neuron.get_nt() == &"inh".to_string()
                        {
                            weight = self.liq_weights[1];
                        }
                        /* IE */
                        else if pre_syn_neuron.get_nt() == &"inh".to_string()
                            && curr_neuron.get_nt() == &"exc".to_string()
                        {
                            weight = self.liq_weights[2];
                        }
                        /* II */
                        else {
                            weight = self.liq_weights[3];
                        }

                        let model_calc = self.liq_response(
                            model,
                            t as i32,
                            *spike_time as i32,
                            delay,
                            first_tau,
                            tau_s1,
                            tau_s2,
                        );
                        delta_v += (weight as f32) * model_calc;
                    }
                }
                self.neurons[n_idx].update_voltage(delta_v);
                self.neurons[n_idx].update_spike_times(t);

                if delta_v != 0. {
                    if self.neurons[n_idx].get_voltage() == -5. {
                        f1.write_all(
                            format!("{}, input w: {}, dv: {} ========================================> (SPIKE!)\n",
                            self.neurons[n_idx],
                            self.neurons[n_idx].get_input_weight(),
                            delta_v).as_bytes()).expect("Unable to write");
                    } else {
                        f1.write_all(
                            format!(
                                "{}, input w: {}, dv: {}\n",
                                self.neurons[n_idx],
                                self.neurons[n_idx].get_input_weight(),
                                delta_v
                            )
                            .as_bytes(),
                        )
                        .expect("Unable to write");
                    }
                }
            }
            f1.write_all(format!("---------------------------------------------------------------------------------------------------------------------------------------------------------{}\n", t as i32 + delay).as_bytes()).expect("Unable to draw");
            self.readout_read(model, t, delay, first_tau, f2);
            // self.readout_output();

            // guesses are in self.outputs' most recent element
            // Train based on this guess
        }
        self.readout_output();
        if train {
            // Sets desired activity of each readout set based on input labels
            self.set_c_d(label);

            // Trains based on the potentiation graph (Figure 6) in Zhang et
            // al paper
            self.graph_analysis(f2, prev_epoch_accuracy);
        }
        f5.write_all(format!("---------------------------------------------------------------------------------------------------------------------------------------------------------{}\n", epoch).as_bytes()).expect("Unable to write weights");
        for col in 0..self.n_readout_clusters {
            f5.write_all(
                format!("{}: [", CLUSTERS[self.readouts[0][col].get_cluster()],).as_bytes(),
            )
            .expect("Unable to write weights");
            for row in 0..self.n_total {
                f5.write_all(format!("{}, ", self.readout_weights[0][(col, row)]).as_bytes())
                    .expect("Unable to write weights");
            }
            f5.write_all(format!("]\n").as_bytes())
                .expect("Unable to write weights");
        }

        // In mins
        let run_time = now.elapsed().as_millis() as f64 / 1000. / 60.;
        // self.print_test_accuracy(epoch, &labels, f3, f4, additional_delay, run_time)
        self.epoch_result(f3, f4, label, epoch, run_time)
    }

    pub fn reset(&mut self) {
        // Loop through Input layer
        for input_idx in 0..self.n_inputs {
            self.input_layer[input_idx].set_spike_times(Vec::new());
            // self.input_layer[input_idx].set_time_out(0);
            // self.input_layer[input_idx].set_voltage(0.);
            // self.input_layer[input_idx].set_calcium(0.);
        }

        // Loop through Liquid
        for n_idx in 0..self.n_total {
            self.neurons[n_idx].set_spike_times(Vec::new());
            self.neurons[n_idx].set_time_out(0);
            self.neurons[n_idx].set_voltage(0.);
            self.neurons[n_idx].set_calcium(0.);
        }

        // Loop through Readouts
        for cluster_idx in 0..self.n_clusters {
            for r_idx in 0..self.readouts[cluster_idx].len() {
                self.readouts[cluster_idx][r_idx].set_spike_times(Vec::new());
                self.readouts[cluster_idx][r_idx].set_time_out(0);
                self.readouts[cluster_idx][r_idx].set_voltage(0.);
                self.readouts[cluster_idx][r_idx].set_calcium(0.);
            }
        }
        self.responsible_liq = Vec::new();
    }

    fn epoch_result(
        &self,
        f3: &mut File,
        f4: &mut File,
        label: &String,
        epoch: usize,
        run_time: f64,
    ) -> String {
        let answer: String;
        let guess = &self.outputs[epoch];
        // assert_eq!(label == &"talk".to_string() || label == &"hide".to_string(), true);
        // assert_eq!(guess == &"talk".to_string() || guess == &"hide".to_string(), true);
        if label == guess {
            answer = "correct".to_string();
            f3.write_all(
                format!("Label: {} -- Guess: {} ======> (CORRECT!)\n", label, guess).as_bytes(),
            )
            .expect("Unable to write");
            f4.write_all(format!("CORRECT------\n").as_bytes())
                .expect("Unable to write");
        } else {
            answer = "incorrect".to_string();
            f3.write_all(format!("Label: {} -- Guess: {}\n", label, guess).as_bytes())
                .expect("Unable to write");
            f4.write_all(format!("-\n").as_bytes())
                .expect("Unable to write");
        }
        println!("Runtime: {:.3}", run_time);

        answer
    }

    fn readout_read(
        &mut self,
        model: &str,
        curr_t: usize,
        delay: i32,
        first_tau: u32,
        f: &mut File,
    ) {
        // For each time step, we calculate from the neuron activity of the
        // reservoir and update the calcium / voltage of the readout neurons.

        // let mut f = File::create("readout.txt").expect("Unable to create a file");

        // Assuming each cluster has the same # of neurons
        let count: usize = self.n_total / self.n_clusters;
        for cluster_idx in 0..self.n_clusters {
            // For each readout neuron
            for r_idx in 0..self.readouts[cluster_idx].len() {
                // Calcium decay
                let mut delta_c: f32 =
                    -self.readouts[cluster_idx][r_idx].get_calcium() / (self.tau_c as f32);
                self.readouts[cluster_idx][r_idx].update_calcium(delta_c);

                // Time out stuff
                if self.readouts[cluster_idx][r_idx].get_time_out() > 0 {
                    self.readouts[cluster_idx][r_idx].update_time_out();
                    continue;
                }

                let curr_readout: &Neuron = &self.readouts[cluster_idx][r_idx];
                let mut delta_v: f32 = -curr_readout.get_voltage() / (self.tau_m as f32);
                delta_c = 0.;

                // For each neuron in the cluster; these are pre-synaptic neurons
                // to each readout neuron
                for n_idx in (cluster_idx * count)..((cluster_idx + 1) * count) {
                    // n_idx is each neuron in the cluster. And each liquid
                    // neuron in a cluster is a PRE-Synaptic neuron to the readout
                    let pre_syn_neuron: &Neuron = &self.neurons[n_idx];
                    let tau_s1 = pre_syn_neuron.get_second_tau()[0];
                    let tau_s2 = pre_syn_neuron.get_second_tau()[1];

                    let spike_times: &Vec<u32> = pre_syn_neuron.get_spike_times();
                    // if spike_times.len() != 0 {
                    //     println!("Spike times: {:?}, Curr_time: {}", spike_times, curr_t);
                    // }
                    // if spike_times.len() != 0 {
                    //     f.write_all(
                    //         format!(
                    //             "Spike times of Neuron {}: {:?}\n",
                    //             self.neurons[n_idx].get_id(),
                    //             spike_times
                    //         )
                    //         .as_bytes(),
                    //     )
                    //     .expect("Unable to write");
                    // }

                    let mut change: f32 = 0.;
                    for spike_time in spike_times.iter() {
                        // Update voltage and calcium (a la equation 14, 21 in Zhang et al)

                        // println!("{}",n_idx)
                        let weight: f32 = self.readout_weights[cluster_idx][(r_idx, n_idx % count)];
                        let model_calc: f32 = self.liq_response(
                            model,
                            curr_t as i32,
                            spike_time.clone() as i32,
                            delay,
                            first_tau,
                            tau_s1,
                            tau_s2,
                        );
                        // let teacher_signal: f32 = injected_current(); MAYBE LATER
                        // delta_v += weight * model_calc + teacher_signal;
                        // if model_calc != 0. {
                        //     f.write_all(
                        //         format!(
                        //             "Neuron {}, Spike time {}: Delta voltage increment:[ w: {}, m_c: {}, d_v: {} ]\n",
                        //             self.neurons[n_idx].get_id(),
                        //             spike_time,
                        //             weight,
                        //             model_calc,
                        //             weight * model_calc
                        //         )
                        //         .as_bytes(),
                        //     )
                        //     .expect("Unable to write");
                        // }
                        change += weight * model_calc;
                        delta_v += change;
                    }
                    self.responsible_liq[r_idx][n_idx] += change;
                }
                self.readouts[cluster_idx][r_idx].update_voltage(delta_v);
                self.readouts[cluster_idx][r_idx].update_spike_times(curr_t);

                // Update of Calcium
                let r_spike_times = self.readouts[cluster_idx][r_idx].get_spike_times();
                if !r_spike_times.is_empty() {
                    let last_spike_time = r_spike_times[r_spike_times.len() - 1];
                    // If the readout spiked in the current time step, add one to
                    // calcium
                    delta_c += self.delta_calcium(curr_t as i32 - last_spike_time.clone() as i32);
                }
                self.readouts[cluster_idx][r_idx].update_calcium(delta_c);

                // Writing to file!
                if delta_v != 0. || delta_c != 0. {
                    if self.readouts[cluster_idx][r_idx].get_voltage() == -5. {
                        f.write_all(
                            format!(
                                "{}, dv: {}, dc:{} ========================================> (SPIKE!)\n",
                                self.readouts[cluster_idx][r_idx], delta_v, delta_c,
                            )
                            .as_bytes(),
                        )
                        .expect("Unable to write");
                    } else {
                        f.write_all(
                            format!(
                                "{}, dv: {}, dc: {}\n",
                                self.readouts[cluster_idx][r_idx], delta_v, delta_c
                            )
                            .as_bytes(),
                        )
                        .expect("Unable to write");
                    }
                }
            }
        }
        f.write_all(format!("---------------------------------------------------------------------------------------------------------------------------------------------------------{}\n", curr_t as i32 + delay).as_bytes()).expect("Unable to draw");
    }

    fn readout_output(&mut self) {
        // MUST CHANGE TO WORK WITH NOTHING
        // Get the readout's guess once per epoch, at the end
        let mut all_c_r: Vec<Vec<f32>> = Vec::new();
        for cluster_idx in 0..self.n_clusters {
            let mut curr_readout_cluster: Vec<f32> = Vec::new();
            for r_idx in 0..self.readouts[cluster_idx].len() {
                curr_readout_cluster.push(self.readouts[cluster_idx][r_idx].get_calcium());
            }
            all_c_r.push(curr_readout_cluster);
        }
        self.readouts_c_r.push(all_c_r); // Holds all the real calciums per epoch, may be large

        // Find the cluster whose calcium is biggest
        let mut output_c_total: Vec<f32> = vec![0.; self.n_readout_clusters];
        // Find total amount of calcium per cluster
        for cluster_idx in 0..self.n_clusters {
            for r_idx in 0..self.readouts[cluster_idx].len() {
                let curr_readout: &Neuron = &self.readouts[cluster_idx][r_idx];
                output_c_total[curr_readout.get_cluster()] += curr_readout.get_calcium();
            }
        }

        let mut max_val_idx: Vec<usize> = Vec::new();
        let mut max_c_val: f32 = 0.; // Most calcium in any cluster
        let mut max_c_idx: usize = 0; // Index of cluster with most calcium
        for (key, val) in output_c_total.iter().enumerate() {
            if val == &max_c_val {
                max_val_idx.push(key.clone());
            }
            if val > &max_c_val {
                max_c_val = val.clone();
                max_c_idx = key.clone();
                max_val_idx = vec![key.clone()];
            }
        }

        let mut r_volts = vec![0.; self.n_readout_clusters];
        if max_val_idx.len() > 1 {
            for r_idx in 0..self.readouts[0].len() {
                for max_idx in max_val_idx.iter() {
                    if &self.readouts[0][r_idx].get_cluster() == max_idx {
                        r_volts[max_idx.clone()] += self.readouts[0][r_idx].get_voltage();
                    }
                }
            }
            let mut max_v_val: f32 = 0.; // Most calcium in any cluster
            let mut max_v_idx: usize = 0; // Index of cluster with most calcium
            for (key, val) in r_volts.iter().enumerate() {
                if val > &max_v_val {
                    max_v_val = val.clone();
                    max_v_idx = key.clone();
                }
            }
            if output_c_total[max_v_idx]
                >= self.c_th * (self.readouts[0].len() / self.n_readout_clusters) as f32
            {
                self.outputs.push(CLUSTERS[max_v_idx].to_string());
            } else {
                self.outputs.push("nothing".to_string());
            }
        } else {
            // Outputs the cluster guess of the cluster with the highest calcium conc.
            if output_c_total[max_c_idx]
                >= self.c_th * (self.readouts[0].len() / self.n_readout_clusters) as f32
            {
                self.outputs.push(CLUSTERS[max_c_idx].to_string());
            } else {
                self.outputs.push("nothing".to_string());
            }
        }
    }

    fn update_weights(
        &mut self,
        f2: &mut File,
        cluster_idx: usize,
        r_idx: usize,
        delta_weight: f32,
        instruction: &str,
        contribution: &str,
    ) {
        assert_eq!(
            instruction == "potentiate" || instruction == "depress",
            true
        );
        assert_eq!(contribution == "pos" || contribution == "neg", true);

        // loop through its pre-syn neurons and increase the weights for them
        // let seed = [13; 32];
        // let mut rng = StdRng::from_seed(seed);

        // let mut rng = rand::thread_rng();
        let n_count = self.n_total / self.n_clusters; // Number of neurons in 1 cluster
                                                      // readout_weights is a list of matrices representing the weights
                                                      // between neurons in the liquid and the readout neurons
        for n_idx in 0..n_count {
            if instruction == "potentiate" {
                if contribution == "pos" {
                    if self.responsible_liq[r_idx][n_idx] > 0. {
                        let sum = self.readout_weights[cluster_idx][(r_idx, n_idx)] + delta_weight;
                        if sum < self.r_weight_max {
                            f2.write_all(
                                format!(
                                    "----> Weight of Neuron {} ↑ from {} to {}",
                                    self.neurons[n_idx + (cluster_idx * n_count)].get_id(),
                                    self.readout_weights[cluster_idx][(r_idx, n_idx)],
                                    sum
                                )
                                .as_bytes(),
                            )
                            .expect("Unable to write");
                            self.readout_weights[cluster_idx][(r_idx, n_idx)] = sum;
                        }
                    }
                } else if contribution == "neg" {
                    if self.responsible_liq[r_idx][n_idx] < 0. {
                        let sum = self.readout_weights[cluster_idx][(r_idx, n_idx)] + delta_weight;
                        if sum < self.r_weight_max {
                            f2.write_all(
                                format!(
                                    "----> Weight of Neuron {} ↑ from {} to {}",
                                    self.neurons[n_idx + (cluster_idx * n_count)].get_id(),
                                    self.readout_weights[cluster_idx][(r_idx, n_idx)],
                                    sum
                                )
                                .as_bytes(),
                            )
                            .expect("Unable to write");
                            self.readout_weights[cluster_idx][(r_idx, n_idx)] = sum;
                        }
                    }
                }
            } else if instruction == "depress" {
                if contribution == "pos" {
                    if self.responsible_liq[r_idx][n_idx] > 0. {
                        let sum = self.readout_weights[cluster_idx][(r_idx, n_idx)] - delta_weight;
                        if sum > self.r_weight_min {
                            f2.write_all(
                                format!(
                                    "----> Weight of Neuron {} ↓ from {} to {}",
                                    self.neurons[n_idx + (cluster_idx * n_count)].get_id(),
                                    self.readout_weights[cluster_idx][(r_idx, n_idx)],
                                    sum
                                )
                                .as_bytes(),
                            )
                            .expect("Unable to write");
                            self.readout_weights[cluster_idx][(r_idx, n_idx)] = sum;
                        }
                    }
                } else if contribution == "neg" {
                    if self.responsible_liq[r_idx][n_idx] < 0. {
                        let sum = self.readout_weights[cluster_idx][(r_idx, n_idx)] - delta_weight;
                        if sum > self.r_weight_min {
                            f2.write_all(
                                format!(
                                    "----> Weight of Neuron {} ↓ from {} to {}",
                                    self.neurons[n_idx + (cluster_idx * n_count)].get_id(),
                                    self.readout_weights[cluster_idx][(r_idx, n_idx)],
                                    sum
                                )
                                .as_bytes(),
                            )
                            .expect("Unable to write");
                            self.readout_weights[cluster_idx][(r_idx, n_idx)] = sum;
                        }
                    }
                }
            }
        }
    }

    fn set_c_d(&mut self, curr_label: &String) {
        // This function happens in one epoch.
        // Check the current label in labels and set the c_d's in that cluster to a
        // high number and the other clusters to a low c_d.
        let high_c_d: usize = 10;
        let low_c_d: usize = 0;

        // // Current label is the label from the input at this time step.
        let mut all_calciums: Vec<Vec<f32>> = Vec::new();
        // For each cluster, push the readout neurons' desired calciums into a vector
        // and the column of a matrix
        for cluster_idx in 0..self.n_clusters {
            let mut curr_readout_cluster: Vec<f32> = Vec::new();
            for r_idx in 0..self.readouts[cluster_idx].len() {
                let current_cluster = CLUSTERS[self.readouts[cluster_idx][r_idx].get_cluster()];
                if curr_label == current_cluster {
                    // Set the neurons to have a high c_d
                    self.readouts[cluster_idx][r_idx].set_calcium_desired(high_c_d as f32);
                } else {
                    // Set the neurons to have a low c_d
                    self.readouts[cluster_idx][r_idx].set_calcium_desired(low_c_d as f32);
                }
                curr_readout_cluster.push(self.readouts[cluster_idx][r_idx].get_calcium_desired());
            }
            all_calciums.push(curr_readout_cluster);
        }
        self.readouts_c_d.push(all_calciums); // readouts_c_d is filled once per epoch
    }

    fn graph_analysis(&mut self, f2: &mut File, _prev_epoch_accuracy: f32) {
        // For one time step, if the calcium is within a certain range of
        // the threshold, then we potentiate weights.
        // Otherwise, we depress weights.
        // d - depress, p - potentiate, c_m - margin above / below c_th
        // c_r
        //              |  d    |    N
        // c_th + c_m   |  d    |    p
        // c_th         |  d    |    p
        // c_th - c_m   |  d    |    p
        //              |  N    |    p
        //              |___________________
        //                     c_th      c_d
        // For each readout neuron's calcium values, if it's within a range
        // of the threshold, we potentiate.
        // let acc_coeff = 1.5 * f32::exp((-1. / 3.0) * prev_epoch_accuracy);
        let acc_coeff = 1.;
        // let mut rng = rand::thread_rng();
        let scale = 1.;
        let high_d_w = 0.1 * scale * acc_coeff; // (0.05-0.15)
        let mid_d_w = 0.05 * scale * acc_coeff; // (0.025-0.075)
        let low_d_w = 0.025 * scale * acc_coeff; // (0.01 - 0.035)
                                                 // let prob: f32 = (rng.gen::<f32>() / 2.) + 0.25; // Probability of Potentiation or depression
        for cluster_idx in 0..self.n_clusters {
            for r_idx in 0..self.readouts[cluster_idx].len() {
                let c_r: f32 = self.readouts[cluster_idx][r_idx].get_calcium();
                let c_d: f32 = self.readouts[cluster_idx][r_idx].get_calcium_desired();
                // Right half of x axis
                if c_d > self.c_th {
                    if c_r < self.c_th - self.c_margin {
                        self.update_weights(f2, cluster_idx, r_idx, high_d_w, "potentiate", "pos");
                        self.update_weights(f2, cluster_idx, r_idx, high_d_w, "potentiate", "neg");
                    } else if c_r < self.c_th {
                        self.update_weights(f2, cluster_idx, r_idx, mid_d_w, "potentiate", "pos");
                        self.update_weights(f2, cluster_idx, r_idx, mid_d_w, "potentiate", "neg");
                    } else if c_r < self.c_th + self.c_margin {
                        self.update_weights(f2, cluster_idx, r_idx, low_d_w, "potentiate", "pos");
                        self.update_weights(f2, cluster_idx, r_idx, low_d_w, "potentiate", "neg");
                    }
                // if c_r > c_th + c_margin, we don't learn
                // Left
                } else {
                    if c_r > self.c_th + self.c_margin {
                        self.update_weights(f2, cluster_idx, r_idx, high_d_w, "depress", "pos");
                        self.update_weights(f2, cluster_idx, r_idx, high_d_w, "depress", "neg");
                    } else if c_r > self.c_th {
                        self.update_weights(f2, cluster_idx, r_idx, mid_d_w, "depress", "pos");
                        self.update_weights(f2, cluster_idx, r_idx, mid_d_w, "depress", "neg");
                    } else if c_r > self.c_th - self.c_margin {
                        self.update_weights(f2, cluster_idx, r_idx, low_d_w, "depress", "pos");
                        self.update_weights(f2, cluster_idx, r_idx, low_d_w, "depress", "neg");
                    }
                    /*
                    // if c_r == 0 and c_d == 0 it ends here without going
                    // through either if statement
                    // Perhaps because on the graph it says not to learn, so we ignore it.
                     */
                }
            }
        }
    }
}

// END OF FILE
