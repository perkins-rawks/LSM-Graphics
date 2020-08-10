/*
    The Liquid State Machine (LSM) struct contains functionality for the input layer, liquid reservoir, readout neurons, 
    and training. Each neuron in the liquid is a Neuron object, and are connected to each other by probability depending
    on proximity to other neurons. Input that is read from another file in the input.rs file in src folder is processed
    here and effects the liquid.
    
    Once every epoch (controlled in main file), a guess for what signal is recevied is the LSM's response. 

    This file also writes on and fills other files including: 

    o Voltage of the liquid at any given time
    o Voltage and calcium of the readout at any time
    o Weights between liquid and readout
    o Epoch results
    o Epoch guesses
*/

// Imports

// kiss3d is a package for graphics
use kiss3d::scene::SceneNode; // For neuron spheres
use kiss3d::window::Window; // A window to draw on

use nalgebra::base::DMatrix; // Matrix of variable size
use nalgebra::geometry::Translation3; // A tuple of size 3 for SceneNode centers
use nalgebra::{Point3, Vector3}; // Tuples of size 3 for other random uses

use rand::prelude::*; 
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};

mod neuron;
use neuron::Neuron;
use neuron::CLUSTERS; // static variable containing readout function names

mod utils;

use std::io::Write;
use std::time::Instant;
use std::{fs, fs::File};

pub struct LSM {
    n_total: usize,                     // The number of total neurons
    n_inputs: usize,                    // The number of input neurons
    n_input_copies: usize,              // The number of copies of the input is sent into the LSM
    n_clusters: usize,                  // The number of cluster neurons
    n_readout_clusters: usize,          // The number of readout clusters in a single cluster
    neurons: Vec<Neuron>,               // List of all neurons which has ownership of each element
    clusters: Vec<Vec<Neuron>>,         // A cluster is a set of neurons
    readouts: Vec<Vec<Neuron>>,         // List of all clusters of readout neurons
    readout_weights: Vec<DMatrix<f32>>, // List of all the plastic weights between readouts and reservoir
    input_layer: Vec<Neuron>,           // Set of neurons that read input outside the reservoir
    connections: DMatrix<u32>, // List of all connections in liquid (1 if connected, 0 if not)
    liq_weights: [i32; 4],     // Fixed synaptic weights in the liquid
    readouts_c_r: Vec<Vec<Vec<f32>>>, // Each readout neuron's calcium values from each cluster for each epoch
    readouts_c_d: Vec<Vec<Vec<f32>>>, // Each readout neuron's desired calcium values from each cluster for each epoch
    responsible_liq: Vec<Vec<f32>>, // The indices of neurons that caused spikes for each readout in one time step
    outputs: Vec<String>,           // The output of the LSM at every epoch
    e_ratio: f32,      // The ratio of excitatory to inhibitory neurons (between 0 and 1)
    tau_m: u32,        // Time constant for voltage decay
    tau_c: u32,        // Time constant for calcium decay
    r_weight_max: f32, // The maximum weight between readout and liquid
    r_weight_min: f32, // The minimum weight between readout and liquid
    c_th: f32,         // Calcium threshold for a readout neuron
    c_margin: f32,     // Margin around the calcium threshold
}

impl LSM {
    pub fn new(
        n_inputs: usize,
        n_input_copies: usize,
        n_clusters: usize,
        n_readout_clusters: usize,
        liq_weights: [i32; 4],
        e_ratio: f32,
    ) -> LSM {
        assert_eq!(true, e_ratio <= 1.);
        Self {
            n_total: n_inputs,
            n_inputs,
            n_input_copies,
            n_clusters,
            n_readout_clusters,
            neurons: Vec::new(),
            clusters: Vec::new(),
            readouts: Vec::new(),
            readout_weights: Vec::new(),
            input_layer: Vec::new(),
            connections: DMatrix::<u32>::zeros(0, 0),
            liq_weights, // [Exc to Exc, Exc to Inh, Inh to Exc, Inh to Inh]
            readouts_c_r: Vec::new(),
            readouts_c_d: Vec::new(),
            responsible_liq: Vec::new(),
            outputs: Vec::new(),
            e_ratio,
            tau_m: 32,
            tau_c: 64,
            r_weight_max: 8.,
            r_weight_min: -8.,
            c_th: 5.,
            c_margin: 3.,
        }
    }

    // Cluster Methods \\
    pub fn make_cluster(
        &mut self,
        window: &mut Window,        // Our screen
        n: usize,                   // The number of neurons we want in a cluster
        radius: f32,                // Radius of each sphere neuron
        var: f32, // The variance in the distribution (Larger number means bell curve is wider)
        cluster: &str, // The task for which the cluster is created (talk, eat, etc.)
        loc: &Point3<f32>, // Center of the cluster
        (r, g, b): (f32, f32, f32), // Color of the cluster
        n_readouts: usize, // The number of readouts per cluster
    ) {
        // Creates a cluster of n (r,g,b)-colored, radius sized neurons around a
        // center at loc, distributed around the center with variance var.
        // Also creates the readout set associated with each cluster and assigns
        // the neurons in it to be input randomly. At the end, the cluster is
        // stored in a cluster list, and its neurons are stored in the
        // neurons list instance variable.
        assert_ne!(n, 0); // so that we don't create an empty cluster
        let mut neurons: Vec<Neuron> = Vec::new(); // we will set this to be our self.neurons
        let seed = [1_u8; 32];
        let mut rng = StdRng::from_seed(seed);

        for sphere in 0..n {
            // Normal takes mean and then variance
            let normal_x = Normal::new(loc[0], var).unwrap();
            let normal_y = Normal::new(loc[1], var).unwrap();
            let normal_z = Normal::new(loc[2], var).unwrap();

            // Generates a random point for a new neuron.
            // sample takes a random number generator
            let t = Translation3::new(
                normal_x.sample(&mut rng),
                normal_y.sample(&mut rng),
                normal_z.sample(&mut rng),
            );

            let neuron: Neuron = Neuron::new(window.add_sphere(radius), "liq", cluster);
            neurons.push(neuron);
            // get_obj returns a &SceneNode which is a sphere that we draw
            neurons[sphere].get_obj().set_color(r, g, b);
            // append_translation moves its center to the point t
            neurons[sphere].get_obj().append_translation(&t);
            // set_loc is a neuron method that updates its location
            neurons[sphere].set_loc(&t);
        }

        self.assign_input(&mut neurons); // randomly assigns self.n_inputs number of liquid inputs
        let readouts: Vec<Neuron> = self.make_readouts(window, n_readouts, &loc, radius);
        self.add_readouts(readouts); // list assignment for the readouts
        self.add_cluster(neurons); // adds the set of neurons we initialized earlier, treats it as one cluster
        self.unpack(); // Unpacks the cluster list into self.neurons list
    }

    fn unpack(&mut self) {
        // Puts all the neurons in the cluster into self's vector of neurons.
        // Also updates node identifications to be their index in the neurons list.

        // Liquid IDs

        // For each neuron in the most recent neuron list added in self.clusters
        for (id, neuron) in self.clusters[self.clusters.len() - 1].iter().enumerate() {
            self.neurons.push(neuron.clone()); // .clone() to avoid moving problems

            // Setting ID based on previous ID
            if id == 0 && self.clusters.len() == 1 {
                // The very first liquid neuron's id
                self.neurons[id].set_id(0);
            } else {
                let last_idx = self.neurons.len() - 1;
                let prev_id: usize = self.neurons[last_idx - 1].get_id();
                self.neurons[last_idx].set_id(prev_id + 1);
            }
        }
        self.n_total = self.neurons.len(); // updates total neuron count

        // Readout IDs

        // Assuming each cluster has the same # of neurons
        // We CANNOT use self.n_total because the LSM is not filled yet. This is
        // only called in the process of setting up the LSM.
        let count: usize = self.clusters[0].len(); // number of neurons in the first cluster
        let r_len: usize = self.readouts.len(); // the number of readout neurons

        // For each readout in the most recent readout set made
        for r_idx in 0..self.readouts[r_len - 1].len() {
            if r_idx == 0 && r_len == 1 {
                // The very first readout neuron's id
                self.readouts[r_len - 1][r_idx].set_id(count * self.n_clusters);
            } else if r_idx == 0 && r_len > 1 {
                // If NOT the first readout cluster and the index is 0
                let last_readout: &Neuron = &self.readouts[r_len - 2] // prev cluster
                    [self.readouts[r_len - 2].len() - 1]; // last item in that prev cluster
                let prev_id: usize = last_readout.get_id();
                self.readouts[r_len - 1][r_idx].set_id(prev_id + 1);
            } else {
                // Generally
                let last_readout: &Neuron = &self.readouts[r_len - 1][r_idx - 1];
                let prev_id: usize = last_readout.get_id();
                self.readouts[r_len - 1][r_idx].set_id(prev_id + 1);
            }
        }
    }

    fn assign_input(&mut self, neurons: &mut Vec<Neuron>) {
        // Assigns self.n_inputs random liquid neurons to take input. \\
        assert_eq!(true, self.n_inputs > 0);

        // We want to choose n_inputs neurons from all the liquid neurons created to be
        // inputs.
        let seed: [u8; 32] = [2; 32];
        let mut rng = StdRng::from_seed(seed);

        // This list has the indices of every unique choice for an input neuron
        // in self.neurons.
        let mut liq_idx: Vec<usize> = (0..neurons.len()).collect();

        // Input copies is functionality we have to send multiple copies of the input
        // to each cluster. This only is set to function if there is more than 1 cluster.
        for _ in 0..self.n_input_copies {
            for _ in 0..self.n_inputs {
                // choose chooses a random element of a vector
                let in_idx: usize = *liq_idx.choose(&mut rng).unwrap(); // * is to dereference
                                                                        // Set to input, change the color, and remove that index so that we
                                                                        // get unique input neurons
                neurons[in_idx].set_spec("liq_in");
                neurons[in_idx].get_obj().set_color(0.9453, 0.0938, 0.6641); // every liquid input is this color
                                                                             // the index of in_idx in liq_idx
                let idx = liq_idx.iter().position(|&r| r == in_idx).unwrap();
                liq_idx.remove(idx);
            }
        }
    }

    fn assign_nt(&mut self) {
        // Assign_nt (neuro transmitter) calculates the number of excitatory
        // and inhibitory neurons in the liquid and chooses randomly from the
        // liquid to assign those roles. This is also where the second order
        // model's hyper parameters (tau_1, tau_2) for the liq_response
        // function are set.

        // For a small LSM, make it excitatory
        if self.n_total == 1 {
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

        // This list has the indices of every unique liquid neuron
        // in self.neurons list.
        let mut liq_idx: Vec<usize> = (0..self.n_total).collect();
        // Setting the excitatory neurons
        for _ in 0..n_exc {
            // Choose picks a random element of liq_idx
            let exc_idx: usize = *liq_idx.choose(&mut rng).unwrap(); // * dereferences
                                                                     // Set the role, second order model's hyper parameters, and remove
                                                                     // the index as a choice for liq_idx
            self.neurons[exc_idx].set_nt("exc");
            self.neurons[exc_idx].set_second_tau(4, 8);
            // Finds the idx of exc_idx in liq_idx
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

    fn add_cluster(&mut self, new_cluster: Vec<Neuron>) {
        // Adds a set of neurons (a cluster) to the clusters list
        self.clusters.push(new_cluster);
    }

    pub fn get_neurons(&mut self) -> &mut Vec<Neuron> {
        // Accessor method for all neurons in the liquid.
        &mut self.neurons
    }

    fn add_readouts(&mut self, readouts: Vec<Neuron>) {
        // Pushes a cluster's readout neurons in the readouts list
        self.readouts.push(readouts);
    }

    fn make_readouts(
        &mut self,
        window: &mut Window, // Window we will draw on
        n_readouts: usize,   // Total number of readout neurons to make
        loc: &Point3<f32>,   // The center of the cluster
        radius: f32,         // The radius of each readout neuron's associated sphere
    ) -> Vec<Neuron> {
        // Creates the readouts for all clusters, and returns that set of
        // neurons.
        // self.n_readout_clusters is the number of signals an LSM has
        // There must be at least one readout neuron per readout_cluster.
        // For example, if I want the LSM to recognize "hide" and "run,"
        // I may say n_readouts is 2, so one readout is dedicated to each
        // function.
        assert_eq!(n_readouts % self.n_readout_clusters, 0);

        // One readout set to return
        let mut readouts: Vec<Neuron> = Vec::new();
        for idx in 0..n_readouts {
            // The location of the readout set

            // To put the readouts behind the cluster
            let x: f32 = loc.x * 2.5;
            let y: f32 = loc.y;
            let z: f32 = loc.z * 2.5;
            let t = Translation3::new(x, y + n_readouts as f32 / 2. - idx as f32, z);

            // CLUSTERS is a list of all readout functions there are.
            // The idx is so that the indexes of this are
            // 0..(self.n_readout_clusters - 1) repeating until n_readouts
            // neurons are created
            let function: &str = CLUSTERS[idx % self.n_readout_clusters];
            let neuron: Neuron = Neuron::new(window.add_sphere(radius), "readout", function);
            // Same as in make_cluster method above
            readouts.push(neuron);
            readouts[idx].get_obj().append_translation(&t);
            readouts[idx].set_loc(&t);
        }
        readouts
    }

    pub fn make_connects(
        &mut self,
        window: &mut Window, // The window we draw our graphics on
        c: [f32; 5], // This and lambda are hyper parameters for connect_chance function in utils.rs.
        lambda: f32,
    ) -> (
        Vec<(Point3<f32>, Point3<f32>, Point3<f32>)>,
        Vec<f32>,
        Vec<(Point3<f32>, Point3<f32>, Point3<f32>)>,
    ) {
        // Returns a list of connections between liquid neurons, the distances
        // between those connections, and the connections between readout neurons.

        // Sets the weights between the input layer and the liquid input neurons
        self.set_input_weights();
        self.assign_nt(); // Sets neurons to excitatory or inhibitory
        let n_len = self.n_total; // Because of moving value problems, we can't use self too often

        // This matrix represents connections between two liquid neurons.
        // At connects[(0, 1)], it will display a 1 if neuron 0 in self.neurons
        // is connected to neuron 1 in self.neurons.
        let mut connects = DMatrix::<u32>::zeros(n_len, n_len);

        // This function makes the connections between neurons based on a probability
        // function previously defined. We don't render the lines graphically
        // until later.
        // The tuples in connect_lines represent
        // (Neuron 1 Location, Neuron 2 Location, Color of their connection)
        let mut connect_lines: Vec<(Point3<f32>, Point3<f32>, Point3<f32>)> = Vec::new();
        // Dist_list is a list of distances between each neuron to their
        // connected neurons.
        let mut dist_list: Vec<f32> = Vec::new();

        let seed = [12; 32];
        let mut rng = StdRng::from_seed(seed);

        for idx1 in 0..n_len {
            let coord1: &Vector3<f32> = self.neurons[idx1].get_loc();
            // x, y, and z are components of a Vector3
            let (x1, y1, z1) = (coord1.x, coord1.y, coord1.z);
            let idx1_nt: String = self.neurons[idx1].get_nt().clone(); // either "exc" or "inh"

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
                // C is a list that holds the c parameters to the connect_chance
                // function in utils.rs.
                // It is in the order: [E to E, E to I, I to E, I to I, Loop]
                // where Loop means connected to itself (a very rare chance)
                let idx2_nt: String = self.neurons[idx2].get_nt().clone();
                // Connection starts with E
                if idx1_nt == "exc".to_string() {
                    // EI
                    if idx2_nt == "inh".to_string() {
                        c_idx = 1;
                    }
                // if it ends with E, then the index needs to be 0, but it
                // already is 0, so don't change it

                // Otherwise, the connection starts with I
                } else {
                    // IE
                    if idx2_nt == "exc".to_string() {
                        c_idx = 2;
                    // II
                    } else if idx2_nt == "inh".to_string() {
                        c_idx = 3;
                    }
                }

                // If connects[(idx1, idx2)] == connects[(idx2, idx1)] == 1,
                // then we call it a loop
                if connects[(idx2, idx1)] == 1 {
                    c_idx = 4; // a loop
                } else if self.neurons[idx2].get_spec() == &"liq_in".to_string()
                    && self.neurons[idx1].get_spec() != &"liq_in".to_string()
                {
                    // The probability of a non-liquid input connecting to a
                    // liquid input is lower.
                    // This prevents liquid input from spiking when it
                    // shouldn't, which skews the input.
                    c_idx = 4;
                }

                let coord2: &Vector3<f32> = self.neurons[idx2].get_loc();
                let (x2, y2, z2) = (coord2.x, coord2.y, coord2.z);
                let d = utils::dist(&(x1, y1, z1), &(x2, y2, z2)); // distance between the two points

                // Choosing the correct weight based on the combination of
                // pre-synaptic to postsynaptic neuron's type

                // Makes connections based on distance and some hyper parameters
                let prob_connect: f32 = utils::connect_chance(d, c[c_idx], lambda);
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
        self.update_pre_syn_connects(&connects); // Updates newest connections
        self.add_connections(connects); // Adds the current connections
        let readout_lines = self.make_readout_connects(); // Edges between readout neurons
        self.make_input_layer(); // Makes the input layer
        self.make_input_connects(); // Connects the input layer to the liquid reservoir
        (connect_lines, dist_list, readout_lines)
    }

    fn update_pre_syn_connects(&mut self, connects: &DMatrix<u32>) {
        // Updates each liquid neuron's pre-synaptic connections

        // Basic algorithm:
        // Go through all the neurons, (column wise in the connects DMatrix)
        // For each neuron connecting to it, add that index to a list.
        // That index is the index of that neuron in the self.neurons list.

        // We know that connects is n_total x n_total
        for col in 0..self.n_total {
            let mut pre_connects_idx: Vec<usize> = Vec::new();
            for row in 0..self.n_total {
                // Reminder that D Matrices are column-major.
                // This order of indexing results in getting pre-synaptic connects.
                // This row connects to this col (self.neurons[col] is our current neuron)
                if connects[(row, col)] == 1 {
                    pre_connects_idx.push(row);
                }
            }
            self.neurons[col].set_pre_syn_connects(pre_connects_idx);
        }
    }

    pub fn load_readout_weights(&mut self) {
        // If the LSM is not training at all, this method is called to load a file with
        // good readout weights for the three main symbols we attached
        // In order, we have: "hide", "run", "eat" weights set in this trained_weights.txt file.
        // NOTE: THIS METHOD WORKS FOR ONLY ONE CLUSTER
        let count = self.n_total / self.n_clusters;
        // The count of all readouts, if we only have one cluster
        let readout_len = self.readouts[0].len();

        // Read a file and split it into a vector by line
        // Each line is one function's readout weights.
        let contents =
            fs::read_to_string("trained_weights.txt").expect("Unable to read input file");
        let contents: Vec<&str> = contents.split("\n").collect();
        let mut weights = DMatrix::<f32>::zeros(readout_len, count); // we will fill this matrix
        for (r_idx, line) in contents.iter().enumerate() {
            let new_line: Vec<&str> = line.trim().split(", ").collect();
            for (n_idx, num) in new_line.iter().enumerate() {
                weights[(r_idx, n_idx)] = num.parse::<f32>().unwrap();
            }
        }
        // Each vector in self.readout_weights is one readout neuron's readout
        // weights at a time.
        self.readout_weights = vec![weights];
    }

    fn make_readout_connects(&mut self) -> Vec<(Point3<f32>, Point3<f32>, Point3<f32>)> {
        // Returns a vector containing tuples of the form
        // (Readout 1 location, Readout 2 location, Edge color)
        // for each readout neuron.

        // Making sure we only call this method once.
        assert_eq!(self.readout_weights.is_empty(), true);
        // The vector we will return
        let mut connect_lines: Vec<(Point3<f32>, Point3<f32>, Point3<f32>)> = Vec::new();
        let seed = [99; 32];
        let mut rng1 = StdRng::from_seed(seed);

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

                // If there is one cluster, neuron_idx is from 0 to count.
                // At two clusters, neuron_idx is from count to 2*count.
                // This index is possible only if each cluster has the same
                // number of neurons.
                for neuron_idx in count * cluster_idx..count * (cluster_idx + 1) {
                    let n_loc = self.neurons[neuron_idx].get_loc();
                    let (x2, y2, z2) = (n_loc.x, n_loc.y, n_loc.z);
                    connect_lines.push((
                        Point3::new(x1, y1, z1),
                        Point3::new(x2, y2, z2),
                        Point3::new(227. / 255., 120. / 255., 105. / 255.), // light pink
                    ));

                    // The 'neuron_idx % count' is to sort by cluster
                    // All readout weights are set to a random float
                    // between -1 and 1.
                    readout_weights[(readout_idx, neuron_idx % count)] =
                        (rng1.gen::<f32>() * 2.) - 1.;
                }
            }
            self.readout_weights.push(readout_weights);
        }
        connect_lines
    }

    pub fn get_readouts(&mut self) -> &mut Vec<Vec<Neuron>> {
        // Accessor method for the readouts of each function.
        &mut self.readouts
    }

    pub fn remove_disconnects(&mut self, window: &mut Window) {
        // Removes neurons that are not connected to any other neurons. \\
        // The indices of neurons which have no connections
        let mut rm_n: Vec<usize> = Vec::new();
        for col in 0..self.n_total {
            // If the current neuron has any connection, this is true
            let mut connected: bool = false;
            for row in 0..self.n_total {
                // We need to remove only if the neuron has no connections into
                // or out from it.
                if self.connections[(col, row)] == 1 || self.connections[(row, col)] == 1 {
                    connected = true;
                    break;
                }
            }
            if !connected {
                window.remove_node(self.neurons[col].get_obj()); // Erases it from window
                rm_n.push(col);
            }
        }

        for idx in 0..rm_n.len() {
            // Removes the neuron from the neurons list.
            // Note: remove is a function for vectors that takes an index
            self.neurons.remove(rm_n[idx] - idx);
        }
    }

    fn add_connections(&mut self, connects: DMatrix<u32>) {
        // Assignment for connections matrix. Just moves the variable.
        self.connections = connects;
    }

    pub fn get_connections(&mut self) -> &mut DMatrix<u32> {
        // Accessor method for the connections matrix.
        &mut self.connections
    }

    fn make_input_layer(&mut self) {
        // Creates n_input neurons that will feed into the reservoir's input
        let mut empty = SceneNode::new_empty();
        empty.set_visible(false);
        for _ in 0..self.n_inputs {
            self.input_layer
                .push(Neuron::new(empty.clone(), "in", "input layer"));
        }
    }

    fn set_input_weights(&mut self) {
        // Sets the weight between input layer neurons and liquid inputs to be either -8 or 8
        let seed = [10; 32];
        let mut rng = StdRng::from_seed(seed);
        let percentage = self.e_ratio.clone();

        let weights = [-8, 8];
        for neuron in self.get_neurons().iter_mut() {
            if neuron.get_spec() == &"liq_in".to_string() {
                if rng.gen::<f32>() < percentage {
                    neuron.set_input_weight(weights[1]);
                } else {
                    neuron.set_input_weight(weights[0]);
                }
            }
        }
    }

    fn make_input_connects(&mut self) {
        // Connects the outside input layer to each liquid input neuron in each cluster

        // Updates a neuron classified as "liq_in"'s input_connect attribute
        // This represents the index of that neuron in self.input_layer vector

        // The number of neurons in a cluster
        let count: usize = self.n_total / self.n_clusters;
        for cluster in 0..self.n_clusters {
            let mut input_idx: usize = 0;
            // Same as for loop in make_readout_connects
            // Only works if each cluster is the same length.
            for neuron_idx in count * cluster..count * (cluster + 1) {
                // Connects an input layer neuron to one reservoir input in each cluster
                if input_idx >= self.n_inputs {
                    // This is for multi-cluster functionality, so that the
                    // input is copied across multiple clusters
                    input_idx = 0;
                }
                if self.neurons[neuron_idx].get_spec() == &"liq_in".to_string() {
                    self.neurons[neuron_idx].set_input_connect(input_idx);
                    input_idx += 1;
                }
            }
        }
    }

    fn set_spike_times(
        &mut self,
        input: &Vec<Vec<u32>>, // The input spike trains read from another file
    ) {
        // This sets the spike times array of the liquid input neurons by
        // reading the activity in the input directly. Activity is defined by
        // a 1 in a spike train.

        // There should be as many liquid input neurons as there are spike
        // trains.
        assert_eq!(self.n_inputs, input.len());
        // For each liq_in neuron's pre-synaptic connects, set the
        // spike_times array to be the index of 1's in the input vector

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

    // This allows the talk method to not be called for one test or another.
    #[allow(dead_code)]
    pub fn talk(
        &self,
        word: &str, // The word that will be converted to a spike train
        n_time_steps: u32,
    ) -> Vec<Vec<u32>> // An output spike train that can be fed into another LSM
    {
        // Talk is a function from a string to a spike train. It takes a small
        // string and generates a unique spike train that can be used as input
        // for another LSM.

        let seed = utils::string_to_seed(word);
        let mut rng = StdRng::from_seed(seed);
        let prob_of_one = 0.667; // The probability for each randomly generated number to be active
        let prob_noise = 0.05; // The probability for random noise on the spike train

        let mut spike_trains: Vec<Vec<u32>> = Vec::new();

        // Creates an n_inputs by n_time steps grid of spike trains.
        // Each row is a spike train for another LSM.
        for _ in 0..self.n_inputs {
            let mut spike_train: Vec<u32> = Vec::new();
            if rng.gen::<f32>() < prob_of_one {
                for _ in 0..n_time_steps {
                    // Will most likely be a one, but sometimes a zero
                    if rng.gen::<f32>() < prob_noise {
                        spike_train.push(0);
                    } else {
                        spike_train.push(1);
                    }
                }
            } else {
                for _ in 0..n_time_steps {
                    // Will most likely be a zero, but sometimes a one
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
        train: bool,              // Whether or not we are training
        epoch: usize,             // The total number of training + testing epochs
        f1: &mut File,            // The outputs of the liquid voltage per time step
        f2: &mut File,            // The readouts per time step
        f3: &mut File,            // The guesses per epoch
        f4: &mut File,            // Epoch guesses (Correct or Incorrect)
        f5: &mut File,            // The readout weights over time
        input: &Vec<Vec<u32>>,    // The input spike trains
        label: &String,           // Label of the signal per epoch. Each epoch has one signal.
        model: &str,              // The model we want to use in voltage calculation.
        delay: i32,               // The delay for how we calculate spike times
        first_tau: u32,           // If this is first order model, this is its hyper parameter.
        prev_epoch_accuracy: f32, // A running average of the past few epochs.
    ) -> String {
        // Updates voltage connections for all neurons in the LSM.

        // For special writing of the initial readout weights to file in the first epoch
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

        // For timing each epoch
        let now = Instant::now();

        // Tracks the activity of the spike trains and puts that information into the input layer.
        self.set_spike_times(input);
        let input_time_steps: usize = input[0].len();
        let additional_delay: i32 = 8; // A delay hyper parameter for how long it takes the LSM to guess accurately.
        let n_time_steps = input_time_steps + (additional_delay as usize);
        // Resets each time this function is called
        self.responsible_liq = vec![vec![0.; self.n_total]; self.readouts[0].len()];

        for t in 0..n_time_steps {
            // For every post synaptic neuron in the liquid
            for n_idx in 0..self.neurons.len() {
                // If the neuron has just fired, then skip and update timeout.
                // Using self.neurons[n_idx] instead of curr_neuron because of
                // differing mutability calls with get and update methods.
                if self.neurons[n_idx].get_time_out() > 0 {
                    self.neurons[n_idx].update_time_out();
                    continue;
                }

                let curr_neuron = &self.neurons[n_idx];
                // The first part of the voltage dynamics equation, this will be updated
                // later on as we go through the presynaptic spikes.
                let mut delta_v: f32 = -curr_neuron.get_voltage() / (self.tau_m as f32);

                // If the postsynaptic neuron is a reservoir input
                if curr_neuron.get_spec() == &"liq_in".to_string() {
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
                            model_calc = utils::liq_response(
                                model,
                                t as i32,
                                *spike_time as i32,
                                delay,
                                first_tau,
                                4,
                                8,
                            );
                        } else {
                            // inhibitory synapse
                            model_calc = utils::liq_response(
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

                // The indices of the pre-synaptic connections the current liquid neuron
                // has with the rest of the liquid
                let pre_syn_connects: &Vec<usize> = curr_neuron.get_pre_syn_connects();
                for connect_idx_idx in 0..pre_syn_connects.len() {
                    // For each pre synaptic neuron, loop through all its spikes
                    // The pre synaptic neuron that we are currently looking at
                    // pre_syn_connects[connect_idx_idx] is the index (in self.neurons) of each
                    // pre synaptic connection for the current neuron.
                    let pre_syn_neuron: &Neuron = &self.neurons[pre_syn_connects[connect_idx_idx]];

                    // Tau values for second order model are based on whether the pre-
                    // synaptic neuron was Excitatory or Inhibitory
                    // E -> I or E -> E, [4, 8]
                    // I -> I or I -> E, [4, 2]
                    let tau_s1 = pre_syn_neuron.get_second_tau()[0];
                    let tau_s2 = pre_syn_neuron.get_second_tau()[1];
                    let spike_times: &Vec<u32> = pre_syn_neuron.get_spike_times();
                    // Looks through all the spikes of the pre-syn neuron and calculates
                    // how the voltage will change
                    for spike_time in spike_times.iter() {
                        // The weight depends on the neurotransmitter that each pre and
                        // post synaptic neurons puts out
                        let weight: i32;
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

                        let model_calc = utils::liq_response(
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

                // Writing to each file
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
            self.readout_read(model, t, delay, first_tau, f2); // Updates the readout neurons' firing
        }
        // The epoch has finished.

        // Guesses what the signal was
        self.readout_output();
        if train {
            // Sets desired activity of each readout set based on input labels
            self.set_c_d(label);

            // Trains based on the potentiation graphs
            self.graph_analysis(f2, prev_epoch_accuracy);
        }

        // Writing readout weights to files once an epoch
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
        // The weird conversion is to give accuracy in the thousandths for time taken
        let run_time = now.elapsed().as_millis() as f64 / 1000. / 60.;
        // Epoch result prints out the LSM's guess for the epoch and writes it to a file.
        self.epoch_result(f3, f4, label, epoch, run_time)
    }

    pub fn reset(&mut self) {
        // Once an epoch, reset all of the LSM and neurons instance variables EXCEPT for
        // readout weights.

        // Loop through Input layer
        for input_idx in 0..self.n_inputs {
            self.input_layer[input_idx].set_spike_times(Vec::new());
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
        label: &String, // The name of the signal in an epoch
        epoch: usize,   // The current epoch
        run_time: f64,  // The run time will be written to file here
    ) -> String {
        let answer: String; // A scoring of the guess, either correct or incorrect
        let guess = &self.outputs[epoch]; // The most active readout cluster in an epoch

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
        model: &str,     // The model for calculation we will use (Static, first order, second order)
        curr_t: usize,   // The current time step at which to update readouts
        delay: i32,      // The delay for voltage dynamics
        first_tau: u32,  // If first order, this is its hyper parameter
        f: &mut File,    // Readout information file
    ) {
        // For each time step, we calculate from the neuron activity of the
        // reservoir and update the calcium / voltage of the readout neurons.

        // Assuming each cluster has the same number of neurons
        let count: usize = self.n_total / self.n_clusters;
        for cluster_idx in 0..self.n_clusters {
            for r_idx in 0..self.readouts[cluster_idx].len() {
                // Calcium decay, the first part of the calcium dynamics equation.
                // If there were no presynaptic spikes, the calcium decays slowly. 
                let mut delta_c: f32 =
                    -self.readouts[cluster_idx][r_idx].get_calcium() / (self.tau_c as f32);
                self.readouts[cluster_idx][r_idx].update_calcium(delta_c); // adds delta_c to its c_r value

                // If a readout has just fired, update its refractory period
                if self.readouts[cluster_idx][r_idx].get_time_out() > 0 {
                    self.readouts[cluster_idx][r_idx].update_time_out();
                    continue;
                }

                let curr_readout: &Neuron = &self.readouts[cluster_idx][r_idx];
                let mut delta_v: f32 = -curr_readout.get_voltage() / (self.tau_m as f32);
                delta_c = 0.; // since we already updated the calcium earlier

                // Goes (0..(size of the cluster)) in a loop for as many equal sized
                // clusters there are.
                for n_idx in (cluster_idx * count)..((cluster_idx + 1) * count) {
                    // self.neurons[n_idx] is the current liquid neuron in the cluster. 
                    // Each liquid neuron in a cluster is a presynaptic neuron to the readout
                    let pre_syn_neuron: &Neuron = &self.neurons[n_idx];
                    let tau_s1 = pre_syn_neuron.get_second_tau()[0];
                    let tau_s2 = pre_syn_neuron.get_second_tau()[1];

                    let spike_times: &Vec<u32> = pre_syn_neuron.get_spike_times();
                    // Used in learning rule for calculating which neurons spiked into readout.
                    // It does not take into account voltage decay, just the spikes and weights. 
                    let mut change: f32 = 0.; 
                    for spike_time in spike_times.iter() {
                        // Updates voltage and calcium dynamics for readouts
                        let weight: f32 = self.readout_weights[cluster_idx][(r_idx, n_idx % count)];
                        let model_calc: f32 = utils::liq_response(
                            model,
                            curr_t as i32,
                            spike_time.clone() as i32,
                            delay,
                            first_tau,
                            tau_s1,
                            tau_s2,
                        );
                        change += weight * model_calc;
                        delta_v += change;
                    }
                    self.responsible_liq[r_idx][n_idx] += change;
                }
                self.readouts[cluster_idx][r_idx].update_voltage(delta_v);
                self.readouts[cluster_idx][r_idx].update_spike_times(curr_t);

                // Update of calcium
                let r_spike_times = self.readouts[cluster_idx][r_idx].get_spike_times();
                if !r_spike_times.is_empty() {
                    let last_spike_time = r_spike_times[r_spike_times.len() - 1];
                    // If the readout spiked in the current time step, add one to
                    // calcium
                    delta_c += utils::delta(curr_t as i32 - last_spike_time.clone() as i32);
                }
                self.readouts[cluster_idx][r_idx].update_calcium(delta_c);

                // Writing to file for readouts
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
        // Determines the LSM's readout's guess at the end of every epoch by
        // calculating the activity of each function

        // Storing all the c_r for every readout neuron per epoch
        let mut all_c_r: Vec<Vec<f32>> = Vec::new();
        for cluster_idx in 0..self.n_clusters {
            let mut curr_readout_cluster: Vec<f32> = Vec::new();
            for r_idx in 0..self.readouts[cluster_idx].len() {
                curr_readout_cluster.push(self.readouts[cluster_idx][r_idx].get_calcium());
            }
            all_c_r.push(curr_readout_cluster);
        }
        self.readouts_c_r.push(all_c_r); 

        // Find the cluster whose total calcium is largest
        // We sum each calcium in this list which is the number of functions long. 
        let mut output_c_total: Vec<f32> = vec![0.; self.n_readout_clusters];

        for cluster_idx in 0..self.n_clusters {
            for r_idx in 0..self.readouts[cluster_idx].len() {
                let curr_readout: &Neuron = &self.readouts[cluster_idx][r_idx];
                output_c_total[curr_readout.get_cluster()] += curr_readout.get_calcium();
            }
        }

        // The index of every max calcium total (accounting for ties)
        let mut max_val_idx: Vec<usize> = Vec::new();
        let mut max_c_val: f32 = 0.; // Most calcium in any cluster
        let mut max_c_idx: usize = 0; // Index of cluster with most calcium
        for (key, val) in output_c_total.iter().enumerate() {
            // Finds the maximum value and stores their indices (in any case)
            if val == &max_c_val {
                max_val_idx.push(key.clone());
            }
            if val > &max_c_val {
                max_c_val = val.clone();
                max_c_idx = key.clone();
                max_val_idx = vec![key.clone()];
            }
        }

        // If there is a tie, we calculate the readout's voltage and compare those
        let mut r_volts = vec![0.; self.n_readout_clusters];
        if max_val_idx.len() > 1 {
            for r_idx in 0..self.readouts[0].len() {
                for max_idx in max_val_idx.iter() {
                    if &self.readouts[0][r_idx].get_cluster() == max_idx {
                        r_volts[max_idx.clone()] += self.readouts[0][r_idx].get_voltage();
                    }
                }
            }
            let mut max_v_val: f32 = 0.; // Most voltage in any cluster
            let mut max_v_idx: usize = 0; // Index of cluster with most voltage
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
            // If there is no tie
            // Outputs the cluster guess of the cluster with the highest calcium conc.
            if output_c_total[max_c_idx]
                >= self.c_th * (self.readouts[0].len() / self.n_readout_clusters) as f32
            {
                self.outputs.push(CLUSTERS[max_c_idx].to_string());
            // If there is not enough calcium activity, treat it as nothing / ambience
            } else {
                self.outputs.push("nothing".to_string()); // "nothing" is an arbitrary name
            }
        }
    }

    fn update_weights(
        &mut self,
        f2: &mut File,      // Records readout weights
        cluster_idx: usize, // The current cluster we are in
        r_idx: usize,       // The current readout neuron
        delta_weight: f32,  // The weight by which we will potentiate
        instruction: &str,  // Either "potentiate" or "depress"
        contribution: &str, // Either "pos" or "neg" 
    ) {
        // Updates the weights between readout and liquid for a readout function based
        // on some instruction and its activity (in graph_analysis below).
        assert_eq!(
            instruction == "potentiate" || instruction == "depress",
            true
        );
        assert_eq!(contribution == "pos" || contribution == "neg", true);

        // Loop through its pre-syn neurons and increase the weights for them
        let n_count = self.n_total / self.n_clusters; // Number of neurons in 1 cluster
        for n_idx in 0..n_count {
            // Potentiates the weights if the calcium desired is high
            if instruction == "potentiate" { 
                if contribution == "pos" {
                    // If a presynaptic neuron has spiked into this readout
                    if self.responsible_liq[r_idx][n_idx] > 0. {
                        let sum = self.readout_weights[cluster_idx][(r_idx, n_idx)] + delta_weight;
                        // Limiting the readout weights, again by -8 and 8
                        if sum < self.r_weight_max {
                            // Writing to a file that tracks readout weights per time step and epoch
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
                    // If a presynaptic neuron has not spiked into this readout, but it should have
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
            // Depresses the weights if the desired calcium is low
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
        // In one epoch, the LSM checks the current label in labels and sets the c_d's in that cluster to a
        // high number and the other clusters to a low c_d.
        let high_c_d: usize = 10;
        let low_c_d: usize = 0;

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
        self.readouts_c_d.push(all_calciums); 
    }

    fn graph_analysis(&mut self, f2: &mut File, _prev_epoch_accuracy: f32) {
        // Once a time step, during training, we analyze whether the calcium is
        // at the right spot for training. This function only doesn't potentiate or
        // depress if the real calcium of the readout is at the right position. 
        // Note that C_d changes, most probably, every epoch, so the outcome changes
        // often. 
        
        // Below is a model of our training rule, with
        // d - depress, p - potentiate, c_m - margin above / below c_th
        // c_r
        //              |  d    |    N
        // c_th + c_m   |  d    |    p
        // c_th         |  d    |    p
        // c_th - c_m   |  d    |    p
        //              |  N    |    p
        //              |___________________
        //                     c_th      c_d

        let high_d_w = 0.1; 
        let mid_d_w = 0.05; 
        let low_d_w = 0.025;
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
                    // On the top right of the graph, no learning
                // Left of the x axis
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
                    // On the bottom left of the graph, no learning
                }
            }
        }
    }
}

// END OF FILE
