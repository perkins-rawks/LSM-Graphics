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

pub struct LSM {
    // Big structure of clustered up neurons and connections
    clusters: Vec<Vec<Neuron>>,         // A cluster is a set of neurons
    neurons: Vec<Neuron>,               // List of all neurons // has ownership of neurons
    connections: DMatrix<u32>,          // List of all connections (1 if connected, 0 if not)
    n_total: usize,                     // The number of total neurons
    n_inputs: usize,                    // The number of input neurons
    n_clusters: usize,                  // The number of cluster neurons
    e_ratio: f32,                       // 0-1 the ratio of excitatory to inhibitory neurons
    readouts: Vec<Vec<Neuron>>,         // List of all clusters of read out neurons
    readout_weights: Vec<DMatrix<u32>>, // List of all the plastic weights between readouts and reservoir
    input_layer: Vec<Neuron>,           // Set of neurons that read input outside the reservoir
}

impl LSM {
    pub fn new(n_inputs: usize, n_clusters: usize, e_ratio: f32 /*n_readout: f32*/) -> LSM {
        Self {
            clusters: Vec::new(),
            neurons: Vec::new(),
            connections: DMatrix::<u32>::zeros(0, 0), // Starts empty because you can't fill until later
            n_total: n_inputs,                        // at least n_inputs + n_outputs
            n_inputs: n_inputs,
            n_clusters: n_clusters,
            e_ratio: e_ratio,
            readouts: Vec::new(),
            readout_weights: Vec::new(),
            input_layer: Vec::new(),
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
        // The last neuron list in clusters
        for neuron in self.clusters[self.clusters.len() - 1].iter() {
            self.neurons.push(neuron.clone()); // .clone() to avoid moving problems
        }
        self.n_total = self.neurons.len(); // update total neuron count
    }

    fn assign_input(&mut self, neurons: &mut Vec<Neuron>) {
        // Assign input neurons. All neurons are also output now. \\
        // assert_eq!(true, self.n_total >= self.n_inputs);
        assert_eq!(true, self.n_inputs > 0);

        // We want to choose n_inputs neurons from all the neurons created to be
        // inputs.
        let seed = [2; 32];
        let mut rng = StdRng::from_seed(seed);

        // This list has the indices of every unique choice for an input neuron
        // in this struct 'neurons' list.
        let mut liq_idx: Vec<usize> = (0..neurons.len()).collect();

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
            // Find the idx at which it chose that index and remove it
            let idx = liq_idx.iter().position(|&r| r == exc_idx).unwrap();
            liq_idx.remove(idx);
        }
        // The same thing as above for inhibitory
        for _ in 0..n_inh {
            let inh_idx: usize = *liq_idx.choose(&mut rng).unwrap();
            self.neurons[inh_idx].set_nt("inh");

            let idx = liq_idx.iter().position(|&r| r == inh_idx).unwrap();
            liq_idx.remove(idx);
        }
    }

    fn add_cluster(&mut self, add_cluster: Vec<Neuron>) {
        self.clusters.push(add_cluster);
    }

    // Neuron Methods \\

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
        cluster: &str,       // The cluster type
    ) -> Vec<Neuron> {
        // Returns a set of neurons.

        // Makes a \\
        let mut readouts: Vec<Neuron> = Vec::new();
        for idx in 0..n_readouts {
            let x: f32 = loc.x * 2.5;
            let y: f32 = loc.y;
            let z: f32 = loc.z * 2.5;

            let t = Translation3::new(x, y + n_readouts as f32 / 2. - idx as f32, z);

            let neuron: Neuron = Neuron::new(window.add_sphere(radius), "readout", cluster);
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
        let seed = [0; 32];
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

    fn make_readout_connects(&mut self) -> Vec<(Point3<f32>, Point3<f32>, Point3<f32>)> {
        // connect_lines: <(point 1, point 2, color), ... >
        let mut connect_lines: Vec<(Point3<f32>, Point3<f32>, Point3<f32>)> = Vec::new();
        // Number of neurons in one cluster, assuming all clusters are equally sized
        let count = self.n_total / self.n_clusters;
        // For all clusters..
        for cluster_idx in 0..self.n_clusters {
            // Number of readout neurons in a given cluster
            let readout_len = self.readouts[cluster_idx].len();
            // Each column is all the connections for one readout neuron.
            let mut readout_weights = DMatrix::<u32>::zeros(readout_len, count);
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
                    readout_weights[(readout_idx, neuron_idx % count)] = 1;
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
        // Change

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

    fn _comments() {
        /*
        Brainstorming:
        We initialize an LSM with a few clusters.


        We assume each neuron knows:
        o its pre-synaptic neurons
        o its pre-synaptic weights
        Input we will manipulate from a file
        o from txt file
        o goes to the outside input layer neurons
        o For each spiketrain, if it's 1, put in the index into the j list
        o It should be manually connected to some reservoir input neurons
                e.g.
                Say we have outside input layer neuron n1
                There are 5 of these n1 ... n5
                There are 10 reservoir input neurons.
                Then, lets say each outside input is connected to EVERY inside input.
                An example matrix is
                10 x 5 matrix containing just values between -8 and 8

                Lets say each outside input is only connected to 4 inside inputs.
                Then, the matrix is
                10 x 5 matrix with each column having 4 values between -8 and 8,
                other 6 have zeros

        Input neurons (in the reservoir)
        Reservoir business
        Readout neurons


        The first run through the reservoir
        We read the input txt file and put it through special outside input layer
        neurons (not really neurons). What is important is that we fill initial
        weight values and spike times.

        Say we have read the input, and we have a list of spike trains. Then
        for each spike train, we multiply it by some weight (-8 to 8) and those are
        our weights that we will put into the reservoir.
        The spike times will be tracked thusly:
        If the spike reads 1, push the index of that value into the times list.

        ^ Where should this happen? Right after the clusters are made because
        that's when we want the LSM to run.
        */
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
        // Sets the weight of input for one neuron to be a random number from -8 to 8
        let seed = [5; 32];
        let mut rng = StdRng::from_seed(seed);
        // let weight = 16. * rng.gen::<f32>() - 8.;

        for neuron in self.get_neurons().iter_mut() {
            if neuron.get_spec() == &"liq_in".to_string() {
                neuron.set_input_weight(16. * rng.gen::<f32>() - 8.);
            }
        }
    }

    fn make_input_connects(&mut self) {
        // Connects the outside input layer to an input neuron in each cluster
        // in the reservoir

        // Updates a neuron classified as "liq_in"'s input_connect attribute
        // This represents the index of that neuron in LSM input_layer vector

        // We have: o all neurons in reservoir
        //          o all neurons in input layer
        //          o all input neurons in reservoir
        //
        // We want  o to connect one input layer neuron to one reservoir input
        //            neuron in each cluster
        //          o store in the connected neuron the index of this new pre-synaptic
        //            connection

        let count: usize = self.n_total / self.n_clusters;
        for cluster in 0..self.n_clusters {
            let mut input_idx: usize = 0;
            for neuron_idx in count * cluster..count * (cluster + 1) {
                // Connect an input layer neuron to one reservoir input in each cluster
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

    fn delta(&self, n: u32) -> f32 {
        // Dirac delta function
        // A spike helper function
        // It outputs 1 only at 0
        if n == 0 {
            return 1.;
        }
        0.
    }

    fn heaviside(&self, n: u32) -> f32 {
        // Heaviside step function
        // When n < 0, H(n) = 0
        //      n > 0, H(n) = 1
        //      n = 0, H is undefined, but we put 1
        if n < 0 {
            return 0.;
        }
        1.
    }

    fn s(
        &self,
        model: &str,
        curr_t: u32,
        t_spike: u32,
        delay: u32,
        tau_s1: u32,
        tau_s2: u32,
    ) -> f32 {
        if model == "static" {
            return self.delta(curr_t - t_spike - delay);
        } else if model == "first order" {
            let exponent = ((curr_t - t_spike - delay) as f32 * -1.) / (tau_s1 as f32);
            return (1. / tau_s1 as f32)
                * f32::exp(exponent)
                * self.heaviside(curr_t - t_spike - delay);
        } else if model == "second order" {
            // returns two different things
            // one with exponent1 and the other with exponent2
            let exponent1: f32 = ((curr_t - t_spike - delay) as f32 * -1.) / (tau_s1 as f32);
            let exponent2: f32 = ((curr_t - t_spike - delay) as f32 * -1.) / (tau_s2 as f32);
            let denominator: f32 = (tau_s1 - tau_s2) as f32;
            let h = self.heaviside(curr_t - t_spike - delay);
            let part1 = f32::exp(exponent1) * h / denominator;
            let part2 = f32::exp(exponent2) * h / denominator;
            // In equation 21, if we factor out w, we can just
            // subtract the two parts together instead of calculate
            // two separately
            return part1 - part2;
        }
        panic!("Model was chosen incorrectly");
    }

    pub fn run(
        &mut self,
        input: &Vec<Vec<u32>>,
        model: &str,
        delay: u32,
        tau_s1: u32,
        tau_s2: u32,
    ) {
        // Updates voltage connections for all neurons in the LSM.
        // Implementation of Equation 14 in Zhang et al 2015

        /* 2 Different kinds of running:
            o Reservoir input neurons
            o Liquid to liquid
           First time, we want to only run for reservoir inputs.
           New voltage = Old voltage - Old voltage / Time constant for membrane
                        + the sum of weights if the times are close
        */

        self.set_spike_times(input);

        let n_time_steps = input[0].len();

        // For every time step in all the time steps
        for t in 0..n_time_steps {
            // For every post synaptic neuron in the liquid
            for n_idx in 0..self.neurons.len() {
                // If the neuron is a reservoir input
                if self.neurons[n_idx].get_spec() == &"liq_in".to_string() {
                    // We have it already so that the reservoir input knows
                    // which input layer neuron it's connected to
                    // We want to change the current neuron's voltage.

                    // let curr_v = self.neurons[n_idx].get_voltage();
                    // let tau_m  = 4; // for static model
                    // let weight = ; // idk how to get it, we should already
                    // have it somewhere
                    // let t_ij = curr_spike_time; // we should loop through spikes
                    // let d_i = 2; // some constant delay value
                    // let diff_eq: f32 = curr_v + curr_v/tau_m + weight*
                    // delta(t - t_ij - d_i) - spike_time
                    // self.s(t, );
                    // self.neurons[n_idx].set_voltage(diff_eq);
                }

                // The indices of the pre-syn. connections self.neurons[n_idx]
                // has with the rest of the LSM
                let pre_syn_connects = self.neurons[n_idx].get_pre_syn_connects();
                // Loop through all the pre synaptic neuron indices
                for connect_idx_idx in 0..pre_syn_connects.len() {
                    // For each pre synaptic neuron, loop through all its spikes

                    // pre_syn_connects[connect_idx_idx] is the index of one of
                    // the pre synaptic connections of self.neurons[n_idx]

                    // spike_time is a time of one of the spikes from one of the
                    // pre synaptic neurons
                    for spike_time in self.neurons[pre_syn_connects[connect_idx_idx]]
                        .get_spike_times()
                        .iter()
                    {
                        self.s(model, t as u32, spike_time.clone(), delay, tau_s1, tau_s2);
                    }
                }
            }
        }
    }
}
